"""DAgger-style data collection for the Tetris policy.

The loop rolls a trained policy forward in env and labels each visited state
with the beam search's dense action-indexed target (the same schema gen_ar
stores):

  * Roll the trained policy forward in env (greedy decode under valid-sequence
    masking).
  * At each visited state, query the beam search for the per-action scores +
    key-sequences and scatter them into a dense action-indexed target.
  * Step the env with the POLICY's choice (the DAgger invariant) - this shifts
    the visited-state distribution toward what the policy sees in deployment,
    the canonical fix for compounding error in BC on long-horizon games
    (Ross & Bagnell, 2010).
  * Record (state, dense search target).

The output schema mirrors gen_ar exactly, so DAgger transitions accumulate into
the same dataset across BC + DAgger rounds. ``family`` selects which policy
checkpoint drives the rollout; the stored target is identical for both.
"""

import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras

from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from qtris.models.ar.model import PolicyModel
from qtris.models.flat.model import FlatPolicyModel
from qtris.data.gen_ar import NUM_ACTIONS, dense_target


def collect_dagger(
    p_model,
    seed,
    num_steps,
    search_depth,
    beam_width,
    queue_size,
    max_len,
    max_height,
    max_holes,
    max_steps_env,
    garbage_chance,
    garbage_min,
    garbage_max,
    garbage_push_delay,
    num_row_tiers,
    log_every=1000,
):
    """Roll the policy forward; label each visited state with the search target.

    The env is stepped with the policy's greedy choice (under valid-sequence
    masking) - the DAgger invariant that shifts the visited-state distribution
    toward deployment. The beam search labels each state with the same dense
    action-indexed target as gen_ar; the search's own best move does not drive
    transitions.
    """
    env = PyTetrisEnv(
        queue_size=queue_size,
        max_holes=max_holes,
        max_height=max_height,
        max_steps=max_steps_env,
        max_len=max_len,
        pathfinding=True,
        seed=seed,
        idx=0,
        garbage_chance=garbage_chance,
        garbage_min=garbage_min,
        garbage_max=garbage_max,
        garbage_push_delay=garbage_push_delay,
        auto_push_garbage=True,
        auto_fill_queue=True,
        num_row_tiers=num_row_tiers,
    )

    time_step = env.reset()
    searcher = CB2BSearch()

    transitions = []
    beam_dead = 0
    deaths = 0
    max_b2b = 0
    policy_disagrees = 0

    for step in range(num_steps):
        obs = time_step.observation
        board = obs["board"].astype(np.float32)
        pieces = obs["pieces"].astype(np.int64)
        bcg = obs["b2b_combo_garbage"].astype(np.float32)
        valid_sequences = obs["sequences"].astype(np.int64)

        # Policy's greedy choice under valid-sequence masking (drives the env).
        # PolicyModel and FlatPolicyModel share predict(): both return the
        # selected key-sequence as the first tuple element.
        b_in = tf.constant(board[None, ...], dtype=tf.float32)
        p_in = tf.constant(pieces[None, ...], dtype=tf.int64)
        g_in = tf.constant(bcg[None, ...], dtype=tf.float32)
        vs_in = tf.constant(valid_sequences[None, ...], dtype=tf.int64)
        policy_seq, _, _, _ = p_model.predict(
            (b_in, p_in, g_in),
            greedy=True,
            valid_sequences=vs_in,
            temperature=1.0,
        )
        policy_seq = policy_seq.numpy()[0].astype(np.int64)

        # Beam's dense action-indexed target for this state (the label).
        best_action, best_seq, cand_actions, cand_scores, cand_seqs = (
            searcher.search_with_scores(
                board=env._board,
                active_piece=env._active_piece.piece_type.value,
                hold_piece=env._hold_piece.value,
                queue=np.array([p.value for p in env._queue], dtype=np.int32),
                b2b=int(env._scorer._b2b),
                combo=int(env._scorer._combo),
                total_garbage=int(env._get_total_garbage()),
                garbage_push_delay=env._garbage_push_delay,
                search_depth=search_depth,
                beam_width=beam_width,
                max_len=max_len,
            )
        )

        if best_action < 0 or len(cand_scores) == 0:
            # No labelable placement - treat as terminal and reset.
            beam_dead += 1
            deaths += 1
            time_step = env.reset()
            continue

        seqs, scores = dense_target(cand_actions, cand_scores, cand_seqs, max_len)
        transitions.append((board, pieces, bcg, seqs, scores))

        # Step env with the POLICY's choice (DAgger invariant).
        time_step = env._step(policy_seq)
        if not np.array_equal(policy_seq, best_seq):
            policy_disagrees += 1
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if time_step.is_last():
            deaths += 1
            time_step = env.reset()

        if (step + 1) % log_every == 0:
            disagree_rate = 100.0 * policy_disagrees / (step + 1)
            print(
                f"Step {step + 1}/{num_steps} | transitions={len(transitions)} "
                f"beam_dead={beam_dead} deaths={deaths} max_b2b={max_b2b} "
                f"policy≠beam={policy_disagrees} ({disagree_rate:.1f}%)",
                flush=True,
            )

    return transitions, beam_dead, deaths, max_b2b, policy_disagrees


def _build_ar_model(args):
    p_model = PolicyModel(
        batch_size=1,
        piece_dim=args.piece_dim,
        key_dim=args.key_dim,
        depth=args.depth,
        max_len=args.max_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        output_dim=args.key_dim,
    )
    p_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(args.queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(shape=(args.max_len,), dtype=tf.int64),
        )
    )
    return p_model


def _build_flat_model(args):
    num_sequences = 160 * args.num_row_tiers
    p_model = FlatPolicyModel(
        batch_size=1,
        piece_dim=args.piece_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        num_sequences=num_sequences,
    )
    p_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(args.queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
        )
    )
    return p_model


def main(cli_args):
    from types import SimpleNamespace
    from qtris.config import ModelConfig, EnvConfig

    m = ModelConfig()
    e = EnvConfig()
    args = SimpleNamespace(
        family=cli_args.family,
        policy_checkpoint=getattr(cli_args, "policy_checkpoint", None),
        dataset_path=getattr(cli_args, "output", None),
        num_steps=cli_args.steps,
        seed=getattr(cli_args, "seed", 10_000_000),
        search_depth=16,
        beam_width=200,
        queue_size=m.queue_size,
        max_len=m.max_len,
        key_dim=m.key_dim,
        piece_dim=m.piece_dim,
        depth=m.depth,
        num_heads=m.num_heads,
        num_layers=m.num_layers,
        dropout_rate=m.dropout_rate,
        max_height=e.max_height,
        max_holes=e.max_holes,
        max_steps_env=9_999_999,
        garbage_chance=e.garbage_chance,
        garbage_min=e.garbage_min,
        garbage_max=e.garbage_max,
        garbage_push_delay=e.garbage_push_delay,
        num_row_tiers=m.num_row_tiers,
        log_every=1000,
    )

    family_defaults = {
        "ar": {
            "policy_checkpoint": "checkpoints/ar_pretrained_policy",
            "dataset_path": "datasets/tetris_expert_dataset_b2b",
            "build_model": _build_ar_model,
        },
        "flat": {
            "policy_checkpoint": "checkpoints/flat_pretrained_policy",
            "dataset_path": "datasets/tetris_expert_dataset_flat",
            "build_model": _build_flat_model,
        },
    }
    cfg = family_defaults[args.family]
    policy_checkpoint = str(args.policy_checkpoint or cfg["policy_checkpoint"])
    dataset_path = str(args.dataset_path) if args.dataset_path else cfg["dataset_path"]

    p_model = cfg["build_model"](args)

    p_checkpoint = tf.train.Checkpoint(model=p_model)
    p_checkpoint_manager = tf.train.CheckpointManager(
        p_checkpoint,
        policy_checkpoint,
        max_to_keep=3,
    )
    if p_checkpoint_manager.latest_checkpoint is None:
        print(f"ERROR: no policy checkpoint at {policy_checkpoint}.", flush=True)
        return 1
    p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint).expect_partial()
    print(
        f"Restored {args.family} policy from {p_checkpoint_manager.latest_checkpoint}",
        flush=True,
    )

    existing_count = 0
    existing = None
    if os.path.exists(dataset_path):
        try:
            existing_ds = tf.data.Dataset.load(dataset_path)
            existing = {
                k: v.numpy()
                for k, v in next(iter(existing_ds.batch(10_000_000))).items()
            }
            existing_count = len(existing.get("cand_scores", []))
            if (
                "cand_scores" not in existing
                or existing["cand_scores"].shape[1] != NUM_ACTIONS
            ):
                print(
                    "Existing dataset is an older schema (not dense 320-action "
                    "`cand_scores`) - starting fresh for search-aligned targets.",
                    flush=True,
                )
                existing = None
                existing_count = 0
            else:
                print(
                    f"Found existing dataset with {existing_count} transitions",
                    flush=True,
                )
        except Exception:
            print("Existing dataset load failed, starting fresh", flush=True)

    print(
        f"Collecting {args.num_steps} DAgger steps "
        f"(seed offset {args.seed + existing_count})...",
        flush=True,
    )

    new_transitions, beam_dead, deaths, max_b2b, policy_disagrees = collect_dagger(
        p_model=p_model,
        seed=args.seed + existing_count,
        num_steps=args.num_steps,
        search_depth=args.search_depth,
        beam_width=args.beam_width,
        queue_size=args.queue_size,
        max_len=args.max_len,
        max_height=args.max_height,
        max_holes=args.max_holes,
        max_steps_env=args.max_steps_env,
        garbage_chance=args.garbage_chance,
        garbage_min=args.garbage_min,
        garbage_max=args.garbage_max,
        garbage_push_delay=args.garbage_push_delay,
        num_row_tiers=args.num_row_tiers,
        log_every=args.log_every,
    )

    if not new_transitions:
        print("No new transitions collected; dataset unchanged.", flush=True)
        return 0

    print(
        f"Collected {len(new_transitions)} DAgger transitions | "
        f"beam_dead: {beam_dead} | deaths: {deaths} | max_b2b: {max_b2b} | "
        f"policy≠beam: {policy_disagrees}",
        flush=True,
    )

    boards = np.stack([t[0] for t in new_transitions]).astype(np.float32)
    pieces = np.stack([t[1] for t in new_transitions]).astype(np.int64)
    bcg = np.stack([t[2] for t in new_transitions]).astype(np.float32)
    cand_sequences = np.stack([t[3] for t in new_transitions]).astype(np.int8)
    cand_scores = np.stack([t[4] for t in new_transitions]).astype(np.float32)

    if existing is not None:
        boards = np.concatenate([existing["boards"], boards])
        pieces = np.concatenate([existing["pieces"], pieces])
        bcg = np.concatenate([existing["b2b_combo_garbage"], bcg])
        cand_sequences = np.concatenate([existing["cand_sequences"], cand_sequences])
        cand_scores = np.concatenate([existing["cand_scores"], cand_scores])
        print(
            f"Combined: {existing_count} existing + {len(new_transitions)} "
            f"new = {len(cand_scores)} total",
            flush=True,
        )

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "boards": boards,
            "pieces": pieces,
            "b2b_combo_garbage": bcg,
            "cand_sequences": cand_sequences,
            "cand_scores": cand_scores,
        }
    )
    dataset.save(dataset_path)
    print(f"Saved {len(cand_scores)} transitions to {dataset_path}", flush=True)
    return 0
