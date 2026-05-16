"""DAgger-style data collection for the Tetris policy.

Supports both ``ar`` (autoregressive) and ``flat`` policy variants via
``--mode``. In either case the loop is:

  * Roll the trained policy forward in env.
  * At each visited state, query beam search for the expert label.
  * Step the env with the POLICY's choice (the DAgger invariant — this
    is what shifts the visited-state distribution toward what the policy
    actually sees in deployment).
  * Record (state, beam_label) in the dataset.

Output schema mirrors DataGen / DataGenFlat exactly so transitions
accumulate seamlessly across BC + DAgger rounds in a single dataset.

Distinction from DataGen / DataGenFlat:
- DataGen* has BEAM play; (state, beam_action) are co-trajectory.
- DAggerGen has POLICY play; we ASK beam what it would do at each
  visited state but step the env with the policy's choice. This shifts
  the state distribution toward what the policy actually visits in
  deployment, which is the canonical fix for compounding-error in BC
  on long-horizon, fragile-strategy games (Ross & Bagnell, 2010).
"""

import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras

from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Moves import Keys
from qtris.models.ar.model import PolicyModel
from qtris.models.flat.model import FlatPolicyModel
from qtris.data.gen_ar import _build_mask


HARD_DROP_ID = Keys.HARD_DROP


def _make_flat_mask(valid_sequences):
    """1D placement-validity mask (matches DataGenFlat.py)."""
    return np.any(valid_sequences == HARD_DROP_ID, axis=-1)


def _flat_action_idx(beam_seq, valid_sequences):
    """Scalar index into valid_sequences matching beam_seq, or -1."""
    matches = np.all(valid_sequences == beam_seq[None, :], axis=-1)
    if not np.any(matches):
        return -1
    return int(np.argmax(matches))


def collect_dagger(
    p_model,
    mode,
    seed,
    num_steps,
    search_depth,
    beam_width,
    queue_size,
    max_len,
    key_dim,
    max_height,
    max_holes,
    max_steps_env,
    garbage_chance,
    garbage_min,
    garbage_max,
    garbage_push_delay,
    num_row_tiers,
    death_trim_count,
    gamma,
    log_every=1000,
):
    """Roll the policy forward; record beam's expert label per visited state.

    The env is stepped with the POLICY's chosen sequence (greedy decode
    under valid-sequence masking). Beam is queried only to provide the
    label that gets appended to the dataset; beam's choice does NOT
    drive env transitions.

    ``mode`` is ``"ar"`` or ``"flat"`` and controls only the label /
    mask format stored per transition. The rollout itself is identical.
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
        gamma=gamma,
    )

    time_step = env.reset()
    searcher = CB2BSearch()

    transitions = []
    episode_buf = []
    unmatched = 0
    beam_dead = 0
    deaths = 0
    max_b2b = 0
    policy_disagrees = 0

    def flush(buf, death_kind):
        # death_kind is one of:
        #   "beam_fail"    – beam returned action_idx < 0 from a buffered
        #                    state; the tail is genuinely unrecoverable, so
        #                    trim like DataGen.
        #   "policy_fail"  – env signalled terminal while every recorded
        #                    state had a beam-valid label; those tail
        #                    transitions are exactly the high-signal
        #                    correction examples DAgger exists to harvest,
        #                    so keep them all.
        #   "graceful_end" – ran out of num_steps mid-episode. Not a death.
        if not buf:
            return
        returns_arr = np.zeros(len(buf), dtype=np.float32)
        last = 0.0
        for t in reversed(range(len(buf))):
            r = buf[t][6]
            d = float(buf[t][7])
            last = r + gamma * last * (1.0 - d)
            returns_arr[t] = last

        if death_kind == "beam_fail":
            kept_count = (
                len(buf) - death_trim_count
                if len(buf) > death_trim_count else 0
            )
        else:
            kept_count = len(buf)

        for t in range(kept_count):
            board, pieces, bcg, label, mask, sample_weight, _r, _d = buf[t]
            transitions.append(
                (board, pieces, bcg, label, mask, sample_weight, returns_arr[t])
            )

    for step in range(num_steps):
        obs = time_step.observation
        board = obs["board"].astype(np.float32)
        pieces = obs["pieces"].astype(np.int64)
        bcg = obs["b2b_combo_garbage"].astype(np.float32)
        valid_sequences = obs["sequences"].astype(np.int64)

        # Policy's greedy choice in this state — under valid-sequence
        # masking the model never emits a sequence the env can't replay.
        # PolicyModel and FlatPolicyModel share the same predict()
        # signature: both return (selected_sequence, ...) as the first
        # tuple element regardless of internal factorization.
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

        # Beam's expert choice for this state — the supervised label.
        action_idx, beam_seq = searcher.search(
            board=env._board,
            active_piece=env._active_piece.piece_type.value,
            hold_piece=env._hold_piece.value,
            queue=np.array(
                [p.value for p in env._queue], dtype=np.int32
            ),
            b2b=int(env._scorer._b2b),
            combo=int(env._scorer._combo),
            total_garbage=int(env._get_total_garbage()),
            garbage_push_delay=env._garbage_push_delay,
            search_depth=search_depth,
            beam_width=beam_width,
            max_len=max_len,
        )

        if action_idx < 0:
            # Beam can't label this state (no valid placements).
            # Treat as terminal: flush as beam-fail death and reset.
            flush(episode_buf, death_kind="beam_fail")
            episode_buf = []
            beam_dead += 1
            deaths += 1
            time_step = env.reset()
            continue

        beam_seq = beam_seq.astype(np.int64)

        # Build the per-mode label/mask around BEAM's sequence (the
        # supervised target). For AR we record the full key sequence
        # plus the per-position next-token mask; for flat we collapse
        # to the scalar placement index plus the 1D HARD_DROP-presence
        # mask. If beam's sequence isn't present in valid_sequences
        # (rare; same case as DataGen's `unmatched`), we can't build a
        # clean label, so skip recording. Still step the env with the
        # policy's choice — that's the DAgger invariant.
        if mode == "ar":
            beam_in_valid = np.any(
                np.all(valid_sequences == beam_seq[None, :], axis=-1)
            )
            if not beam_in_valid:
                unmatched += 1
                time_step = env._step(policy_seq)
                if time_step.is_last():
                    flush(episode_buf, death_kind="policy_fail")
                    episode_buf = []
                    deaths += 1
                    time_step = env.reset()
                continue
            label = beam_seq
            mask = _build_mask(beam_seq, valid_sequences, max_len, key_dim)
        else:  # flat
            flat_idx = _flat_action_idx(beam_seq, valid_sequences)
            if flat_idx < 0:
                unmatched += 1
                time_step = env._step(policy_seq)
                if time_step.is_last():
                    flush(episode_buf, death_kind="policy_fail")
                    episode_buf = []
                    deaths += 1
                    time_step = env.reset()
                continue
            label = np.int64(flat_idx)
            mask = _make_flat_mask(valid_sequences)

        # Step env with POLICY's choice (DAgger invariant).
        time_step = env._step(policy_seq)
        reward = float(time_step.reward["total_reward"])
        done = bool(time_step.is_last())

        if not np.array_equal(policy_seq, beam_seq):
            policy_disagrees += 1

        episode_buf.append(
            (board, pieces, bcg, label, mask, np.float32(search_depth), reward, done)
        )
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if done:
            flush(episode_buf, death_kind="policy_fail")
            episode_buf = []
            deaths += 1
            time_step = env.reset()

        if (step + 1) % log_every == 0:
            disagree_rate = (
                100.0 * policy_disagrees / (step + 1) if step + 1 > 0 else 0.0
            )
            print(
                f"Step {step + 1}/{num_steps} | "
                f"transitions={len(transitions)} unmatched={unmatched} "
                f"beam_dead={beam_dead} deaths={deaths} "
                f"max_b2b={max_b2b} "
                f"policy≠beam={policy_disagrees} ({disagree_rate:.1f}%)",
                flush=True,
            )

    flush(episode_buf, death_kind="graceful_end")
    return transitions, unmatched, beam_dead, deaths, max_b2b, policy_disagrees


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
    args = SimpleNamespace(
        mode=cli_args.family,
        policy_checkpoint=getattr(cli_args, "policy_checkpoint", None),
        dataset_path=getattr(cli_args, "output", None),
        num_steps=cli_args.steps,
        seed=getattr(cli_args, "seed", 10_000_000),
        search_depth=8,
        beam_width=96,
        queue_size=5,
        max_len=15,
        key_dim=12,
        piece_dim=8,
        depth=64,
        num_heads=4,
        num_layers=4,
        dropout_rate=0.0,
        max_height=18,
        max_holes=50,
        max_steps_env=9_999_999,
        garbage_chance=0.15,
        garbage_min=1,
        garbage_max=4,
        garbage_push_delay=1,
        num_row_tiers=2,
        death_trim_count=20,
        gamma=0.99,
        log_every=1000,
    )

    mode_defaults = {
        "ar": {
            "policy_checkpoint": "checkpoints/ar_pretrained_policy",
            "dataset_path": "datasets/tetris_expert_dataset_b2b",
            "label_key": "actions",
            "mask_key": "masks",
            "build_model": _build_ar_model,
        },
        "flat": {
            "policy_checkpoint": "checkpoints/flat_pretrained_policy",
            "dataset_path": "datasets/tetris_expert_dataset_flat",
            "label_key": "action_indices",
            "mask_key": "valid_masks",
            "build_model": _build_flat_model,
        },
    }
    cfg = mode_defaults[args.mode]
    policy_checkpoint = args.policy_checkpoint or cfg["policy_checkpoint"]
    dataset_path = args.dataset_path or cfg["dataset_path"]
    label_key = cfg["label_key"]
    mask_key = cfg["mask_key"]

    p_model = cfg["build_model"](args)

    p_checkpoint = tf.train.Checkpoint(model=p_model)
    p_checkpoint_manager = tf.train.CheckpointManager(
        p_checkpoint, policy_checkpoint, max_to_keep=3,
    )
    if p_checkpoint_manager.latest_checkpoint is None:
        print(
            f"ERROR: no policy checkpoint at {policy_checkpoint}. "
            f"Pretrain via {'Pretrainer.py' if args.mode == 'ar' else 'PretrainFlat.py'} first.",
            flush=True,
        )
        return 1
    p_checkpoint.restore(
        p_checkpoint_manager.latest_checkpoint
    ).expect_partial()
    print(
        f"Restored {args.mode} policy from {p_checkpoint_manager.latest_checkpoint}",
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
            existing_count = len(existing[label_key])
            if "returns" not in existing:
                print(
                    "Existing dataset has no `returns` field — starting "
                    "fresh (value pretraining requires returns).",
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

    new_transitions, unmatched, beam_dead, deaths, max_b2b, policy_disagrees = (
        collect_dagger(
            p_model=p_model,
            mode=args.mode,
            seed=args.seed + existing_count,
            num_steps=args.num_steps,
            search_depth=args.search_depth,
            beam_width=args.beam_width,
            queue_size=args.queue_size,
            max_len=args.max_len,
            key_dim=args.key_dim,
            max_height=args.max_height,
            max_holes=args.max_holes,
            max_steps_env=args.max_steps_env,
            garbage_chance=args.garbage_chance,
            garbage_min=args.garbage_min,
            garbage_max=args.garbage_max,
            garbage_push_delay=args.garbage_push_delay,
            num_row_tiers=args.num_row_tiers,
            death_trim_count=args.death_trim_count,
            gamma=args.gamma,
            log_every=args.log_every,
        )
    )

    if not new_transitions:
        print("No new transitions collected; dataset unchanged.", flush=True)
        return 0

    print(
        f"Collected {len(new_transitions)} DAgger transitions | "
        f"unmatched: {unmatched} | beam_dead: {beam_dead} | "
        f"deaths: {deaths} | max_b2b: {max_b2b} | "
        f"policy≠beam: {policy_disagrees}",
        flush=True,
    )

    boards = np.stack([t[0] for t in new_transitions]).astype(np.float32)
    pieces = np.stack([t[1] for t in new_transitions]).astype(np.int64)
    bcg = np.stack([t[2] for t in new_transitions]).astype(np.float32)
    if args.mode == "ar":
        labels = np.stack([t[3] for t in new_transitions]).astype(np.int64)
        masks = np.stack([t[4] for t in new_transitions]).astype(bool)
    else:
        labels = np.array([t[3] for t in new_transitions]).astype(np.int64)
        masks = np.stack([t[4] for t in new_transitions]).astype(bool)
    sample_weights = np.array([t[5] for t in new_transitions]).astype(np.float32)
    returns = np.array([t[6] for t in new_transitions]).astype(np.float32)

    if existing is not None:
        boards = np.concatenate([existing["boards"], boards])
        pieces = np.concatenate([existing["pieces"], pieces])
        bcg = np.concatenate([existing["b2b_combo_garbage"], bcg])
        labels = np.concatenate([existing[label_key], labels])
        masks = np.concatenate([existing[mask_key], masks])
        existing_weights = existing.get(
            "sample_weights",
            np.ones(existing_count, dtype=np.float32),
        )
        sample_weights = np.concatenate([existing_weights, sample_weights])
        returns = np.concatenate([existing["returns"], returns])
        print(
            f"Combined: {existing_count} existing + {len(new_transitions)} "
            f"new = {len(labels)} total",
            flush=True,
        )

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "boards": boards,
            "pieces": pieces,
            "b2b_combo_garbage": bcg,
            label_key: labels,
            mask_key: masks,
            "sample_weights": sample_weights,
            "returns": returns,
        }
    )
    dataset.save(dataset_path)
    print(f"Saved {len(labels)} transitions to {dataset_path}", flush=True)
    return 0


