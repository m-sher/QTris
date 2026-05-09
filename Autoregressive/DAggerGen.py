"""DAgger-style data collection for the autoregressive Tetris policy.

Rolls the trained policy forward in env, queries beam search at each
visited state for the expert action, and records (state, beam_seq)
pairs with the same schema as DataGen.py so transitions accumulate
seamlessly across BC + DAgger rounds in a single dataset.

Distinction from DataGen.py:
- DataGen has BEAM play; (state, beam_action) are co-trajectory.
- DAggerGen has POLICY play; we ASK beam what it would do at each
  visited state but step the env with the policy's choice. This shifts
  the state distribution toward what the policy actually visits in
  deployment, which is the canonical fix for compounding-error in BC
  on long-horizon, fragile-strategy games (Ross & Bagnell, 2010).

Output schema matches DataGen exactly:
  {boards, pieces, b2b_combo_garbage, actions, masks, sample_weights, returns}
where `actions` is BEAM's expert sequence (the supervised label) and
`masks` is built around BEAM's sequence — the env transitions used to
generate `returns` reflect the policy's actual rollout, so subsequent
re-pretraining sees both the corrected labels and the on-policy
trajectory statistics.
"""

import argparse
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras

from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisModel import PolicyModel
from DataGen import _build_mask


def collect_dagger(
    p_model,
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

    def flush(buf, is_death):
        if not buf:
            return
        returns_arr = np.zeros(len(buf), dtype=np.float32)
        last = 0.0
        for t in reversed(range(len(buf))):
            r = buf[t][6]
            d = float(buf[t][7])
            last = r + gamma * last * (1.0 - d)
            returns_arr[t] = last

        if is_death:
            kept_count = (
                len(buf) - death_trim_count
                if len(buf) > death_trim_count else 0
            )
        else:
            kept_count = len(buf)

        for t in range(kept_count):
            board, pieces, bcg, sequence, mask, sample_weight, _r, _d = buf[t]
            transitions.append(
                (board, pieces, bcg, sequence, mask, sample_weight, returns_arr[t])
            )

    for step in range(num_steps):
        obs = time_step.observation
        board = obs["board"].astype(np.float32)
        pieces = obs["pieces"].astype(np.int64)
        bcg = obs["b2b_combo_garbage"].astype(np.float32)
        valid_sequences = obs["sequences"].astype(np.int64)

        # Policy's greedy choice in this state — under valid-sequence
        # masking the model never emits a sequence the env can't replay.
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
            # Treat as terminal: flush as death and reset.
            flush(episode_buf, is_death=True)
            episode_buf = []
            beam_dead += 1
            deaths += 1
            time_step = env.reset()
            continue

        beam_seq = beam_seq.astype(np.int64)

        # Sanity: beam's sequence should appear in valid_sequences.
        # If not (rare; same case as DataGen's `unmatched`), we can't
        # build a clean per-position mask, so skip recording. Still
        # step the env with the policy's choice — that's the DAgger
        # invariant; we want the policy's state distribution.
        beam_in_valid = np.any(
            np.all(valid_sequences == beam_seq[None, :], axis=-1)
        )
        if not beam_in_valid:
            unmatched += 1
            time_step = env._step(policy_seq)
            if time_step.is_last():
                flush(episode_buf, is_death=True)
                episode_buf = []
                deaths += 1
                time_step = env.reset()
            continue

        # Build the per-position validity mask around BEAM's sequence,
        # since beam_seq is what the model will be teacher-forced on.
        mask = _build_mask(beam_seq, valid_sequences, max_len, key_dim)

        # Step env with POLICY's choice (DAgger invariant).
        time_step = env._step(policy_seq)
        reward = float(time_step.reward["total_reward"])
        done = bool(time_step.is_last())

        if not np.array_equal(policy_seq, beam_seq):
            policy_disagrees += 1

        episode_buf.append(
            (board, pieces, bcg, beam_seq, mask, np.float32(search_depth), reward, done)
        )
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if done:
            flush(episode_buf, is_death=True)
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

    flush(episode_buf, is_death=False)
    return transitions, unmatched, beam_dead, deaths, max_b2b, policy_disagrees


def main():
    ap = argparse.ArgumentParser(
        description="DAgger collection for the AR policy. Rolls the "
                    "policy in env, queries beam at each visited "
                    "state, appends (state, beam_action) pairs to "
                    "the existing pretrain dataset.",
    )
    ap.add_argument(
        "--policy-checkpoint", type=str, default="./pretrained_checkpoints/",
        help="Path to a tf.train.CheckpointManager directory containing "
             "the AR PolicyModel checkpoint to roll out.",
    )
    ap.add_argument(
        "--dataset-path", type=str, default="../tetris_expert_dataset_b2b",
        help="Dataset path to APPEND DAgger transitions into. Must "
             "share the schema written by DataGen.py.",
    )
    ap.add_argument("--num-steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=10_000_000,
                    help="Seed offset; defaults to a value far from "
                         "DataGen's typical seeds to avoid replaying "
                         "identical garbage streams.")
    ap.add_argument("--search-depth", type=int, default=8)
    ap.add_argument("--beam-width", type=int, default=96)
    ap.add_argument("--queue-size", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=15)
    ap.add_argument("--key-dim", type=int, default=12)
    ap.add_argument("--piece-dim", type=int, default=8)
    ap.add_argument("--depth", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dropout-rate", type=float, default=0.0)
    ap.add_argument("--max-height", type=int, default=18)
    ap.add_argument("--max-holes", type=int, default=50)
    ap.add_argument("--max-steps-env", type=int, default=9_999_999)
    ap.add_argument("--garbage-chance", type=float, default=0.15)
    ap.add_argument("--garbage-min", type=int, default=1)
    ap.add_argument("--garbage-max", type=int, default=4)
    ap.add_argument("--garbage-push-delay", type=int, default=1)
    ap.add_argument("--num-row-tiers", type=int, default=2)
    ap.add_argument("--death-trim-count", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

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
    # Build the model so the checkpoint's weights have something to
    # restore into. Match the input signature used at pretrain time.
    p_model(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(args.queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(shape=(args.max_len,), dtype=tf.int64),
        )
    )

    p_checkpoint = tf.train.Checkpoint(model=p_model)
    p_checkpoint_manager = tf.train.CheckpointManager(
        p_checkpoint, args.policy_checkpoint, max_to_keep=3,
    )
    if p_checkpoint_manager.latest_checkpoint is None:
        print(
            f"ERROR: no policy checkpoint at {args.policy_checkpoint}. "
            f"Pretrain via Pretrainer.py first.",
            flush=True,
        )
        return 1
    p_checkpoint.restore(
        p_checkpoint_manager.latest_checkpoint
    ).expect_partial()
    print(
        f"Restored AR policy from {p_checkpoint_manager.latest_checkpoint}",
        flush=True,
    )

    existing_count = 0
    existing = None
    if os.path.exists(args.dataset_path):
        try:
            existing_ds = tf.data.Dataset.load(args.dataset_path)
            existing = {
                k: v.numpy()
                for k, v in next(iter(existing_ds.batch(10_000_000))).items()
            }
            existing_count = len(existing["actions"])
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
    actions = np.stack([t[3] for t in new_transitions]).astype(np.int64)
    masks = np.stack([t[4] for t in new_transitions]).astype(bool)
    sample_weights = np.array([t[5] for t in new_transitions]).astype(np.float32)
    returns = np.array([t[6] for t in new_transitions]).astype(np.float32)

    if existing is not None:
        boards = np.concatenate([existing["boards"], boards])
        pieces = np.concatenate([existing["pieces"], pieces])
        bcg = np.concatenate([existing["b2b_combo_garbage"], bcg])
        actions = np.concatenate([existing["actions"], actions])
        masks = np.concatenate([existing["masks"], masks])
        existing_weights = existing.get(
            "sample_weights",
            np.ones(existing_count, dtype=np.float32),
        )
        sample_weights = np.concatenate([existing_weights, sample_weights])
        returns = np.concatenate([existing["returns"], returns])
        print(
            f"Combined: {existing_count} existing + {len(new_transitions)} "
            f"new = {len(actions)} total",
            flush=True,
        )

    if os.path.exists(args.dataset_path):
        shutil.rmtree(args.dataset_path)

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "boards": boards,
            "pieces": pieces,
            "b2b_combo_garbage": bcg,
            "actions": actions,
            "masks": masks,
            "sample_weights": sample_weights,
            "returns": returns,
        }
    )
    dataset.save(args.dataset_path)
    print(f"Saved {len(actions)} transitions to {args.dataset_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
