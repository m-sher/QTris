from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Moves import Keys
import os
import shutil
import numpy as np
import tensorflow as tf

HARD_DROP_ID = Keys.HARD_DROP


def collect(
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
    death_trim_count,
    gamma,
    log_every=1000,
):
    """Single-env sequential collection. Runs num_steps total transitions,
    resetting on death. Per-episode discounted returns are computed in flush
    so kept transitions carry the discounted upcoming death penalty."""
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
    deaths = 0
    max_b2b = 0

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
            kept_count = len(buf) - death_trim_count if len(buf) > death_trim_count else 0
        else:
            kept_count = len(buf)

        for t in range(kept_count):
            board, pieces, bcg, action_idx, valid_mask, sample_weight, _r, _d = buf[t]
            transitions.append(
                (board, pieces, bcg, action_idx, valid_mask, sample_weight, returns_arr[t])
            )

    for step in range(num_steps):
        obs = time_step.observation
        board = obs["board"].astype(np.float32)
        pieces = obs["pieces"].astype(np.int64)
        bcg = obs["b2b_combo_garbage"].astype(np.float32)
        valid_sequences = obs["sequences"].astype(np.int64)

        action_idx, sequence = searcher.search(
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
            flush(episode_buf, is_death=True)
            episode_buf = []
            deaths += 1
            time_step = env.reset()
            continue

        sequence = sequence.astype(np.int64)
        matches = np.all(valid_sequences == sequence[None, :], axis=-1)

        if not np.any(matches):
            unmatched += 1
            time_step = env._step(sequence)
            if time_step.is_last():
                flush(episode_buf, is_death=True)
                episode_buf = []
                deaths += 1
                time_step = env.reset()
            continue

        flat_action_idx = int(np.argmax(matches))
        valid_mask = np.any(valid_sequences == HARD_DROP_ID, axis=-1)

        time_step = env._step(sequence)
        reward = float(time_step.reward["total_reward"])
        done = bool(time_step.is_last())

        episode_buf.append(
            (
                board,
                pieces,
                bcg,
                flat_action_idx,
                valid_mask,
                np.float32(search_depth),
                reward,
                done,
            )
        )
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if done:
            flush(episode_buf, is_death=True)
            episode_buf = []
            deaths += 1
            time_step = env.reset()

        if (step + 1) % log_every == 0:
            print(
                f"Step {step + 1}/{num_steps} | "
                f"transitions={len(transitions)} unmatched={unmatched} "
                f"deaths={deaths} max_b2b={max_b2b}",
                flush=True,
            )

    flush(episode_buf, is_death=False)
    return transitions, unmatched, deaths, max_b2b


def main():
    dataset_path = "../tetris_expert_dataset_flat"
    num_steps = 200_000
    seed = 0

    search_depth = 7
    beam_width = 96
    queue_size = 5
    max_len = 15
    max_height = 18
    max_holes = 50
    max_steps_env = 9999999
    garbage_chance = 0.15
    garbage_min = 1
    garbage_max = 4
    garbage_push_delay = 1
    num_row_tiers = 2
    death_trim_count = 20
    gamma = 0.99

    existing_count = 0
    existing = None
    if os.path.exists(dataset_path):
        try:
            existing_ds = tf.data.Dataset.load(dataset_path)
            existing = {
                k: v.numpy()
                for k, v in next(iter(existing_ds.batch(10_000_000))).items()
            }
            existing_count = len(existing["action_indices"])
            if "returns" not in existing:
                print(
                    "Existing dataset has no `returns` field — starting fresh "
                    "(value pretraining requires returns).",
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
        f"Collecting {num_steps} steps in single env (seed offset {existing_count})...",
        flush=True,
    )

    new_transitions, unmatched, deaths, max_b2b = collect(
        seed=seed + existing_count,
        num_steps=num_steps,
        search_depth=search_depth,
        beam_width=beam_width,
        queue_size=queue_size,
        max_len=max_len,
        max_height=max_height,
        max_holes=max_holes,
        max_steps_env=max_steps_env,
        garbage_chance=garbage_chance,
        garbage_min=garbage_min,
        garbage_max=garbage_max,
        garbage_push_delay=garbage_push_delay,
        num_row_tiers=num_row_tiers,
        death_trim_count=death_trim_count,
        gamma=gamma,
    )

    print(
        f"Collected {len(new_transitions)} transitions | "
        f"unmatched: {unmatched} | deaths: {deaths} | max_b2b: {max_b2b}",
        flush=True,
    )

    boards = np.stack([t[0] for t in new_transitions]).astype(np.float32)
    pieces = np.stack([t[1] for t in new_transitions]).astype(np.int64)
    bcg = np.stack([t[2] for t in new_transitions]).astype(np.float32)
    action_indices = np.array([t[3] for t in new_transitions]).astype(np.int64)
    valid_masks = np.stack([t[4] for t in new_transitions]).astype(bool)
    sample_weights = np.array([t[5] for t in new_transitions]).astype(np.float32)
    returns = np.array([t[6] for t in new_transitions]).astype(np.float32)

    if existing is not None:
        boards = np.concatenate([existing["boards"], boards])
        pieces = np.concatenate([existing["pieces"], pieces])
        bcg = np.concatenate([existing["b2b_combo_garbage"], bcg])
        action_indices = np.concatenate([existing["action_indices"], action_indices])
        valid_masks = np.concatenate([existing["valid_masks"], valid_masks])
        existing_weights = existing.get(
            "sample_weights",
            np.ones(existing_count, dtype=np.float32),
        )
        sample_weights = np.concatenate([existing_weights, sample_weights])
        returns = np.concatenate([existing["returns"], returns])
        print(
            f"Combined: {existing_count} existing + {len(new_transitions)} new = "
            f"{len(action_indices)} total",
            flush=True,
        )

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "boards": boards,
            "pieces": pieces,
            "b2b_combo_garbage": bcg,
            "action_indices": action_indices,
            "valid_masks": valid_masks,
            "sample_weights": sample_weights,
            "returns": returns,
        }
    )
    dataset.save(dataset_path)
    print(f"Saved {len(action_indices)} transitions to {dataset_path}", flush=True)


if __name__ == "__main__":
    main()
