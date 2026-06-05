from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from qtris.config import DataGenConfig
from qtris.data.placement_features import (
    CANDIDATE_CAPACITY,
    PLACEMENT_FEATURE_DIM,
    build_placement_target,
)
import os
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm


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
    headless=False,
    log_every=1000,
):
    """Single-env sequential collection of candidate-ranking placement targets.

    For each position the beam search scores every reachable root placement; the
    target is a 128-slot pack of fusion-style placement vectors (64 no-hold + 64
    hold) plus their raw search scores. The env advances by playing the best move.
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
    deaths = 0
    max_b2b = 0

    pbar = tqdm(
        range(num_steps), disable=headless, desc="datagen placement", unit="step"
    )
    for step in pbar:
        obs = time_step.observation
        board = obs["board"].astype(np.float32)
        pieces = obs["pieces"].astype(np.int64)
        bcg = obs["b2b_combo_garbage"].astype(np.float32)

        queue = np.array([p.value for p in env._queue], dtype=np.int32)
        best_action, best_seq, cand_actions, cand_scores, _cand_seqs, cand_rows = (
            searcher.search_with_scores(
                board=env._board,
                active_piece=env._active_piece.piece_type.value,
                hold_piece=env._hold_piece.value,
                queue=queue,
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
            deaths += 1
            time_step = env.reset()
            continue

        row_norm = env._board.shape[0] - 1
        placements, scores = build_placement_target(
            cand_actions,
            cand_scores,
            cand_rows,
            active_piece=env._active_piece.piece_type.value,
            hold_piece=env._hold_piece.value,
            queue0=int(queue[0]),
            row_norm=row_norm,
        )
        transitions.append((board, pieces, bcg, placements, scores))

        time_step = env._step(best_seq.astype(np.int64))
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if time_step.is_last():
            deaths += 1
            time_step = env.reset()

        if (step + 1) % log_every == 0:
            stats = f"transitions={len(transitions)} deaths={deaths} max_b2b={max_b2b}"
            if headless:
                print(f"Step {step + 1}/{num_steps} | {stats}", flush=True)
            else:
                pbar.set_postfix_str(stats)

    return transitions, deaths, max_b2b


def main(args):
    dataset_path = (
        str(args.output) if args.output else "datasets/tetris_oracle_placement"
    )
    num_steps = args.steps
    seed = getattr(args, "seed", 0)

    datagen_cfg = DataGenConfig()
    search_depth = datagen_cfg.search_depth
    beam_width = datagen_cfg.beam_width
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
            cp = existing.get("cand_placements")
            if cp is None or cp.shape[1:] != (
                CANDIDATE_CAPACITY,
                PLACEMENT_FEATURE_DIM,
            ):
                print(
                    "Existing dataset is an older schema (not 128-slot "
                    "`cand_placements`) - starting fresh.",
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

    new_transitions, deaths, max_b2b = collect(
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
        headless=getattr(args, "headless", False),
    )

    print(
        f"Collected {len(new_transitions)} transitions | "
        f"deaths: {deaths} | max_b2b: {max_b2b}",
        flush=True,
    )

    boards = np.stack([t[0] for t in new_transitions]).astype(np.float32)
    pieces = np.stack([t[1] for t in new_transitions]).astype(np.int64)
    bcg = np.stack([t[2] for t in new_transitions]).astype(np.float32)
    cand_placements = np.stack([t[3] for t in new_transitions]).astype(np.float32)
    cand_scores = np.stack([t[4] for t in new_transitions]).astype(np.float32)

    if existing is not None:
        boards = np.concatenate([existing["boards"], boards])
        pieces = np.concatenate([existing["pieces"], pieces])
        bcg = np.concatenate([existing["b2b_combo_garbage"], bcg])
        cand_placements = np.concatenate([existing["cand_placements"], cand_placements])
        cand_scores = np.concatenate([existing["cand_scores"], cand_scores])
        print(
            f"Combined: {existing_count} existing + {len(new_transitions)} new = "
            f"{len(cand_scores)} total",
            flush=True,
        )

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "boards": boards,
            "pieces": pieces,
            "b2b_combo_garbage": bcg,
            "cand_placements": cand_placements,
            "cand_scores": cand_scores,
        }
    )
    dataset.save(dataset_path)
    print(f"Saved {len(cand_scores)} transitions to {dataset_path}", flush=True)
