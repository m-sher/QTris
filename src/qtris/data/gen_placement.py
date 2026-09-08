from TetrisEnv.Moves import Keys
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from qtris.config import DataGenConfig
from qtris.data.placement_features import (
    CANDIDATE_CAPACITY,
    PLACEMENT_FEATURE_DIM,
    build_placement_target,
)
import functools
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
    hold) plus their root-ranking scores and the played line's value. The env advances
    by playing the best move.
    """
    env = PyTetrisEnv(
        queue_size=queue_size,
        max_holes=max_holes,
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
    total_attack = 0.0
    pieces_placed = 0

    pbar = tqdm(
        range(num_steps), disable=headless, desc="datagen placement", unit="step"
    )
    for step in pbar:
        obs = time_step.observation
        board = obs["board"].astype(np.float32)
        pieces = obs["pieces"].astype(np.int64)
        bcg = obs["b2b_combo_garbage"].astype(np.float32)

        queue = np.array([p.value for p in env._queue], dtype=np.int32)
        best_action, best_seq, cand_actions, cand_scores, _seqs, cand_rows, value = (
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
        transitions.append((board, pieces, bcg, placements, scores, value))

        time_step = env._step(best_seq.astype(np.int64))
        total_attack += float(time_step.reward["attack"])
        pieces_placed += 1
        max_b2b = max(max_b2b, int(env._scorer._b2b))

        if time_step.is_last():
            deaths += 1
            time_step = env.reset()

        if (step + 1) % log_every == 0:
            app = total_attack / max(pieces_placed, 1)
            stats = (
                f"transitions={len(transitions)} deaths={deaths} "
                f"max_b2b={max_b2b} app={app:.3f}"
            )
            if headless:
                print(f"Step {step + 1}/{num_steps} | {stats}", flush=True)
            else:
                pbar.set_postfix_str(stats)

    app = total_attack / max(pieces_placed, 1)
    return transitions, deaths, max_b2b, app


def collect_batched(
    seed,
    num_steps,
    search_depth,
    beam_width,
    queue_size,
    max_len,
    max_holes,
    max_steps_env,
    garbage_chance,
    garbage_min,
    garbage_max,
    garbage_push_delay,
    num_row_tiers,
    batch,
    headless=False,
    log_every=1000,
):
    """Collect the same targets as `collect`, from `batch` envs run in lockstep.

    One batched search per round serves every env. `num_steps` counts transitions,
    not rounds; the last round is truncated.
    """
    from qtris.search.placement_search import descriptor_key_sequence
    from teacher.api import GpuTeacher, placement_of

    # Each env resets to its own seed + 1 on death, so stride the seeds by more than
    # the resets any one env can make.
    envs = [
        PyTetrisEnv(
            queue_size=queue_size,
            max_holes=max_holes,
            max_steps=max_steps_env,
            max_len=max_len,
            pathfinding=True,
            seed=seed + i * (num_steps + 1),
            idx=i,
            garbage_chance=garbage_chance,
            garbage_min=garbage_min,
            garbage_max=garbage_max,
            garbage_push_delay=garbage_push_delay,
            auto_push_garbage=True,
            auto_fill_queue=True,
            num_row_tiers=num_row_tiers,
        )
        for i in range(batch)
    ]
    steps = [env.reset() for env in envs]
    teacher = GpuTeacher(
        max_batch=batch, width=beam_width, depth=search_depth, queue_len=queue_size
    )

    transitions = []
    deaths = 0
    max_b2b = 0
    total_attack = 0.0
    pieces_placed = 0
    forced = np.full(max_len, Keys.PAD, dtype=np.int64)
    forced[0], forced[1] = Keys.START, Keys.HARD_DROP
    fallbacks = 0

    pbar = tqdm(
        total=num_steps, disable=headless, desc="datagen placement", unit="step"
    )
    while len(transitions) < num_steps:
        obs = [ts.observation for ts in steps]
        queues = np.stack(
            [np.array([p.value for p in e._queue], dtype=np.int32) for e in envs]
        )
        result = teacher.search_batch(
            np.stack([e._board for e in envs]),
            np.array([e._active_piece.piece_type.value for e in envs], dtype=np.int32),
            np.array([e._hold_piece.value for e in envs], dtype=np.int32),
            queues,
            queue_size,
            np.array([int(e._scorer._b2b) for e in envs], dtype=np.int32),
            np.array([int(e._scorer._combo) for e in envs], dtype=np.int32),
            np.array([int(e._get_total_garbage()) for e in envs], dtype=np.int32),
        )
        if (
            result.placement_overflow
            or result.pool_overflow
            or result.workitem_overflow
        ):
            raise RuntimeError(
                "teacher buffers overflowed; lower --batch or the beam width"
            )

        for i, env in enumerate(envs):
            action = int(result.action[i])
            n = int(result.root_count[i])
            ri = int(result.root_index[i])
            if action < 0 or n == 0 or not 0 <= ri < n:
                deaths += 1
                steps[i] = env.reset()
                continue

            board = obs[i]["board"].astype(np.float32)
            pieces = obs[i]["pieces"].astype(np.int64)
            bcg = obs[i]["b2b_combo_garbage"].astype(np.float32)
            placements, scores = build_placement_target(
                result.root_action[i, :n],
                result.root_score[i, :n],
                result.root_row[i, :n],
                active_piece=env._active_piece.piece_type.value,
                hold_piece=env._hold_piece.value,
                queue0=int(queues[i, 0]),
                row_norm=env._board.shape[0] - 1,
            )
            transitions.append(
                (board, pieces, bcg, placements, scores, float(result.best_score[i]))
            )

            seq = descriptor_key_sequence(
                env, placement_of(action, int(result.root_row[i, ri])), max_len
            )
            fallbacks += int(np.array_equal(seq, forced))
            steps[i] = env._step(np.asarray(seq, dtype=np.int64))
            total_attack += float(steps[i].reward["attack"])
            pieces_placed += 1
            max_b2b = max(max_b2b, int(env._scorer._b2b))
            if steps[i].is_last():
                deaths += 1
                steps[i] = env.reset()

        done = min(len(transitions), num_steps)
        pbar.n = done
        stats = (
            f"transitions={done} deaths={deaths} max_b2b={max_b2b} "
            f"app={total_attack / max(pieces_placed, 1):.3f} fallbacks={fallbacks}"
        )
        if headless and done % log_every < batch:
            print(f"Step {done}/{num_steps} | {stats}", flush=True)
        else:
            pbar.set_postfix_str(stats)
        pbar.refresh()
    pbar.close()

    app = total_attack / max(pieces_placed, 1)
    return transitions[:num_steps], deaths, max_b2b, app


def main(args):
    dataset_path = (
        str(args.output) if args.output else "datasets/tetris_oracle_placement"
    )
    num_steps = args.num_steps
    seed = getattr(args, "seed", 0)

    datagen_cfg = DataGenConfig()
    queue_size = datagen_cfg.queue_size
    max_len = 15
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
            if (
                cp is None
                or cp.shape[1:] != (CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM)
                or "value_scores" not in existing
            ):
                print(
                    "Existing dataset is an older schema (needs 128-slot "
                    "`cand_placements` and `value_scores`) - starting fresh.",
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

    engine = getattr(args, "engine", "c")
    batch = int(getattr(args, "batch", 64))
    if engine == "gpu":
        print(
            f"Collecting {num_steps} steps over {batch} envs on the GPU teacher "
            f"(seed offset {existing_count})...",
            flush=True,
        )
        collector = functools.partial(collect_batched, batch=batch)
    else:
        print(
            f"Collecting {num_steps} steps in single env "
            f"(seed offset {existing_count})...",
            flush=True,
        )
        collector = collect

    new_transitions, deaths, max_b2b, app = collector(
        seed=seed + existing_count,
        num_steps=num_steps,
        search_depth=datagen_cfg.search_depth,
        beam_width=datagen_cfg.beam_width,
        queue_size=queue_size,
        max_len=max_len,
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
        f"deaths: {deaths} | max_b2b: {max_b2b} | APP: {app:.3f}",
        flush=True,
    )

    boards = np.stack([t[0] for t in new_transitions]).astype(np.float32)
    pieces = np.stack([t[1] for t in new_transitions]).astype(np.int64)
    bcg = np.stack([t[2] for t in new_transitions]).astype(np.float32)
    cand_placements = np.stack([t[3] for t in new_transitions]).astype(np.float32)
    cand_scores = np.stack([t[4] for t in new_transitions]).astype(np.float32)
    value_scores = np.array([t[5] for t in new_transitions], dtype=np.float32)

    if existing is not None:
        boards = np.concatenate([existing["boards"], boards])
        pieces = np.concatenate([existing["pieces"], pieces])
        bcg = np.concatenate([existing["b2b_combo_garbage"], bcg])
        cand_placements = np.concatenate([existing["cand_placements"], cand_placements])
        cand_scores = np.concatenate([existing["cand_scores"], cand_scores])
        value_scores = np.concatenate([existing["value_scores"], value_scores])
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
            "value_scores": value_scores,
        }
    )
    dataset.save(dataset_path)
    print(f"Saved {len(cand_scores)} transitions to {dataset_path}", flush=True)
