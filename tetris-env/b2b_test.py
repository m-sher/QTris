#!/usr/bin/env python3
"""Headless benchmark for b2b_search.

Runs N games per configuration and reports:
  - Avg/max height over the game
  - Total attack, total clears, lines/piece, attack/piece
  - Survival (steps before death)

Usage:
  uv run python b2b_test.py              # default benchmark (garbage)
  uv run python b2b_test.py nogarb       # no-garbage B2B-only benchmark

Optional flags:
  --depth N   search depth (default 7)
  --beam  N   beam width (default 128)
"""

import numpy as np
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch


def run_game(seed, num_steps=200, search_depth=7, beam_width=128,
             garbage_chance=0.15, garbage_min=1, garbage_max=4, queue_size=5):
    """Run a single headless game and return stats dict."""
    env = PyTetrisEnv(
        queue_size=queue_size, max_holes=None, max_height=20,
        max_steps=None, max_len=15, pathfinding=False,
        seed=seed, idx=0,
        garbage_chance=garbage_chance, garbage_min=garbage_min,
        garbage_max=garbage_max, auto_push_garbage=True, auto_fill_queue=True,
    )
    env.reset()
    searcher = CB2BSearch()

    total_attack = 0.0
    total_clears = 0
    heights = []
    max_b2b = -1
    step = 0
    game_over = False

    for step in range(1, num_steps + 1):
        if game_over:
            break

        board = env._board
        active = env._active_piece.piece_type.value
        hold = env._hold_piece.value
        queue_types = np.array([p.value for p in env._queue], dtype=np.int32)
        b2b = env._scorer._b2b
        combo = env._scorer._combo
        total_garb = env._get_total_garbage()

        action_idx, sequence = searcher.search(
            board=board, active_piece=active, hold_piece=hold,
            queue=queue_types, b2b=b2b, combo=combo,
            total_garbage=total_garb,
            garbage_push_delay=env._garbage_push_delay,
            search_depth=search_depth,
            beam_width=beam_width, max_len=15,
        )

        if action_idx < 0:
            game_over = True
            break

        ts = env._step(sequence)

        atk = float(ts.reward["attack"])
        clr = float(ts.reward["clear"])
        total_attack += atk
        total_clears += int(clr)
        if env._scorer._b2b > max_b2b:
            max_b2b = env._scorer._b2b

        h = 0
        for r in range(24):
            if np.any(env._board[r] != 0):
                h = 24 - r
                break
        heights.append(h)

        if ts.is_last():
            game_over = True

    return {
        "steps": step,
        "total_attack": total_attack,
        "total_clears": total_clears,
        "lines_per_piece": total_clears / max(step, 1),
        "avg_height": np.mean(heights) if heights else 0,
        "max_height": max(heights) if heights else 0,
        "survived": not game_over,
        "attack_per_piece": total_attack / max(step, 1),
        "max_b2b": max_b2b,
    }


def benchmark(label, num_games=5, num_steps=100, **run_kwargs):
    results = []
    for seed in range(num_games):
        r = run_game(seed=seed + 100, num_steps=num_steps, **run_kwargs)
        results.append(r)

    avg = lambda key: np.mean([r[key] for r in results])
    survived = sum(1 for r in results if r["survived"])

    depth = run_kwargs.get("search_depth", 7)
    beam = run_kwargs.get("beam_width", 128)

    print(f"\n{'='*60}")
    print(f"  {label} (depth={depth}, beam={beam})")
    print(f"{'='*60}")
    print(f"  Games: {num_games} | Steps/game: {num_steps}")
    print(f"  Survived:       {survived}/{num_games}")
    print(f"  Avg steps:      {avg('steps'):.1f}")
    print(f"  Avg height:     {avg('avg_height'):.1f}")
    print(f"  Max height:     {avg('max_height'):.1f}")
    print(f"  Total attack:   {avg('total_attack'):.1f}")
    print(f"  Total clears:   {avg('total_clears'):.1f}")
    print(f"  Lines/piece:    {avg('lines_per_piece'):.3f}")
    print(f"  Attack/piece:   {avg('attack_per_piece'):.3f}")
    print(f"  Max b2b (avg):  {avg('max_b2b'):.1f}")
    return results


def no_garbage_benchmark(label, num_games=3, num_steps=200, **run_kwargs):
    results = []
    for seed in range(num_games):
        r = run_game(
            seed=seed + 500, num_steps=num_steps,
            garbage_chance=0.0, garbage_min=0, garbage_max=0,
            **run_kwargs,
        )
        results.append(r)

    avg = lambda key: np.mean([r[key] for r in results])
    survived = sum(1 for r in results if r["survived"])

    depth = run_kwargs.get("search_depth", 7)
    beam = run_kwargs.get("beam_width", 128)

    print(f"\n{'='*60}")
    print(f"  NO GARBAGE: {label} (depth={depth}, beam={beam})")
    print(f"{'='*60}")
    print(f"  Games: {num_games} | Steps/game: {num_steps}")
    print(f"  Survived:       {survived}/{num_games}")
    print(f"  Avg steps:      {avg('steps'):.1f}")
    print(f"  Avg height:     {avg('avg_height'):.1f}")
    print(f"  Max height:     {avg('max_height'):.1f}")
    print(f"  Total attack:   {avg('total_attack'):.1f}")
    print(f"  Attack/piece:   {avg('attack_per_piece'):.3f}")
    print(f"  Max b2b (avg):  {avg('max_b2b'):.1f}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("mode", nargs="?", default="garb",
                        help="'nogarb' for no-garbage benchmark, otherwise default garbage benchmark")
    parser.add_argument("--depth", type=int, default=7, help="search depth (default 7)")
    parser.add_argument("--beam", type=int, default=128, help="beam width (default 128)")
    parser.add_argument("--steps", type=int, default=None, help="steps per game")
    parser.add_argument("--games", type=int, default=None, help="number of games")
    ns = parser.parse_args()

    kwargs = {"search_depth": ns.depth, "beam_width": ns.beam}

    if ns.mode.lower() == "nogarb":
        no_garbage_benchmark(
            "Hand-designed heuristics",
            num_games=ns.games if ns.games is not None else 3,
            num_steps=ns.steps if ns.steps is not None else 200,
            **kwargs,
        )
    else:
        benchmark(
            "Hand-designed heuristics",
            num_games=ns.games if ns.games is not None else 5,
            num_steps=ns.steps if ns.steps is not None else 200,
            **kwargs,
        )
