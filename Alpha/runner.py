"""Phase 1 solo runner.

Drives solo Tetris games with MCTS (DecomposeOracle) and compares to the
existing C-side beam search via b2b_run_eval_games on the same seeds.

Phase 1 exit criterion: MCTS at NUM_SIMULATIONS solo ≥ b2b_search_c at
(BEAM_SEARCH_DEPTH, BEAM_SEARCH_BEAM) on solo APP, max_b2b, survival.
"""

from __future__ import annotations
import time
from typing import Dict, List

import numpy as np

from TetrisEnv.CTetrisCore import TetrisCore
from TetrisEnv.CB2BSearch import CB2BSearch, GameConfig

from .tree import Node
from .mcts import MCTS, select_action
from .valuator import CheapStateValuator, DecomposeOracle, Valuator
from . import config as cfg


def run_mcts_game(seed: int, num_steps: int, num_simulations: int,
                  valuator: Valuator) -> Dict:
    ts = TetrisCore(
        seed=seed, queue_size=cfg.QUEUE_SIZE,
        garbage_push_delay=cfg.GARBAGE_PUSH_DELAY,
    )
    mcts = MCTS(valuator=valuator)
    root = Node(ts)

    total_attack = 0.0
    max_b2b = 0
    heights: List[int] = []
    steps_done = 0
    survived = True

    for step in range(num_steps):
        mcts.run(root, num_simulations)
        if root.is_terminal or root.num_actions == 0:
            survived = False
            break

        a = select_action(root, temperature=cfg.EVAL_TEMPERATURE)
        if a < 0:
            survived = False
            break

        placement = root.placements[a]
        new_state = root.state.clone()
        ev = new_state.apply_placement(
            int(placement[0]), int(placement[1]),
            int(placement[2]), int(placement[3]), int(placement[4]),
        )

        total_attack += ev.attack
        if new_state.b2b > max_b2b:
            max_b2b = new_state.b2b

        h = 0
        b = new_state.board
        for r in range(40):
            if b[r] != 0:
                h = new_state.board_height - r if r < new_state.board_height else 0
                break
        heights.append(h)
        steps_done = step + 1

        if ev.terminal:
            survived = False
            break

        # Subtree reuse: prefer the child the search built up.
        child = root.children[a] if root.children else None
        if child is None:
            child = Node(new_state, is_terminal=False)
        root = child

    return {
        "steps": steps_done,
        "survived": int(survived),
        "total_attack": total_attack,
        "app": total_attack / max(steps_done, 1),
        "max_b2b": max_b2b,
        "avg_height": float(np.mean(heights)) if heights else 0.0,
        "max_height": max(heights) if heights else 0,
        "end_height": heights[-1] if heights else 0,
    }


def run_mcts_benchmark(seeds: List[int], num_steps: int, num_simulations: int,
                       valuator: Valuator) -> List[Dict]:
    out = []
    for s in seeds:
        t0 = time.perf_counter()
        r = run_mcts_game(s, num_steps, num_simulations, valuator)
        r["wall_seconds"] = time.perf_counter() - t0
        out.append(r)
    return out


def run_beam_benchmark(seeds: List[int], num_steps: int,
                       depth: int, beam: int) -> List[Dict]:
    searcher = CB2BSearch()
    configs = [
        GameConfig(seed=s, garbage_chance=cfg.EVAL_GARBAGE_CHANCE,
                   garbage_min=0, garbage_max=0,
                   garbage_push_delay=cfg.GARBAGE_PUSH_DELAY)
        for s in seeds
    ]
    t0 = time.perf_counter()
    results = searcher.run_eval_games(
        configs, num_steps=num_steps, search_depth=depth,
        beam_width=beam, queue_size=cfg.QUEUE_SIZE,
    )
    total_wall = time.perf_counter() - t0
    per_game = total_wall / max(len(seeds), 1)
    out = []
    for r in results:
        out.append({
            "steps": r.steps_completed,
            "survived": r.survived,
            "total_attack": r.total_attack,
            "app": r.total_attack / max(r.steps_completed, 1),
            "max_b2b": r.max_b2b,
            "avg_height": r.avg_height,
            "max_height": r.max_height,
            "end_height": r.end_height,
            "wall_seconds": per_game,
        })
    return out


def summarize(label: str, results: List[Dict]) -> None:
    avg = lambda k: float(np.mean([r[k] for r in results]))
    print(f"  {label:>26}: "
          f"steps={avg('steps'):>5.1f}  "
          f"surv={avg('survived')*100:>4.0f}%  "
          f"attack={avg('total_attack'):>5.1f}  "
          f"APP={avg('app'):>5.3f}  "
          f"maxB2B={avg('max_b2b'):>5.1f}  "
          f"avgH={avg('avg_height'):>4.1f}  "
          f"maxH={avg('max_height'):>4.1f}  "
          f"wall={avg('wall_seconds'):>6.2f}s/game")


def main():
    seeds = list(range(1, cfg.NUM_EVAL_GAMES + 1))
    print(f"Solo Phase-1 evaluation — {len(seeds)} seeds × {cfg.NUM_STEPS_PER_GAME} steps", flush=True)
    print(flush=True)

    decomp_v = DecomposeOracle()
    sim_budgets = [50, 100, 200]

    mcts_runs = {}
    for sims in sim_budgets:
        print(f"  Running MCTS @ {sims} sims (decompose V + softmax priors) ...", flush=True)
        mcts_runs[sims] = run_mcts_benchmark(seeds, cfg.NUM_STEPS_PER_GAME, sims, decomp_v)

    print(f"  Running beam search (depth={cfg.BEAM_SEARCH_DEPTH}, beam={cfg.BEAM_SEARCH_BEAM}) ...", flush=True)
    beam_results = run_beam_benchmark(
        seeds, cfg.NUM_STEPS_PER_GAME,
        cfg.BEAM_SEARCH_DEPTH, cfg.BEAM_SEARCH_BEAM,
    )

    print(flush=True)
    for sims in sim_budgets:
        summarize(f"MCTS@{sims} decomp", mcts_runs[sims])
    summarize(f"beam d={cfg.BEAM_SEARCH_DEPTH} b={cfg.BEAM_SEARCH_BEAM}", beam_results)

    beam_app = float(np.mean([r["app"] for r in beam_results]))
    print(flush=True)
    print(f"  Phase 1 exit criterion (MCTS APP ≥ beam APP={beam_app:.3f}):")
    for sims in sim_budgets:
        a = float(np.mean([r["app"] for r in mcts_runs[sims]]))
        s = float(np.mean([r["survived"] for r in mcts_runs[sims]])) * 100
        print(f"    @{sims:>4} sims: APP={a:.3f}  surv={s:>3.0f}%  → {'PASS' if a >= beam_app else 'FAIL'}")


if __name__ == "__main__":
    main()
