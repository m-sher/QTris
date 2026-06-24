"""Interactive probes: policy/value forward pass and lightweight MCTS diagnostics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from qtris.mcp.az_placement.paths import DEFAULT_1V1_CKPT, resolve_path


def _fresh_env(seed: int = 0, *, auto_fill_queue: bool = False):
    from TetrisEnv.PyTetrisEnv import PyTetrisEnv

    return PyTetrisEnv(
        queue_size=5,
        max_holes=50,
        max_steps=None,
        max_len=15,
        pathfinding=False,
        seed=seed,
        idx=0,
        garbage_chance=0.0,
        garbage_min=0,
        garbage_max=0,
        auto_push_garbage=False,
        auto_fill_queue=auto_fill_queue,
        num_row_tiers=2,
        placement_candidates=False,
    )


def _search_return_scale(mode: str, loaded) -> float:
    """1v1 search runs at return_scale=1; solo uses the checkpoint's calibrated scale."""
    if mode == "1v1":
        return 1.0
    return float(loaded.return_scale) if loaded.return_scale else 1.0


def probe_policy(
    checkpoint_dir: str | None = None,
    *,
    mode: str = "1v1",
    seed: int = 0,
    top_k: int = 8,
    num_simulations: int = 0,
) -> dict[str, Any]:
    """Run search on a fresh solo board and report root value + top candidates.

    root_value is the net's forward-pass value at the root (independent of sims).
    num_simulations=0 is bumped to a single PUCT sim for the candidate readout; the
    reported policy is the MCTS visit distribution (pi).
    """
    import numpy as np

    from qtris.mcp.az_placement.net_io import load_checkpoint
    from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS

    ckpt_dir = resolve_path(checkpoint_dir, DEFAULT_1V1_CKPT)
    load_mode = "1v1" if mode == "1v1" else "single"
    loaded = load_checkpoint(ckpt_dir, mode=load_mode, batch_size=1, compile_net=True)
    return_scale = _search_return_scale(mode, loaded)

    env = _fresh_env(seed=seed)
    env.reset()

    sims = max(1, int(num_simulations))  # engine needs >=1 round to produce a policy
    cfg = MCTSConfig(
        num_simulations=sims,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.0,  # deterministic probe
        gamma=1.0 if mode == "1v1" else 0.99,
        temp_moves=0,
        w_attack=0.05 if mode == "1v1" else 1.0,
        w_death=1.0 if mode == "1v1" else 100.0,
        w_b2b=0.06 if mode == "1v1" else 0.0,
        leaves_per_round=4,
    )
    searcher = PlacementMCTS(loaded.net, cfg)

    t0 = time.perf_counter()
    r = searcher.search([env], return_scale, 0.0)[0]
    elapsed = time.perf_counter() - t0

    if r.get("dead"):
        return {
            "checkpoint": loaded.checkpoint,
            "mode": mode,
            "seed": seed,
            "dead": True,
            "note": "Root had no legal move.",
        }

    pi = np.asarray(r["pi"], dtype=np.float64)
    legal = np.flatnonzero(np.asarray(r["cand_mask"]))
    order = legal[np.argsort(-pi[legal])][:top_k]
    top = [{"cand_idx": int(i), "policy": float(pi[i])} for i in order]
    return {
        "checkpoint": loaded.checkpoint,
        "mode": mode,
        "seed": seed,
        "num_simulations_requested": int(num_simulations),
        "num_simulations_run": sims,
        "elapsed_sec": round(elapsed, 4),
        "root_value": float(r["value"]),
        "return_scale": loaded.return_scale,
        "num_legal": int(legal.size),
        "total_visits": int(r["visits"]),
        "chosen_slot": int(r["slot"]),
        "chosen_descriptor": list(r["descriptor"]),
        "top_candidates": top,
        "note": "policy is the MCTS visit distribution (pi); root_value is the net forward-pass value.",
    }


def _rollout_cfg(mode: str, num_simulations: int):
    from qtris.search.placement_mcts import MCTSConfig

    return MCTSConfig(
        num_simulations=max(1, int(num_simulations)),
        dirichlet_eps=0.0,  # greedy, deterministic rollout
        gamma=1.0 if mode == "1v1" else 0.99,
        w_attack=0.05 if mode == "1v1" else 1.0,
        w_death=1.0 if mode == "1v1" else 100.0,
        w_b2b=0.06 if mode == "1v1" else 0.0,
    )


def _resolve_opponent(ckpt_dir: Path, opponent: str | None) -> tuple[Path, str]:
    """Pick the 1v1 opponent: explicit path/snapshot id, else pool gen_0, else self."""
    pool = ckpt_dir / "pool"
    if opponent:
        p = Path(opponent)
        if p.is_absolute() or p.exists() or Path(str(p) + ".index").exists():
            return p, str(p)
        return pool / opponent, f"pool/{opponent}"
    if (pool / "gen_0.index").exists():
        return pool / "gen_0", "pool/gen_0"
    return ckpt_dir, "self"


def _play_1v1(
    loaded, learner_mcts, cfg, ckpt_dir, opponent, num_games, max_steps, seed
):
    import random

    from TetrisEnv.CB2BSearch import CB2BSearch
    from qtris.mcp.az_placement.net_io import load_checkpoint
    from qtris.search.placement_mcts import PlacementMCTS
    from qtris.training._1v1_placement_az import _eval_vs_ref

    opp_path, opp_label = _resolve_opponent(ckpt_dir, opponent)
    opp = load_checkpoint(opp_path, mode="1v1", batch_size=num_games, compile_net=True)
    opp_mcts = PlacementMCTS(opp.net, cfg)
    searcher = CB2BSearch()
    rng = random.Random(seed)

    t0 = time.perf_counter()
    wins, losses = _eval_vs_ref(
        learner_mcts, opp_mcts, num_games, 5, 15, max_steps, rng, searcher
    )
    elapsed = time.perf_counter() - t0
    decisive = wins + losses
    return {
        "checkpoint": loaded.checkpoint,
        "mode": "1v1",
        "opponent": opp_label,
        "opponent_checkpoint": opp.checkpoint,
        "num_games": num_games,
        "num_simulations": cfg.num_simulations,
        "max_steps": max_steps,
        "elapsed_sec": round(elapsed, 3),
        "wins": wins,
        "losses": losses,
        "draws": num_games - decisive,
        "win_rate": round(wins / num_games, 3),
        "loss_rate": round(losses / num_games, 3),
        "draw_rate": round((num_games - decisive) / num_games, 3),
        "decisive_win_rate": round(wins / decisive, 3) if decisive else None,
        "note": "Learner (p1) vs opponent (p2), both greedy. Draws = timeouts. Use mode=single for APP/b2b.",
    }


def play_games(
    checkpoint_dir: str | None = None,
    *,
    mode: str = "single",
    num_games: int = 8,
    max_steps: int = 200,
    num_simulations: int = 16,
    seed: int = 0,
    opponent: str | None = None,
) -> dict[str, Any]:
    """Play full games headless and report outcome stats (what single-root probe can't).

    mode=single: solo rollout -> app (attack/piece, the b2b-APP target), b2b, clears, deaths.
    mode=1v1: learner vs opponent (pool gen_0 by default) -> win/loss/draw rates.
    """
    from TetrisEnv.CB2BSearch import CB2BSearch
    from qtris.mcp.az_placement.net_io import load_checkpoint
    from qtris.search.placement_mcts import PlacementMCTS
    from qtris.search.placement_search import placement_step

    ckpt_dir = resolve_path(checkpoint_dir, DEFAULT_1V1_CKPT)
    load_mode = "1v1" if mode == "1v1" else "single"
    loaded = load_checkpoint(
        ckpt_dir, mode=load_mode, batch_size=num_games, compile_net=True
    )
    return_scale = _search_return_scale(mode, loaded)
    cfg = _rollout_cfg(mode, num_simulations)
    mcts = PlacementMCTS(loaded.net, cfg)

    if mode == "1v1":
        return _play_1v1(
            loaded, mcts, cfg, ckpt_dir, opponent, num_games, max_steps, seed
        )

    searcher = CB2BSearch()
    envs = [_fresh_env(seed=seed + i, auto_fill_queue=True) for i in range(num_games)]
    for e in envs:
        e._reset()

    alive = [True] * num_games
    pieces = [0] * num_games
    attack = [0.0] * num_games
    clears = [0] * num_games
    max_b2b = [0] * num_games
    died_flag = [False] * num_games

    t0 = time.perf_counter()
    for _ in range(max_steps):
        idx = [g for g in range(num_games) if alive[g]]
        if not idx:
            break
        res = mcts.search([envs[g] for g in idx], return_scale, 0.0)
        for j, g in enumerate(idx):
            r = res[j]
            if r.get("dead"):
                alive[g] = False
                died_flag[g] = True
                continue
            _total, atk, clr, died = placement_step(envs[g], searcher, r["descriptor"])
            pieces[g] += 1
            attack[g] += float(atk)
            clears[g] += int(clr)
            max_b2b[g] = max(max_b2b[g], int(envs[g]._scorer._b2b))
            if died:
                alive[g] = False
                died_flag[g] = True
    elapsed = time.perf_counter() - t0

    total_pieces = sum(pieces)
    total_attack = sum(attack)
    deaths = sum(died_flag)
    return {
        "checkpoint": loaded.checkpoint,
        "mode": "single",
        "num_games": num_games,
        "num_simulations": cfg.num_simulations,
        "max_steps": max_steps,
        "elapsed_sec": round(elapsed, 3),
        "app": round(total_attack / max(total_pieces, 1), 4),
        "avg_attack_per_game": round(total_attack / num_games, 3),
        "avg_pieces_per_game": round(total_pieces / num_games, 2),
        "avg_clears_per_game": round(sum(clears) / num_games, 2),
        "avg_max_b2b": round(sum(max_b2b) / num_games, 2),
        "max_b2b_overall": max(max_b2b) if max_b2b else 0,
        "death_rate": round(deaths / num_games, 3),
        "note": "Solo rollout, no garbage. app = total attack / total pieces (the b2b-APP target).",
    }


def benchmark_inference(
    checkpoint_dir: str | None = None,
    *,
    mode: str = "1v1",
    batch_size: int = 16,
    warmup: int = 2,
    repeats: int = 5,
    num_simulations: int = 0,
) -> dict[str, Any]:
    """Time batched net forward and optionally batched MCTS search."""
    import numpy as np
    import tensorflow as tf
    from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
    from qtris.mcp.az_placement.net_io import load_checkpoint
    from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS

    ckpt_dir = resolve_path(checkpoint_dir, DEFAULT_1V1_CKPT)
    load_mode = "1v1" if mode == "1v1" else "single"
    loaded = load_checkpoint(
        ckpt_dir, mode=load_mode, batch_size=batch_size, compile_net=True
    )
    net = loaded.net

    boards = np.zeros((batch_size, 24, 10, 1), np.float32)
    pieces = np.full((batch_size, 7), 7, np.int64)
    pieces[:, 1] = 0  # I piece current
    bcg = np.zeros((batch_size, 3), np.float32)
    cands = np.zeros(
        (batch_size, CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM), np.float32
    )
    mask = np.zeros((batch_size, CANDIDATE_CAPACITY), np.bool_)
    mask[:, 0] = True  # one dummy legal arm so softmax is defined
    cands[:, 0, 0] = 1.0

    inputs = (
        tf.constant(boards),
        tf.constant(pieces),
        tf.constant(bcg),
        tf.constant(cands),
        tf.constant(mask),
    )

    def _fwd():
        logits, values = net.policy_value(inputs)
        # force sync
        _ = float(values.numpy()[0, 0])

    for _ in range(warmup):
        _fwd()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _fwd()
        times.append(time.perf_counter() - t0)

    out: dict[str, Any] = {
        "checkpoint": loaded.checkpoint,
        "mode": mode,
        "batch_size": batch_size,
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        "forward_ms_mean": round(1000 * float(np.mean(times)), 3),
        "forward_ms_std": round(1000 * float(np.std(times)), 3),
        "forward_ms_min": round(1000 * float(np.min(times)), 3),
        "per_env_ms_mean": round(1000 * float(np.mean(times)) / batch_size, 4),
        "note": "Runs on CPU unless QTRIS_MCP_GPU is set (server defaults CUDA_VISIBLE_DEVICES empty).",
    }

    if num_simulations and num_simulations > 0:
        return_scale = _search_return_scale(mode, loaded)
        envs = [_fresh_env(seed=i) for i in range(batch_size)]
        for e in envs:
            e.reset()
        cfg = MCTSConfig(
            num_simulations=int(num_simulations),
            dirichlet_eps=0.0,
            gamma=1.0 if mode == "1v1" else 0.99,
            w_attack=0.05 if mode == "1v1" else 1.0,
            w_death=1.0 if mode == "1v1" else 100.0,
            w_b2b=0.06 if mode == "1v1" else 0.0,
        )
        searcher = PlacementMCTS(net, cfg)
        for _ in range(max(1, warmup // 2)):
            searcher.search(envs, return_scale, 0.0)
        st = []
        for _ in range(max(1, repeats // 2)):
            t0 = time.perf_counter()
            searcher.search(envs, return_scale, 0.0)
            st.append(time.perf_counter() - t0)
        out["mcts"] = {
            "num_simulations": num_simulations,
            "batch_envs": batch_size,
            "search_sec_mean": round(float(np.mean(st)), 4),
            "search_ms_per_env": round(1000 * float(np.mean(st)) / batch_size, 3),
        }
    return out
