"""Checkpoint / pool / artifact inventory for AZ placement.

Module is named artifacts.py (not *checkpoint*) so git can track it under the
repo-wide `**/**checkpoint**` ignore used for TF weight directories.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Any

from qtris.mcp.az_placement.paths import (
    AZ_CKPT_GLOBS,
    DEFAULT_1V1_CKPT,
    DEFAULT_AZ_STATES,
    DEFAULT_GARBAGE_TRACES,
    DEFAULT_PRETRAIN_CKPT,
    DEFAULT_SINGLE_CKPT,
    dir_size_mb,
    repo_root,
    resolve_path,
)


def _read_checkpoint_pointer(ckpt_dir: Path) -> dict[str, Any] | None:
    ptr = ckpt_dir / "checkpoint"
    if not ptr.exists():
        return None
    text = ptr.read_text()
    m = re.search(r'model_checkpoint_path:\s*"([^"]+)"', text)
    all_paths = re.findall(r'all_model_checkpoint_paths:\s*"([^"]+)"', text)
    latest = m.group(1) if m else None
    step = None
    if latest:
        sm = re.search(r"ckpt-(\d+)", latest)
        if sm:
            step = int(sm.group(1))
    return {
        "latest": latest,
        "step": step,
        "kept": all_paths,
    }


def _infer_mode(ckpt_dir: Path) -> str:
    name = ckpt_dir.name
    if "1v1" in name:
        return "1v1"
    if name == "placement_pretrained_policy":
        return "pretrain"
    if (ckpt_dir / "pool").is_dir():
        return "1v1"
    return "single"


def list_az_checkpoints(pattern: str | None = None) -> dict[str, Any]:
    """List known AZ-related checkpoint directories under the repo."""
    root = repo_root()
    globs = [pattern] if pattern else list(AZ_CKPT_GLOBS)
    seen: set[Path] = set()
    entries = []
    for g in globs:
        for match in sorted(glob.glob(str(root / g))):
            p = Path(match)
            if not p.is_dir() or p in seen:
                continue
            seen.add(p)
            ptr = _read_checkpoint_pointer(p)
            pool = p / "pool"
            entries.append(
                {
                    "path": str(p.relative_to(root)),
                    "abs_path": str(p),
                    "mode": _infer_mode(p),
                    "checkpoint": ptr,
                    "has_pool": pool.is_dir(),
                    "pool_snapshots": _pool_snapshot_names(pool)
                    if pool.is_dir()
                    else [],
                    "size_mb": dir_size_mb(p),
                }
            )
    return {"repo_root": str(root), "checkpoints": entries}


def inspect_checkpoint(
    checkpoint_dir: str | None = None, mode: str | None = None
) -> dict[str, Any]:
    """Describe a checkpoint dir: pointer, pool, elo presence, optional weight health."""
    from qtris.mcp.az_placement.net_io import load_checkpoint, param_stats

    default = DEFAULT_1V1_CKPT
    ckpt_dir = resolve_path(checkpoint_dir, default)
    raw_mode = mode or _infer_mode(ckpt_dir)
    inferred = "single" if raw_mode == "pretrain" else raw_mode

    out: dict[str, Any] = {
        "path": str(ckpt_dir),
        "exists": ckpt_dir.exists(),
        "mode": inferred,
        "checkpoint": _read_checkpoint_pointer(ckpt_dir) if ckpt_dir.is_dir() else None,
    }
    if raw_mode == "pretrain":
        out["note"] = (
            "Pretrain policy-only checkpoint loaded as single; the value head is "
            "untrained, so its param_stats/restore_matched reflect random weights."
        )
    pool = ckpt_dir / "pool"
    if pool.is_dir():
        out["pool"] = {
            "dir": str(pool),
            "snapshots": _pool_snapshot_names(pool),
            "elo_json": str(pool / "elo.json")
            if (pool / "elo.json").exists()
            else None,
        }

    if not ckpt_dir.exists():
        return out

    # Weight health
    try:
        load_mode = "1v1" if inferred == "1v1" else "single"
        loaded = load_checkpoint(ckpt_dir, mode=load_mode, batch_size=1)
        out["loaded_checkpoint"] = loaded.checkpoint
        out["return_scale"] = loaded.return_scale
        out["extra"] = loaded.extra
        out["param_stats"] = param_stats(loaded.net)
    except Exception as e:
        out["load_error"] = f"{type(e).__name__}: {e}"
    return out


def _pool_snapshot_names(pool_dir: Path) -> list[str]:
    if not pool_dir.is_dir():
        return []
    names = []
    for idx in sorted(pool_dir.glob("gen_*.index")):
        names.append(idx.stem)

    # sort by generation number
    def _key(n: str) -> int:
        try:
            return int(n.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return sorted(names, key=_key)


def get_elo(checkpoint_dir: str | None = None) -> dict[str, Any]:
    """Read opponent-pool Elo book (1v1 AZ only)."""
    from qtris.training.elo import EloBook

    ckpt_dir = resolve_path(checkpoint_dir, DEFAULT_1V1_CKPT)
    elo_path = ckpt_dir / "pool" / "elo.json"
    if not elo_path.exists():
        return {
            "path": str(elo_path),
            "exists": False,
            "hint": "Elo is written by train placement --mode 1v1 --algo az.",
        }
    book = EloBook.from_json(str(elo_path))
    snaps = _pool_snapshot_names(ckpt_dir / "pool")
    summary = book.present_summary([s for s in snaps if s != "gen_0"] + ["gen_0"])
    ratings_sorted = sorted(book.ratings.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "path": str(elo_path),
        "exists": True,
        "init": book.init,
        "k_learner": book.k_learner,
        "k_opp": book.k_opp,
        "anchor": book.anchor,
        "learner": book.learner,
        "ratings": dict(ratings_sorted),
        "games": book.games,
        "pool_snapshots": snaps,
        "summary": summary,
        "top5": ratings_sorted[:5],
        "bottom5": list(reversed(ratings_sorted[-5:])),
    }


def list_pool(checkpoint_dir: str | None = None) -> dict[str, Any]:
    """List opponent-pool snapshots with optional Elo join."""
    ckpt_dir = resolve_path(checkpoint_dir, DEFAULT_1V1_CKPT)
    pool = ckpt_dir / "pool"
    snaps = _pool_snapshot_names(pool)
    elo = get_elo(checkpoint_dir)
    ratings = elo.get("ratings", {}) if elo.get("exists") else {}
    items = []
    for s in snaps:
        items.append(
            {
                "id": s,
                "prefix": str(pool / s),
                "elo": ratings.get(s),
                "games": elo.get("games", {}).get(s) if elo.get("exists") else None,
            }
        )
    return {
        "pool_dir": str(pool),
        "count": len(items),
        "learner_elo": ratings.get("learner"),
        "snapshots": items,
    }


def list_az_states(states_dir: str | None = None) -> dict[str, Any]:
    """Inventory AZ state harvest dirs (data/az_states/run*)."""
    d = resolve_path(states_dir, DEFAULT_AZ_STATES)
    if not d.exists():
        return {"path": str(d), "exists": False, "runs": []}
    runs = []
    for run in sorted(d.iterdir()):
        if not run.is_dir():
            continue
        # TF dataset shards: count snapshot/metadata pairs roughly
        snaps = list(run.rglob("*.snapshot"))
        runs.append(
            {
                "name": run.name,
                "path": str(run),
                "snapshot_files": len(snaps),
                "size_mb": dir_size_mb(run),
            }
        )
    return {"path": str(d), "exists": True, "runs": runs}


def list_garbage_traces(traces_dir: str | None = None) -> dict[str, Any]:
    """Summarize garbage_traces tier libraries used by solo AZ."""
    d = resolve_path(traces_dir, DEFAULT_GARBAGE_TRACES)
    if not d.exists():
        return {"path": str(d), "exists": False, "tiers": []}
    tiers = []
    for tier in sorted(d.iterdir()):
        if not tier.is_dir():
            continue
        npy = list(tier.glob("*.npy"))
        tiers.append(
            {
                "name": tier.name,
                "path": str(tier),
                "trace_files": len(npy),
                "size_mb": dir_size_mb(tier),
            }
        )
    return {"path": str(d), "exists": True, "tiers": tiers}


def defaults() -> dict[str, Any]:
    """Document default paths and modes for agents."""
    root = repo_root()
    return {
        "repo_root": str(root),
        "defaults": {
            "single_az_checkpoint": DEFAULT_SINGLE_CKPT,
            "one_v_one_az_checkpoint": DEFAULT_1V1_CKPT,
            "pretrain_checkpoint": DEFAULT_PRETRAIN_CKPT,
            "tensorboard_root": "tb_logs/Tetris",
            "az_states": DEFAULT_AZ_STATES,
            "garbage_traces": DEFAULT_GARBAGE_TRACES,
        },
        "train_commands": {
            "single_az": "uv run train placement --algo az",
            "one_v_one_az": "uv run train placement --mode 1v1 --algo az",
            "demo_mcts": "uv run demo placement --checkpoint checkpoints/1v1_placement_az --num-simulations 256",
        },
        "scope": "placement AZ only (single-player + 1v1 opponent-pool). PPO/AR/flat not exposed.",
    }
