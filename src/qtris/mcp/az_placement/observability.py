"""TensorBoard / training log discovery for AZ runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qtris.mcp.az_placement.paths import DEFAULT_TB_ROOT, dir_size_mb, resolve_path


def list_tb_runs(
    tb_root: str | None = None,
    *,
    name_filter: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    """List recent TensorBoard run dirs (tb_logs/Tetris/*)."""
    root = resolve_path(tb_root, DEFAULT_TB_ROOT)
    if not root.exists():
        return {"path": str(root), "exists": False, "runs": []}

    runs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if name_filter and name_filter.lower() not in child.name.lower():
            continue
        # AZ runs often tagged in run_name; we can't always know family from dirname.
        events = list(child.glob("events.out.tfevents.*"))
        mtime = child.stat().st_mtime
        if events:
            mtime = max(mtime, max(e.stat().st_mtime for e in events))
        runs.append(
            {
                "name": child.name,
                "path": str(child),
                "event_files": len(events),
                "mtime": mtime,
                "size_mb": dir_size_mb(child),
            }
        )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return {
        "path": str(root),
        "exists": True,
        "total_matched": len(runs),
        "runs": runs[:limit],
        "hint": "Filter with name_filter e.g. '1v1', 'az', 'placement'. Open with: tensorboard --logdir tb_logs/Tetris",
    }


def _latest_run_dir(root: Path) -> Path | None:
    best: Path | None = None
    best_m = -1.0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = child.stat().st_mtime
        events = list(child.glob("events.out.tfevents.*"))
        if events:
            m = max(m, max(e.stat().st_mtime for e in events))
        if m > best_m:
            best_m, best = m, child
    return best


def _as_filters(tags: Any) -> list[str]:
    if not tags:
        return []
    if isinstance(tags, str):
        tags = tags.split(",")
    return [t.strip().lower() for t in tags if str(t).strip()]


def _scalar_series(
    ea, tag: str, scalar_tags: set[str], tensor_util
) -> list[list[float]]:
    out: list[list[float]] = []
    if tag in scalar_tags:
        for e in ea.Scalars(tag):
            out.append([int(e.step), float(e.value)])
    else:
        for e in ea.Tensors(tag):
            arr = tensor_util.make_ndarray(e.tensor_proto)
            try:
                out.append([int(e.step), float(arr)])
            except (TypeError, ValueError):
                return []  # non-scalar tag (image/text) - skip
    return out


def read_tb_scalars(
    tb_run: str | None = None,
    *,
    tags: Any = None,
    tb_root: str | None = None,
    last_n: int = 20,
    max_tags: int = 80,
) -> dict[str, Any]:
    """Read scalar metric series from a TB run's event files (the values list_tb_runs omits).

    tb_run: run dir name under tb_root, an absolute path, or None for the most recent run.
    tags: substring filter(s) on tag name (list or comma string); None returns all.
    """
    from tensorboard.backend.event_processing import event_accumulator
    from tensorboard.util import tensor_util

    root = resolve_path(tb_root, DEFAULT_TB_ROOT)
    if tb_run:
        p = Path(tb_run)
        run_dir = p if p.is_absolute() else root / tb_run
    else:
        run_dir = _latest_run_dir(root) if root.exists() else None

    if run_dir is None or not run_dir.exists():
        return {
            "tb_root": str(root),
            "run": str(run_dir) if run_dir else None,
            "exists": False,
            "hint": "Pass tb_run (a run dir under tb_root) or use az_list_tb_runs.",
        }

    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0, event_accumulator.TENSORS: 0},
    )
    ea.Reload()
    tagset = ea.Tags()
    scalar_tags = set(tagset.get("scalars", []))
    all_tags = sorted(scalar_tags | set(tagset.get("tensors", [])))

    filters = _as_filters(tags)
    matched = [
        t for t in all_tags if not filters or any(f in t.lower() for f in filters)
    ]
    truncated = len(matched) > max_tags

    metrics: dict[str, Any] = {}
    for tag in matched[:max_tags]:
        series = _scalar_series(ea, tag, scalar_tags, tensor_util)
        if not series:
            continue
        vals = [v for _, v in series]
        tail = series[-max(1, last_n) :]
        metrics[tag] = {
            "count": len(series),
            "first": series[0],
            "last": series[-1],
            "min": min(vals),
            "max": max(vals),
            "mean_last_n": round(sum(v for _, v in tail) / len(tail), 6),
            "points": tail,
        }
    return {
        "tb_root": str(root),
        "run": str(run_dir),
        "exists": True,
        "available_tags": all_tags,
        "num_tags_matched": len(matched),
        "truncated": truncated,
        "metrics": metrics,
    }


def describe_az_metrics() -> dict[str, Any]:
    """Schema cheat-sheet: which metrics AZ trainers log (for interpreting TB/wandb)."""
    return {
        "single_agent_az": {
            "config_model": "AlphaZeroTrainConfig",
            "log_model": "SingleAgentAZLog",
            "key_metrics": [
                "policy_loss",
                "value_loss",
                "entropy",
                "policy_kl",
                "update_kl",
                "explained_var",
                "value_mean",
                "return_var",
                "return_scale",
                "avg_total_reward",
                "avg_attacks",
                "avg_deaths",
                "avg_b2b",
                "garbage_in_app",
                "garbage_cancel_frac",
                "avg_visits",
                "pruned_mass",
            ],
            "checkpoint_default": "checkpoints/placement_az",
            "value_target": "search-bootstrapped discounted return (return_scale units)",
        },
        "one_v_one_az": {
            "config_model": "OneVsOnePlacementAZConfig",
            "log_model": "OneVsOneAZLog",
            "key_metrics": [
                "policy_loss",
                "value_loss",
                "entropy",
                "policy_kl",
                "update_kl",
                "explained_var",
                "value_mean",
                "win_rate",
                "win_rate_vs_ref",
                "draw_rate",
                "app",
                "value_calibration",
                "avg_game_len",
                "avg_b2b",
                "surge_rate",
                "avg_visits",
                "dead_rate",
                "elo_learner",
                "elo_best_pool",
                "elo_learner_minus_ref",
                "elo_gap_to_pool",
            ],
            "checkpoint_default": "checkpoints/1v1_placement_az",
            "value_target": "outcome z in {-1, 0, +1} (tanh value head)",
            "search_shaping": "w_attack=0.05, w_b2b=0.06, w_death=1, gamma=1, return_scale=1",
        },
        "red_flags": [
            "policy_kl or update_kl exploding -> optimization divergence",
            "explained_var ~0 or negative with high value_loss -> value head not learning",
            "win_rate_vs_ref stuck ~0.5 while learner elo drifts -> pool/eval mismatch",
            "dead_rate high in 1v1 -> search/death penalty or board-death issues",
            "return_scale NaN/Inf in single AZ inspect_checkpoint -> bad resume",
        ],
        "observability_code": "src/qtris/observability/models.py",
    }
