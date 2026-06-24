"""Repo-root resolution and default AZ artifact paths."""

from __future__ import annotations

import os
from pathlib import Path

# Defaults match training/*_placement_az.py and README.
DEFAULT_SINGLE_CKPT = "checkpoints/placement_az"
DEFAULT_1V1_CKPT = "checkpoints/1v1_placement_az"
DEFAULT_PRETRAIN_CKPT = "checkpoints/placement_pretrained_policy"
DEFAULT_TB_ROOT = "tb_logs/Tetris"
DEFAULT_AZ_STATES = "data/az_states"
DEFAULT_GARBAGE_TRACES = "garbage_traces"

# Phase / variant dirs agents commonly inspect.
AZ_CKPT_GLOBS = (
    "checkpoints/placement_az",
    "checkpoints/1v1_placement_az",
    "checkpoints/1v1_placement_az_phase*",
    "checkpoints/placement_pretrained_policy",
)


def repo_root() -> Path:
    """Walk up from this file to the project root (has pyproject.toml + checkpoints/)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "qtris").is_dir():
            return parent
    # Fallback: cwd when launched via `uv run qtris-mcp` from repo root.
    return Path.cwd()


def resolve_path(path: str | None, default: str) -> Path:
    """Resolve a user path relative to repo root unless absolute."""
    root = repo_root()
    p = Path(path) if path else Path(default)
    if not p.is_absolute():
        p = root / p
    return p


def dir_size_mb(path: Path) -> float:
    """Recursive on-disk size in MB (unreadable files skipped)."""
    if not path.exists():
        return 0.0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return round(total / (1024 * 1024), 2)
