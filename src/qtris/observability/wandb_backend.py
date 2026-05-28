"""Thin wrapper around wandb that takes typed pydantic models in/out.

`init_run` converts a TrainConfig pydantic model to wandb.config; `log_step`
serializes a PPOLog model (incl. wrapping numpy image fields as wandb.Image)
and emits a single `wandb.log` call. Callers pass typed models, not dicts.
"""

from __future__ import annotations

from typing import Any

import wandb

from qtris.observability.models import PPOConfigBase, PPOLogBase


def init_run(*, project: str, config: PPOConfigBase, **kwargs: Any):
    """Open a wandb run. Returns the run handle so callers can `.finish()`."""
    return wandb.init(project=project, config=config.dict(), **kwargs)


def log_step(metrics: PPOLogBase, *, step: int | None = None) -> None:
    payload = metrics.to_wandb_payload()
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


def finish(run) -> None:
    run.finish()
