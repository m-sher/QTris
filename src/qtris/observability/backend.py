"""TensorBoard observability backend (tf.summary), with optional wandb mirroring.

`init_run` opens a file writer under tb_logs/<project>/<timestamp> and logs the
config once as markdown text; `log_step` writes one scalar per numeric field and
one image per `_image_fields` entry of the typed payload model. With
wandb_mirror=True, `wandb.init(sync_tensorboard=True)` is called before the
writer is created so wandb mirrors the event stream to the cloud.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from pydantic import BaseModel

from qtris.observability.models import LogPayloadModel


@dataclass
class Run:
    writer: Any
    logdir: str
    step: int = 0
    wandb_run: Any = None


_current_run: Run | None = None


def _config_markdown(config: BaseModel) -> str:
    rows = [f"| {key} | {val} |" for key, val in config.dict().items()]
    return "\n".join(["| key | value |", "| --- | --- |", *rows])


def _to_image_batch(arr) -> np.ndarray:
    """Normalize a logged array to the (1, H, W, C) batch tf.summary.image expects."""
    img = np.asarray(arr)
    if img.ndim == 2:
        img = img[..., None]
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        peak = float(img.max())
        if peak > 1.0:
            img = img / peak
    return img[None, ...]


def init_run(
    *,
    project: str,
    config: BaseModel,
    wandb_mirror: bool = False,
    run_name: str | None = None,
) -> Run:
    """Open a TensorBoard run (plus a wandb mirror when requested)."""
    global _current_run

    wandb_run = None
    if wandb_mirror:
        # Must run before the file writer is created so wandb can patch the
        # event writer and mirror everything written to it.
        import wandb

        wandb_run = wandb.init(
            project=project, config=config.dict(), sync_tensorboard=True
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    logdir = (
        f"tb_logs/{project}/{stamp}-{run_name}"
        if run_name
        else f"tb_logs/{project}/{stamp}"
    )
    writer = tf.summary.create_file_writer(logdir)
    with writer.as_default():
        tf.summary.text("config", _config_markdown(config), step=0)
    writer.flush()
    print(f"TensorBoard logdir: {logdir}", flush=True)

    _current_run = Run(writer=writer, logdir=logdir, wandb_run=wandb_run)
    return _current_run


def log_step(metrics: LogPayloadModel, *, step: int | None = None) -> None:
    run = _current_run
    if step is None:
        step = run.step
    run.step = step + 1

    payload = metrics.to_payload()
    group_of = {f: g for g, fields in metrics._tag_groups.items() for f in fields}
    with run.writer.as_default():
        for key in metrics._image_fields:
            img = payload.pop(key, None)
            if img is not None:
                tf.summary.image(key, _to_image_batch(img), step=step)
        for key, val in payload.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                tag = f"{group_of[key]}/{key}" if key in group_of else key
                tf.summary.scalar(tag, float(val), step=step)
    run.writer.flush()


def finish(run: Run) -> None:
    run.writer.flush()
    run.writer.close()
    if run.wandb_run is not None:
        run.wandb_run.finish()
