"""Load PlacementPolicyValueNet checkpoints for MCP probes (lazy TF import)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Mode = Literal["single", "1v1"]


@dataclass
class LoadedNet:
    net: Any
    mode: Mode
    checkpoint: str | None
    return_scale: float | None
    extra: dict[str, Any]


def _tf_keras():
    import tensorflow as tf
    from tensorflow import keras

    return tf, keras


def build_net(
    *,
    mode: Mode = "1v1",
    batch_size: int = 1,
    piece_dim: int = 8,
    depth: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    queue_size: int = 5,
):
    """Build an untrained PlacementPolicyValueNet matching trainer shapes."""
    tf, keras = _tf_keras()
    from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
    from qtris.models.placement.model import PlacementPolicyValueNet

    value_activation = "tanh" if mode == "1v1" else None
    net = PlacementPolicyValueNet(
        batch_size=batch_size,
        piece_dim=piece_dim,
        depth=depth,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=0.0,
        value_activation=value_activation,
    )
    net(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(queue_size + 2,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(
                shape=(CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM), dtype=tf.float32
            ),
            keras.Input(shape=(CANDIDATE_CAPACITY,), dtype=tf.bool),
        )
    )
    return net


def load_checkpoint(
    checkpoint_dir: Path,
    *,
    mode: Mode = "1v1",
    batch_size: int = 1,
    compile_net: bool = False,
) -> LoadedNet:
    """Restore latest ckpt from a trainer checkpoint dir (or explicit prefix)."""
    tf, keras = _tf_keras()
    net = build_net(mode=mode, batch_size=batch_size)
    if compile_net:
        net.compile(
            optimizer=keras.optimizers.Adam(1e-4, clipnorm=0.5),
            jit_compile=True,
        )

    extra: dict[str, Any] = {}
    return_scale_val: float | None = None
    ckpt_path: str | None = None

    # Allow either a dir with CheckpointManager layout or an explicit prefix path.
    if checkpoint_dir.is_dir():
        ckpt_path = tf.train.latest_checkpoint(str(checkpoint_dir))
    else:
        # prefix like .../ckpt-1470 or .../pool/gen_0
        if (
            checkpoint_dir.with_suffix(".index").exists()
            or Path(str(checkpoint_dir) + ".index").exists()
        ):
            ckpt_path = str(checkpoint_dir)
        else:
            parent = checkpoint_dir.parent
            if parent.is_dir():
                ckpt_path = tf.train.latest_checkpoint(str(parent))

    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint under {checkpoint_dir}")

    if mode == "single":
        return_scale = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        ck = tf.train.Checkpoint(model=net, return_scale=return_scale)
        status = ck.restore(ckpt_path)
        _record_restore(status, extra)
        status.expect_partial()
        return_scale_val = float(return_scale.numpy())
        extra["return_scale"] = return_scale_val
    else:
        # 1v1 trainer stores model (+ optimizer in full ckpts); pool snaps are weights-only.
        if "pool/gen_" in ckpt_path.replace("\\", "/"):
            status = net.load_weights(ckpt_path)
            _record_restore(status, extra)
            status.expect_partial()
            extra["pool_snapshot"] = True
        else:
            ck = tf.train.Checkpoint(model=net)
            status = ck.restore(ckpt_path)
            _record_restore(status, extra)
            status.expect_partial()
            extra["pool_snapshot"] = False

    extra["checkpoint_path"] = ckpt_path
    return LoadedNet(
        net=net,
        mode=mode,
        checkpoint=ckpt_path,
        return_scale=return_scale_val,
        extra=extra,
    )


def _record_restore(status, extra: dict[str, Any]) -> None:
    """Flag whether every net variable matched the checkpoint (catches arch/weight drift).

    expect_partial() silences unmatched names, so a wrong-arch or policy-only ckpt would
    otherwise load a half-random net and report it as healthy.
    """
    try:
        status.assert_existing_objects_matched()
        extra["restore_matched"] = True
    except Exception as e:
        extra["restore_matched"] = False
        extra["restore_warning"] = f"{type(e).__name__}: {e}"


def param_stats(net) -> dict[str, Any]:
    """Lightweight weight health summary (NaN/Inf + norms)."""
    import numpy as np

    total = 0
    nan_vars = []
    inf_vars = []
    norms = []
    for v in net.trainable_variables:
        arr = v.numpy()
        total += arr.size
        name = v.name
        if np.isnan(arr).any():
            nan_vars.append(name)
        if np.isinf(arr).any():
            inf_vars.append(name)
        norms.append(float(np.linalg.norm(arr.astype(np.float64))))
    return {
        "num_trainable_vars": len(net.trainable_variables),
        "num_params": int(total),
        "nan_vars": nan_vars,
        "inf_vars": inf_vars,
        "weight_norm_mean": float(np.mean(norms)) if norms else 0.0,
        "weight_norm_max": float(np.max(norms)) if norms else 0.0,
        "weight_norm_min": float(np.min(norms)) if norms else 0.0,
    }
