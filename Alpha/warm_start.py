"""Warm-start an AlphaModel from a trained autoregressive PolicyModel checkpoint.

The two models share the same encoder architecture (board patches + cross-
attention with piece+BCG tokens) but have different heads.  This loader:

  1. Builds a sibling QTris/Autoregressive/TetrisModel.PolicyModel with the
     same depth/num_heads/num_layers/piece_dim as the destination AlphaModel.
  2. Runs one dummy forward pass on each to allocate weight variables.
  3. Restores the PolicyModel checkpoint via tf.train.CheckpointManager.
  4. Copies the encoder layer weights into the AlphaModel one component at
     a time (head weights stay fresh — they're shape-incompatible anyway).

The encoder dimensions in the checkpoint MUST match the destination model.
If they don't, set_weights() raises a clear shape-mismatch error that
identifies the offending layer.
"""

from __future__ import annotations
import os
from typing import Optional

import keras
import tensorflow as tf

from .network import AlphaModel


# Names of the encoder attributes that AlphaModel and PolicyModel share.
# Each is a single layer object whose weights are interchangeable when the
# model dimensions match.
_SHARED_SCALAR_LAYERS = (
    "make_patches",
    "patch_pos_encoding",
    "piece_embedding",
    "piece_pos_encoding",
    "_bcg_proj_b2b",
    "_bcg_proj_combo",
    "_bcg_proj_garbage",
    "_bcg_ln",
)

# These are lists; weights are copied position-by-position.
_SHARED_LAYER_STACKS = (
    "board_decoder_layers",
    "piece_decoder_layers",
)


def _allocate_alpha(model: AlphaModel) -> None:
    """Run one tiny forward pass so AlphaModel's variables are created."""
    board = tf.zeros((1, 24, 10, 1), dtype=tf.float32)
    pieces = tf.zeros((1, 7), dtype=tf.int64)
    bcg = tf.zeros((1, 3), dtype=tf.float32)
    _ = model((board, pieces, bcg), training=False)


def _allocate_policy(policy_model) -> None:
    """Run one tiny forward pass so PolicyModel's variables are created."""
    # PolicyModel.process_obs takes (board, pieces, bcg) + training tensor.
    board = tf.zeros((1, 24, 10, 1), dtype=tf.float32)
    pieces = tf.zeros((1, 7), dtype=tf.int64)
    bcg = tf.zeros((1, 3), dtype=tf.float32)
    _ = policy_model.process_obs((board, pieces, bcg), tf.constant(False))


def warm_start_alpha_from_autoregressive(
    alpha_model: AlphaModel,
    ckpt_dir: str,
    *,
    piece_dim: int = 8,
    key_dim: int = 12,
    depth: Optional[int] = None,
    max_len: int = 15,
    num_heads: Optional[int] = None,
    num_layers: Optional[int] = None,
    dropout_rate: float = 0.0,
    output_dim: int = 12,
    log_fn=print,
) -> None:
    """Copy encoder weights from a saved PolicyModel checkpoint into alpha_model.

    Args:
      alpha_model:   destination AlphaModel (will have its encoder weights replaced).
      ckpt_dir:      directory containing a tf.train.CheckpointManager checkpoint
                     of a PolicyModel (or PolicyModel + optimizer; only `model` is read).
      piece_dim, key_dim, depth, max_len, num_heads, num_layers, dropout_rate,
      output_dim:    PolicyModel.__init__ args.  `depth`, `num_heads`, `num_layers`
                     default to the destination AlphaModel's geometry.

    Heads (policy / value) are NOT touched — they keep their fresh
    initialization.  Use `model.summary()` after the call to verify the
    encoder layers updated as expected.
    """
    # Default the encoder geometry to the destination model's so callers
    # don't have to repeat themselves.
    if depth is None:
        depth = alpha_model._depth
    # AlphaModel constructor stashes num_heads/num_layers via the layer count
    # — derive from the actual list lengths to be safe.
    num_layers = num_layers if num_layers is not None else len(alpha_model.board_decoder_layers)
    if num_heads is None:
        num_heads = 4  # AlphaModel default; doesn't affect weight shapes once depth is fixed.

    # Late import — keeps Alpha self-contained when no warm-start is needed.
    from Autoregressive.TetrisModel import PolicyModel

    log_fn(f"[warm-start] building reference PolicyModel "
           f"(depth={depth}, heads={num_heads}, layers={num_layers}, piece_dim={piece_dim})")
    policy_model = PolicyModel(
        batch_size=1,
        piece_dim=piece_dim,
        key_dim=key_dim,
        depth=depth,
        max_len=max_len,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        output_dim=output_dim,
    )

    # Allocate variables on both models.
    _allocate_alpha(alpha_model)
    _allocate_policy(policy_model)

    # Restore PolicyModel weights.  We don't need its optimizer.
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"warm-start checkpoint dir not found: {ckpt_dir}")
    ckpt = tf.train.Checkpoint(model=policy_model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    if manager.latest_checkpoint is None:
        raise FileNotFoundError(
            f"no checkpoint found in {ckpt_dir} (CheckpointManager.latest_checkpoint is None)"
        )
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    log_fn(f"[warm-start] restored PolicyModel from {manager.latest_checkpoint}")

    # Copy shared encoder layers, one at a time so a shape mismatch points
    # to a specific component.
    transferred = 0
    for name in _SHARED_SCALAR_LAYERS:
        src = getattr(policy_model, name)
        dst = getattr(alpha_model, name)
        try:
            dst.set_weights(src.get_weights())
        except ValueError as e:
            raise ValueError(
                f"[warm-start] shape mismatch on layer `{name}`: {e}\n"
                f"Hint: the checkpoint's encoder geometry must match "
                f"alpha_model (depth={depth}, num_layers={num_layers})."
            ) from e
        transferred += 1

    for stack_name in _SHARED_LAYER_STACKS:
        src_stack = getattr(policy_model, stack_name)
        dst_stack = getattr(alpha_model, stack_name)
        if len(src_stack) != len(dst_stack):
            raise ValueError(
                f"[warm-start] {stack_name} length mismatch: "
                f"checkpoint has {len(src_stack)}, alpha_model has {len(dst_stack)}. "
                f"Use --layers={len(src_stack)} or rebuild the checkpoint."
            )
        for i, (src, dst) in enumerate(zip(src_stack, dst_stack)):
            try:
                dst.set_weights(src.get_weights())
            except ValueError as e:
                raise ValueError(
                    f"[warm-start] shape mismatch on `{stack_name}[{i}]`: {e}"
                ) from e
            transferred += 1

    log_fn(f"[warm-start] transferred {transferred} encoder layers; heads kept fresh")
