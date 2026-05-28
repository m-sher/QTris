"""Tetris-specific shared encoders: board CNN + BCG tokenizer.

These were duplicated 4x across PolicyModel / ValueModel / AsymmetricValueModel
/ FlatPolicyModel before Phase 2. They are NOT generic NN code (they know the
board is 24x10x1 and the BCG state is a 3-tuple of scalars), so they live under
`qtris.models.` rather than `qtris.nn.`.
"""

import tensorflow as tf
import keras
from keras import layers


def make_patches(depth: int) -> keras.Sequential:
    """Board (24,10,1) -> sequence of `depth`-dim patches.

    Identical across all AR + Flat models. The Sequential's inner layers
    use auto-generated names; their variable scopes match the pre-Phase-2
    inline definitions as long as the global Keras name counter is in the
    same state when this is called from `__init__`.
    """
    return keras.Sequential(
        [
            keras.Input(shape=(24, 10, 1)),
            layers.Rescaling(scale=2.0, offset=-1.0),
            layers.Conv2D(
                filters=depth // 2,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
            ),
            layers.Conv2D(
                filters=depth,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
            ),
            layers.Conv2D(
                filters=depth,
                kernel_size=2,
                strides=2,
                padding="valid",
                activation="relu",
            ),
            layers.Reshape((-1, depth)),
        ]
    )


def tokenize_bcg(
    b2b_combo_garbage,
    *,
    proj_b2b,
    proj_combo,
    proj_garbage,
    ln,
    training=False,
):
    """Log-compress + project a (B, 3) BCG tuple into (B, 3, depth) tokens.

    Per-feature projections + layernorm passed as kwargs so each model owns
    its own submodules (AR uses 'relu' activation, Flat uses None) and the
    variable graph stays under the model's existing attribute names
    (`_bcg_proj_b2b`, `_bcg_proj_combo`, `_bcg_proj_garbage`, `_bcg_ln`).
    """
    bcg_log = tf.math.log1p(b2b_combo_garbage + 1.0)  # (B, 3)
    t_b2b = proj_b2b(bcg_log[:, 0:1], training=training)
    t_combo = proj_combo(bcg_log[:, 1:2], training=training)
    t_garbage = proj_garbage(bcg_log[:, 2:3], training=training)
    return ln(
        tf.stack([t_b2b, t_combo, t_garbage], axis=1), training=training
    )  # (B, 3, depth)
