"""Value head model. Shared between AR and Flat training.

Lives at `qtris.models.value` (not under `ar/`) so that `flat/model.py`
doesn't need to cross-import from `ar/model.py`.
"""

import tensorflow as tf
import keras
from keras import layers

from qtris.nn.transformer import DecoderLayer, PosEncoding
from qtris.models.base import QtrisModelBase
from qtris.models.encoders import make_patches


class ValueModel(QtrisModelBase):
    def __init__(
        self, piece_dim, depth, num_heads, num_layers, dropout_rate, output_dim
    ):
        super().__init__()

        self._depth = depth

        self.make_patches = make_patches(depth)

        num_patches = self.make_patches.output_shape[1]
        self.patch_pos_encoding = PosEncoding(depth=depth, max_length=num_patches)

        self.board_decoder_layers = [
            DecoderLayer(
                units=depth,
                causal=False,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                name=f"board_dec_{i}",
            )
            for i in range(num_layers)
        ]

        self.piece_embedding = layers.Embedding(
            input_dim=piece_dim,
            output_dim=depth,
        )

        self.piece_pos_encoding = PosEncoding(
            depth=depth,
            max_length=7,
        )

        self.piece_decoder_layers = [
            DecoderLayer(
                units=depth,
                causal=False,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                name=f"piece_dec_{i}",
            )
            for i in range(num_layers)
        ]

        self._bcg_proj_b2b = layers.Dense(depth, activation="relu", name="bcg_proj_b2b")
        self._bcg_proj_combo = layers.Dense(
            depth, activation="relu", name="bcg_proj_combo"
        )
        self._bcg_proj_garbage = layers.Dense(
            depth, activation="relu", name="bcg_proj_garbage"
        )
        self._bcg_ln = layers.LayerNormalization(name="bcg_ln")

        self.trunk_bcg = keras.Sequential(
            [
                layers.Flatten(),
                layers.Dropout(dropout_rate),
                layers.Dense(depth, activation="relu"),
                layers.Dense(depth // 2, activation="relu"),
            ],
            name="trunk_bcg",
        )

        self.top = layers.Dense(output_dim)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, return_scores=False):
        board, piece, b2b_combo_garbage = inputs

        piece_dec, piece_scores = self.process_obs(
            (board, piece, b2b_combo_garbage), training=training
        )

        trunk_out = self.trunk_bcg(piece_dec, training=training)

        output = self.top(trunk_out, training=training)

        if return_scores:
            return output, piece_scores
        else:
            return output

    @tf.function(
        jit_compile=True,
        input_signature=[
            (
                tf.TensorSpec(shape=(None, 24, 10, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            ),
        ],
    )
    def predict(self, inputs):
        board, piece, b2b_combo_garbage = inputs

        piece_dec, piece_scores = self.process_obs(
            (board, piece, b2b_combo_garbage), training=False
        )

        trunk_out = self.trunk_bcg(piece_dec, training=False)

        output = self.top(trunk_out, training=False)

        return output
