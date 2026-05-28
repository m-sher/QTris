"""Shared `keras.Model` base for QTris policy/value architectures.

Hosts the `process_obs` method that's identical across PolicyModel,
ValueModel, AsymmetricValueModel, and FlatPolicyModel. The base class
owns no Variables; it assumes the subclass `__init__` creates the
submodule attributes the method reads:
    self.make_patches, self.patch_pos_encoding,
    self.piece_embedding, self.piece_pos_encoding,
    self.board_decoder_layers, self.piece_decoder_layers,
    self._bcg_proj_b2b / _bcg_proj_combo / _bcg_proj_garbage / _bcg_ln
    (read by self._tokenize_bcg, which subclasses define as a one-line
    wrapper around qtris.models.encoders.tokenize_bcg).
"""

import tensorflow as tf
import keras

from qtris.models.encoders import tokenize_bcg


class QtrisModelBase(keras.Model):
    def _tokenize_bcg(self, b2b_combo_garbage, training=False):
        return tokenize_bcg(
            b2b_combo_garbage,
            proj_b2b=self._bcg_proj_b2b,
            proj_combo=self._bcg_proj_combo,
            proj_garbage=self._bcg_proj_garbage,
            ln=self._bcg_ln,
            training=training,
        )

    @tf.function(
        jit_compile=True,
        input_signature=[
            (
                tf.TensorSpec(shape=(None, 24, 10, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        ],
    )
    def process_obs(self, inputs, training=False):
        board, piece, b2b_combo_garbage = inputs

        piece_scores = []
        patches = self.make_patches(board, training=training)
        board_dec = self.patch_pos_encoding(patches)

        piece_embedding = self.piece_embedding(piece, training=training)
        piece_dec = self.piece_pos_encoding(piece_embedding)

        bcg_tokens = self._tokenize_bcg(b2b_combo_garbage, training=training)
        piece_dec = tf.concat([piece_dec, bcg_tokens], axis=1)
        board_dec = tf.concat([board_dec, bcg_tokens], axis=1)

        for board_dec_layer, piece_dec_layer in zip(
            self.board_decoder_layers, self.piece_decoder_layers
        ):
            board_dec, last_board_attn = board_dec_layer(
                [piece_dec, board_dec], training=training
            )
            piece_dec, last_piece_attn = piece_dec_layer(
                [board_dec, piece_dec], training=training
            )
            piece_scores.append(last_piece_attn)

        return piece_dec, piece_scores
