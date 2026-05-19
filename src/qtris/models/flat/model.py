import tensorflow as tf
import keras
from keras import layers
from tensorflow_probability import distributions
from qtris.nn.transformer import DecoderLayer, PosEncoding
from qtris.models.encoders import make_patches, tokenize_bcg
from qtris.models.base import QtrisModelBase
from qtris.models.value import ValueModel  # noqa: F401 - re-exported for trainers
from TetrisEnv.Moves import Keys

HARD_DROP_ID = Keys.HARD_DROP


class FlatPolicyModel(QtrisModelBase):
    def __init__(
        self,
        batch_size,
        piece_dim,
        depth,
        num_heads,
        num_layers,
        dropout_rate,
        num_sequences=160,
    ):
        super().__init__()

        self._batch_size = batch_size
        self._depth = depth
        self._num_sequences = num_sequences

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

        self._bcg_proj_b2b = layers.Dense(depth, activation=None, name="bcg_proj_b2b")
        self._bcg_proj_combo = layers.Dense(depth, activation=None, name="bcg_proj_combo")
        self._bcg_proj_garbage = layers.Dense(depth, activation=None, name="bcg_proj_garbage")
        self._bcg_ln = layers.LayerNormalization(name="bcg_ln")

        self.trunk = keras.Sequential(
            [
                layers.Flatten(),
                layers.Dropout(dropout_rate),
                layers.Dense(4 * depth, activation="relu"),
                layers.Dense(2 * depth, activation="relu"),
                layers.Dense(depth, activation="relu"),
            ],
            name="trunk_layers",
        )

        self.top = layers.Dense(num_sequences, name="action_logits")

    @tf.function(jit_compile=True)
    def score_actions(self, piece_dec, training=False):
        trunk_out = self.trunk(piece_dec, training=training)
        logits = self.top(trunk_out, training=training)
        return logits

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, return_scores=False):
        board, piece, b2b_combo_garbage = inputs

        piece_dec, piece_scores = self.process_obs(
            (board, piece, b2b_combo_garbage), training=training
        )

        logits = self.score_actions(piece_dec, training=training)

        if return_scores:
            return logits, piece_scores
        else:
            return logits

    @tf.function(
        jit_compile=True,
        input_signature=[
            (
                tf.TensorSpec(shape=(None, 24, 10, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 7), dtype=tf.int64),
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=None, dtype=tf.bool),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.float32),
        ],
    )
    def predict(self, inputs, greedy=False, valid_sequences=None, temperature=1.0):
        piece_dec, piece_scores = self.process_obs(inputs, training=False)

        logits = self.score_actions(piece_dec, training=False)

        valid_mask = tf.reduce_any(
            tf.equal(valid_sequences, tf.constant(HARD_DROP_ID, dtype=tf.int64)),
            axis=-1,
        )
        masked_logits = tf.where(
            valid_mask, logits / temperature, tf.constant(-1e9, dtype=tf.float32)
        )

        dist = distributions.Categorical(logits=masked_logits, dtype=tf.int64)

        if greedy:
            action_index = tf.argmax(masked_logits, axis=-1, output_type=tf.int64)
        else:
            action_index = dist.sample()

        log_prob = dist.log_prob(action_index)

        batch_indices = tf.range(tf.shape(valid_sequences)[0], dtype=tf.int64)
        selected_sequence = tf.gather_nd(
            valid_sequences,
            tf.stack([batch_indices, action_index], axis=1),
        )

        return selected_sequence, log_prob, action_index, piece_scores
