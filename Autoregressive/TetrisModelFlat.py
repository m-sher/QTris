import tensorflow as tf
import keras
from keras import layers
from tensorflow_probability import distributions
from TetrisModel import PosEncoding, DecoderLayer, CrossAttentionLayer, ValueModel
from TetrisEnv.Moves import Keys

HARD_DROP_ID = Keys.HARD_DROP


class FlatPolicyModel(keras.Model):
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

        self.make_patches = keras.Sequential(
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

        self._bcg_dense = keras.Sequential(
            [
                layers.Dense(depth // 2, activation="relu"),
                layers.Dense(depth, activation="relu"),
            ],
            name="bcg_dense",
        )

        self.action_embedding = layers.Embedding(
            input_dim=num_sequences,
            output_dim=depth,
        )

        self.action_cross_attn_layers = [
            CrossAttentionLayer(
                units=depth,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                name=f"action_ca_{i}",
            )
            for i in range(num_layers)
        ]

        self.action_proj = layers.Dense(1, name="action_proj")

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

        bcg_embedding = self._bcg_dense(b2b_combo_garbage, training=training)
        piece_dec += bcg_embedding[:, None, :]

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

    @tf.function(jit_compile=True)
    def score_actions(self, piece_dec, training=False):
        batch_size = tf.shape(piece_dec)[0]
        action_ids = tf.broadcast_to(
            tf.range(self._num_sequences, dtype=tf.int64)[None, :],
            [batch_size, self._num_sequences],
        )
        action_dec = self.action_embedding(action_ids, training=training)

        for action_ca_layer in self.action_cross_attn_layers:
            action_dec, _ = action_ca_layer(
                [piece_dec, action_dec], training=training
            )

        logits = tf.squeeze(
            self.action_proj(action_dec, training=training), axis=-1
        )
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
