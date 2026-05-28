import tensorflow as tf
import keras
from keras import layers
from tensorflow_probability import distributions
from TetrisEnv.Moves import Convert

from qtris.nn.attention import CrossAttention, FeedForward, SelfAttention
from qtris.nn.transformer import (
    CrossAttentionLayer,
    DecoderLayer,
    EncoderLayer,
    PosEncoding,
    positional_encoding,
)
from qtris.models.encoders import make_patches, tokenize_bcg
from qtris.models.base import QtrisModelBase


class PolicyModel(QtrisModelBase):
    def __init__(
        self,
        batch_size,
        piece_dim,
        key_dim,
        depth,
        max_len,
        num_heads,
        num_layers,
        dropout_rate,
        output_dim,
    ):
        super().__init__()

        self._batch_size = batch_size
        self._key_dim = key_dim
        self._depth = depth
        self._max_len = max_len

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

        self.key_embedding = layers.Embedding(
            input_dim=key_dim,
            output_dim=depth,
        )

        self.key_pos_encoding = PosEncoding(
            depth=depth,
            max_length=max_len,
        )

        self.key_decoder_layers = [
            DecoderLayer(
                units=depth,
                causal=True,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                name=f"key_dec_{i}",
            )
            for i in range(num_layers)
        ]

        self.trunk = keras.Sequential(
            [
                layers.Dropout(dropout_rate),
                layers.Dense(depth, activation="relu"),
                layers.Dense(depth // 2, activation="relu"),
            ],
            name="trunk_layers",
        )

        self.top = layers.Dense(output_dim)

    @tf.function(jit_compile=True)
    def process_keys(self, inputs, training=False):
        piece_dec, keys = inputs

        key_scores = []
        key_embedding = self.key_embedding(keys, training=training)
        key_dec = self.key_pos_encoding(key_embedding)

        for key_dec_layer in self.key_decoder_layers:
            key_dec, last_attn = key_dec_layer([piece_dec, key_dec], training=training)
            key_scores.append(last_attn)

        trunk_out = self.trunk(key_dec, training=training)

        output = self.top(trunk_out, training=training)

        return output, key_scores

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, return_scores=False):
        board, piece, b2b_combo_garbage, keys = inputs

        piece_dec, piece_scores = self.process_obs(
            (board, piece, b2b_combo_garbage), training=training
        )

        output, key_scores = self.process_keys((piece_dec, keys), training=training)

        if return_scores:
            return output, piece_scores, key_scores
        else:
            return output

    @tf.function(jit_compile=True)
    def _generate_next_key(
        self,
        ind,
        piece_dec,
        key_sequence,
        log_probs,
        masks,
        valid_sequences,
        temperature,
        greedy=False,
    ):
        def generate_mask(ind, stacked_key_sequence, sequences):
            matching_sequence = tf.reduce_all(
                stacked_key_sequence[:, None, :ind] == sequences[:, :, :ind],
                axis=-1,
            )[..., None]
            next_keys = sequences[:, :, ind][..., None]
            possible_keys = tf.range(self._key_dim, dtype=tf.int64)[None, None, ...]
            valid = tf.reduce_any(
                tf.logical_and(matching_sequence, next_keys == possible_keys), axis=1
            )
            return valid

        stacked_key_sequence = tf.transpose(
            key_sequence.stack(), perm=[1, 0]
        )  # len, batch -> batch, len
        mask = generate_mask(ind, stacked_key_sequence, valid_sequences)
        logits, _ = self.process_keys((piece_dec, stacked_key_sequence), training=False)

        temp_adjusted_logits = logits / temperature

        masked_logits = tf.where(
            mask,
            temp_adjusted_logits[:, ind - 1, :],
            tf.constant(-1e9, dtype=tf.float32),
        )

        dist = distributions.Categorical(logits=masked_logits, dtype=tf.int64)

        if greedy:
            action = tf.argmax(masked_logits, axis=-1, output_type=tf.int64)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        key_sequence = key_sequence.write(ind, action)
        log_probs = log_probs.write(ind, log_prob)
        masks = masks.write(ind, mask)

        return ind + 1, key_sequence, log_probs, masks

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
        valid_sequences = tf.convert_to_tensor(valid_sequences, dtype=tf.int64)

        key_sequence = tf.TensorArray(
            dtype=tf.int64,
            size=self._max_len,
            dynamic_size=False,
            element_shape=(self._batch_size,),
        )
        log_probs = tf.TensorArray(
            dtype=tf.float32,
            size=self._max_len,
            dynamic_size=False,
            element_shape=(self._batch_size,),
        )
        masks = tf.TensorArray(
            dtype=tf.bool,
            size=self._max_len,
            dynamic_size=False,
            element_shape=(self._batch_size, self._key_dim),
        )

        # Ind starts at 1 because 0 is the START key
        ind = tf.constant(1, dtype=tf.int32)
        ind, key_sequence, log_probs, masks = tf.while_loop(
            lambda i, ks, lp, m: tf.less(i, self._max_len),
            lambda i, ks, lp, m: self._generate_next_key(
                i, piece_dec, ks, lp, m, valid_sequences, temperature, greedy
            ),
            [ind, key_sequence, log_probs, masks],
            parallel_iterations=1,
        )

        key_sequence = tf.transpose(
            key_sequence.stack(), perm=[1, 0]
        )  # len, batch -> batch, len
        log_probs = tf.transpose(
            log_probs.stack(), perm=[1, 0]
        )  # len, batch -> batch, len
        masks = tf.transpose(
            masks.stack(), perm=[1, 0, 2]
        )  # len, batch, key_dim -> batch, len, key_dim

        return key_sequence, log_probs, masks, piece_scores


class AsymmetricValueModel(QtrisModelBase):
    """Value model that sees both the training player's and opponent's board state.

    Own board processing is identical to ValueModel (conv → cross-attention decoders).
    Opponent board is processed through the shared conv encoder, a separate bcg encoder,
    and a lightweight self-attention encoder, then mean-pooled to a fixed-size vector.
    The two representations are concatenated and fed to a larger trunk.
    """

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

        self.trunk_a_bcg = keras.Sequential(
            [layers.Flatten(), layers.Dense(depth, activation="relu")]
        )

        self.trunk_b_bcg = keras.Sequential(
            [layers.Flatten(), layers.Dense(depth, activation="relu")]
        )

        self.top = keras.Sequential(
            [
                layers.Concatenate(),
                layers.Dropout(dropout_rate),
                layers.Dense(depth, activation="relu"),
                layers.Dense(depth // 2, activation="relu"),
                layers.Dense(output_dim),
            ],
            name="top",
        )

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, return_scores=False):
        board_a, piece_a, bcg_a, board_b, piece_b, bcg_b = inputs

        piece_dec_a, piece_scores_a = self.process_obs(
            (board_a, piece_a, bcg_a), training=training
        )

        piece_dec_b, piece_scores_b = self.process_obs(
            (board_b, piece_b, bcg_b), training=training
        )

        trunk_out_a = self.trunk_a_bcg(piece_dec_a, training=training)
        trunk_out_b = self.trunk_b_bcg(piece_dec_b, training=training)

        top_out_a = self.top((trunk_out_a, trunk_out_b), training=training)
        top_out_b = self.top((trunk_out_b, trunk_out_a), training=training)

        output = 0.5 * (top_out_a - top_out_b)

        if return_scores:
            return output, piece_scores_a, piece_scores_b
        else:
            return output

    @tf.function(
        jit_compile=True,
        input_signature=[
            (
                tf.TensorSpec(shape=(None, 24, 10, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 24, 10, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            ),
        ],
    )
    def predict(self, inputs):
        board_a, piece_a, bcg_a, board_b, piece_b, bcg_b = inputs

        piece_dec_a, piece_scores_a = self.process_obs(
            (board_a, piece_a, bcg_a), training=False
        )

        piece_dec_b, piece_scores_b = self.process_obs(
            (board_b, piece_b, bcg_b), training=False
        )

        trunk_out_a = self.trunk_a_bcg(piece_dec_a, training=False)
        trunk_out_b = self.trunk_b_bcg(piece_dec_b, training=False)

        top_out_a = self.top((trunk_out_a, trunk_out_b), training=False)
        top_out_b = self.top((trunk_out_b, trunk_out_a), training=False)

        output = 0.5 * (top_out_a - top_out_b)

        return output
