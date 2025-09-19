import tensorflow as tf
import keras
from keras import layers
from tensorflow_probability import distributions
from TetrisEnvs.TFTetrisEnv.TFMoves import TFConvert


@tf.function(jit_compile=True)
def positional_encoding(max_length, depth):
    half_depth = depth / 2
    positions = tf.range(max_length, dtype=tf.float32)[..., None]
    depths = tf.range(half_depth, dtype=tf.float32)[None, ...] / half_depth

    angle_rates = 1.0 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
    return pos_encoding


class PosEncoding(layers.Layer):
    def __init__(self, depth, max_length):
        super().__init__()
        self.depth = depth

        self.pos_encoding = positional_encoding(max_length, depth)

    @tf.function(jit_compile=True)
    def call(self, seq):
        # Already embedded seq
        seq_len = tf.shape(seq)[1]

        seq *= tf.cast(self.depth, tf.float32) ** 0.5
        seq += self.pos_encoding[None, :seq_len]

        return seq


class SelfAttention(layers.Layer):
    def __init__(self, causal, **kwargs):
        super().__init__()
        self.causal = causal
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    @tf.function(jit_compile=True)
    def call(self, x, training=False):
        attn, attention_scores = self.mha(
            query=x, value=x, use_causal_mask=self.causal, return_attention_scores=True
        )
        x = self.add([x, attn])
        return self.layernorm(x, training=training), attention_scores


class CrossAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    @tf.function(jit_compile=True)
    def call(self, x, y, training=False, **kwargs):
        attn, attention_scores = self.mha(
            query=x, value=y, return_attention_scores=True
        )

        x = self.add([x, attn])
        return self.layernorm(x, training=training), attention_scores


class FeedForward(layers.Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=2 * units, activation="relu"),
                tf.keras.layers.Dense(units=units),
                tf.keras.layers.Dropout(rate=dropout_rate),
            ]
        )

        self.layernorm = tf.keras.layers.LayerNormalization()

    @tf.function(jit_compile=True)
    def call(self, x, training=False):
        x = x + self.seq(x, training=training)
        return self.layernorm(x, training=training)


class EncoderLayer(layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1, name="Encoder"):
        super().__init__(name=name)

        self.self_attention = SelfAttention(
            causal=False, num_heads=num_heads, key_dim=units, dropout=dropout_rate
        )
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        in_seq = inputs

        out_seq, _ = self.self_attention(in_seq, training=training)

        out_seq = self.ff(out_seq, training=training)

        return out_seq


class DecoderLayer(layers.Layer):
    def __init__(self, units, causal, num_heads=1, dropout_rate=0.1, name="Decoder"):
        super().__init__(name=name)

        self.self_attention = SelfAttention(
            causal=causal, num_heads=num_heads, key_dim=units, dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=units, dropout=dropout_rate
        )
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        in_seq, out_seq = inputs

        out_seq, _ = self.self_attention(out_seq, training=training)

        out_seq, attn_scores = self.cross_attention(out_seq, in_seq, training=training)

        out_seq = self.ff(out_seq, training=training)

        return out_seq, attn_scores


class PolicyModel(keras.Model):
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

        self._b2b_combo_dense = keras.Sequential(
            [
                layers.Dense(depth // 2, activation="relu"),
                layers.Dense(depth, activation="relu"),
            ],
            name="b2b_combo_dense",
        )

        self.key_embedding = layers.Embedding(
            input_dim=key_dim,
            output_dim=depth,
        )

        self.key_pos_encoding = PosEncoding(
            depth=depth,
            max_length=9,
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
            name="trunk",
        )

        self.top = layers.Dense(output_dim)

    @tf.function(jit_compile=True)
    def process_obs(self, inputs, training=False):
        board, piece, b2b_combo = inputs

        piece_scores = []
        patches = self.make_patches(board, training=training)
        board_dec = self.patch_pos_encoding(patches)

        piece_embedding = self.piece_embedding(piece, training=training)
        piece_dec = self.piece_pos_encoding(piece_embedding)

        b2b_combo_embedding = self._b2b_combo_dense(b2b_combo, training=training)
        piece_dec += b2b_combo_embedding[:, None, :]

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
        board, piece, b2b_combo, keys = inputs

        piece_dec, piece_scores = self.process_obs(
            (board, piece, b2b_combo), training=training
        )

        output, key_scores = self.process_keys((piece_dec, keys), training=training)

        if return_scores:
            return output, piece_scores, key_scores
        else:
            return output

    @tf.function(jit_compile=True)
    def _generate_next_key(
        self, ind, piece_dec, key_sequence, log_probs, masks, greedy=False
    ):
        def generate_mask(ind, stacked_key_sequence):
            matching_sequence = tf.reduce_all(
                stacked_key_sequence[:, None, :ind]
                == TFConvert.to_sequence[None, :, :ind],
                axis=-1,
            )[..., None]
            next_keys = tf.tile(
                TFConvert.to_sequence[None, :, ind], (self._batch_size, 1)
            )[..., None]
            possible_keys = tf.range(self._key_dim, dtype=tf.int64)[None, None, ...]
            valid = tf.reduce_any(
                tf.logical_and(matching_sequence, next_keys == possible_keys), axis=1
            )
            return valid

        stacked_key_sequence = tf.transpose(
            key_sequence.stack(), perm=[1, 0]
        )  # len, batch -> batch, len
        mask = generate_mask(ind, stacked_key_sequence)
        logits, _ = self.process_keys((piece_dec, stacked_key_sequence), training=False)

        masked_logits = tf.where(
            mask, logits[:, ind - 1, :], tf.constant(-1e9, dtype=tf.float32)
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
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=None, dtype=tf.bool),
        ],
    )
    def predict(self, inputs, greedy=False):
        piece_dec, piece_scores = self.process_obs(inputs, training=False)

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
                i, piece_dec, ks, lp, m, greedy
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


class ValueModel(keras.Model):
    def __init__(
        self, piece_dim, depth, num_heads, num_layers, dropout_rate, output_dim
    ):
        super().__init__()

        self._depth = depth

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

        self._b2b_combo_dense = keras.Sequential(
            [
                layers.Dense(depth // 2, activation="relu"),
                layers.Dense(depth, activation="relu"),
            ],
            name="b2b_combo_dense",
        )

        self.trunk = keras.Sequential(
            [
                layers.Flatten(),
                layers.Dropout(dropout_rate),
                layers.Dense(depth, activation="relu"),
                layers.Dense(depth // 2, activation="relu"),
            ],
            name="trunk",
        )

        self.top = layers.Dense(output_dim)

    @tf.function(jit_compile=True)
    def process_obs(self, inputs, training=False):
        board, piece, b2b_combo = inputs

        piece_scores = []
        patches = self.make_patches(board, training=training)
        board_dec = self.patch_pos_encoding(patches)

        piece_embedding = self.piece_embedding(piece, training=training)
        piece_dec = self.piece_pos_encoding(piece_embedding)

        b2b_combo_embedding = self._b2b_combo_dense(b2b_combo, training=training)
        piece_dec += b2b_combo_embedding[:, None, :]

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
    def call(self, inputs, training=False, return_scores=False):
        board, piece, b2b_combo = inputs

        piece_dec, piece_scores = self.process_obs(
            (board, piece, b2b_combo), training=training
        )

        trunk_out = self.trunk(piece_dec, training=training)

        output = self.top(trunk_out, training=training)

        if return_scores:
            return output, piece_scores
        else:
            return output

    @tf.function(jit_compile=True)
    def predict(self, inputs):
        board, piece, b2b_combo = inputs

        piece_dec, piece_scores = self.process_obs(
            (board, piece, b2b_combo), training=False
        )

        trunk_out = self.trunk(piece_dec, training=False)

        output = self.top(trunk_out, training=False)

        return output
