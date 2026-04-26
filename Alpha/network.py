"""Alpha network: (P_320, V) over the canonical fixed action space.

Encoder mirrors QTris/Autoregressive/TetrisModel.py — board patches via a
conv stem, cross-attention between board patches ⊕ bcg-tokens and the piece
sequence ⊕ bcg-tokens.  Building blocks (PosEncoding, attention, decoder)
are copied here verbatim so this module is self-contained; the shapes and
hyperparameter conventions stay aligned with the existing models so weights
or training code can be ported back if needed.

Heads:
  * Policy: pooled state → MLP → 320 logits (NUM_ACTIONS in action_space.py).
    Mask + softmax happens at the consumer; the network always emits a
    fixed-shape vector.
  * Value: pooled state → MLP → tanh ∈ [-1, 1].
"""

from __future__ import annotations
import tensorflow as tf
import keras
from keras import layers


# ============================================================
# Building blocks (mirrors QTris/Autoregressive/TetrisModel.py)
# ============================================================

@tf.function(jit_compile=True)
def positional_encoding(max_length, depth):
    half_depth = depth / 2
    positions = tf.range(max_length, dtype=tf.float32)[..., None]
    depths = tf.range(half_depth, dtype=tf.float32)[None, ...] / half_depth
    angle_rates = 1.0 / (10000.0 ** depths)
    angle_rads = positions * angle_rates
    return tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)


class PosEncoding(layers.Layer):
    def __init__(self, depth, max_length, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.pos_encoding = positional_encoding(max_length, depth)

    @tf.function(jit_compile=True)
    def call(self, seq):
        seq_len = tf.shape(seq)[1]
        seq *= tf.cast(self.depth, tf.float32) ** 0.5
        seq += self.pos_encoding[None, :seq_len]
        return seq


class SelfAttention(layers.Layer):
    def __init__(self, causal, **kwargs):
        super().__init__()
        self.causal = causal
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.add = layers.Add()
        self.layernorm = layers.LayerNormalization()

    @tf.function(jit_compile=True)
    def call(self, x, training=False):
        attn = self.mha(query=x, value=x, use_causal_mask=self.causal)
        x = self.add([x, attn])
        return self.layernorm(x, training=training)


class CrossAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.add = layers.Add()
        self.layernorm = layers.LayerNormalization()

    @tf.function(jit_compile=True)
    def call(self, x, y, training=False):
        attn = self.mha(query=x, value=y)
        x = self.add([x, attn])
        return self.layernorm(x, training=training)


class FeedForward(layers.Layer):
    def __init__(self, units, dropout_rate=0.0):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(units=2 * units, activation="relu"),
            layers.Dense(units=units),
            layers.Dropout(rate=dropout_rate),
        ])
        self.layernorm = layers.LayerNormalization()

    @tf.function(jit_compile=True)
    def call(self, x, training=False):
        x = x + self.seq(x, training=training)
        return self.layernorm(x, training=training)


class DecoderLayer(layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.0, name="Decoder"):
        super().__init__(name=name)
        self.self_attention = SelfAttention(
            causal=False, num_heads=num_heads, key_dim=units, dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=units, dropout=dropout_rate
        )
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        in_seq, out_seq = inputs
        out_seq = self.self_attention(out_seq, training=training)
        out_seq = self.cross_attention(out_seq, in_seq, training=training)
        out_seq = self.ff(out_seq, training=training)
        return out_seq


# ============================================================
# Alpha (P, V) model
# ============================================================

# Constants — match TetrisEnv conventions.
PIECE_VOCAB = 8                      # 0=N + 7 piece types
NUM_ACTIONS_DEFAULT = 320            # canonical action-space dim (action_space.py)


class AlphaModel(keras.Model):
    """(P_320, V) network for canonical fixed-action-space MCTS.

    Inputs (tuple):
      board:  (B, 24, 10, 1) float32
      pieces: (B, T) int64        — [active, hold, q0, q1, q2, q3, q4]
      bcg:    (B, 3) float32      — [b2b, combo, garbage_total]

    Outputs:
      P_logits: (B, NUM_ACTIONS) float32  — pre-softmax; consumer applies the
                                            valid-action mask before softmax
      V:        (B,) float32             — tanh ∈ [-1, 1]
    """

    def __init__(
        self,
        depth: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout_rate: float = 0.0,
        num_actions: int = NUM_ACTIONS_DEFAULT,
    ):
        super().__init__()
        self._depth = depth
        self._num_actions = num_actions

        # --- Board patches ---
        self.make_patches = keras.Sequential([
            keras.Input(shape=(24, 10, 1)),
            layers.Rescaling(scale=2.0, offset=-1.0),
            layers.Conv2D(filters=depth // 2, kernel_size=3, strides=1,
                          padding="same", activation="relu"),
            layers.Conv2D(filters=depth, kernel_size=3, strides=1,
                          padding="same", activation="relu"),
            layers.Conv2D(filters=depth, kernel_size=2, strides=2,
                          padding="valid", activation="relu"),
            layers.Reshape((-1, depth)),
        ], name="make_patches")
        num_patches = self.make_patches.output_shape[1]
        self.patch_pos_encoding = PosEncoding(depth=depth, max_length=num_patches,
                                              name="patch_pos")

        # --- Piece embedding ---
        self.piece_embedding = layers.Embedding(input_dim=PIECE_VOCAB,
                                                output_dim=depth)
        self.piece_pos_encoding = PosEncoding(depth=depth, max_length=8,
                                              name="piece_pos")

        # --- B2B / Combo / Garbage tokens ---
        self._bcg_proj_b2b = layers.Dense(depth, name="bcg_proj_b2b")
        self._bcg_proj_combo = layers.Dense(depth, name="bcg_proj_combo")
        self._bcg_proj_garbage = layers.Dense(depth, name="bcg_proj_garbage")
        self._bcg_ln = layers.LayerNormalization(name="bcg_ln")

        # --- Cross-attention encoder ---
        self.board_decoder_layers = [
            DecoderLayer(units=depth, num_heads=num_heads,
                         dropout_rate=dropout_rate, name=f"board_dec_{i}")
            for i in range(num_layers)
        ]
        self.piece_decoder_layers = [
            DecoderLayer(units=depth, num_heads=num_heads,
                         dropout_rate=dropout_rate, name=f"piece_dec_{i}")
            for i in range(num_layers)
        ]

        # --- Pool to a single state vector ---
        self.state_pool = layers.GlobalAveragePooling1D(name="state_pool")

        # --- Policy head: pooled state → fixed 320 logits ---
        self.policy_head = keras.Sequential([
            layers.Dropout(dropout_rate),
            layers.Dense(depth, activation="relu"),
            layers.Dense(depth // 2, activation="relu"),
            layers.Dense(num_actions),
        ], name="policy_head")

        # --- Value head: pooled state → scalar ∈ [-1, 1] ---
        self.value_head = keras.Sequential([
            layers.Dropout(dropout_rate),
            layers.Dense(depth, activation="relu"),
            layers.Dense(depth // 2, activation="relu"),
            layers.Dense(1, activation="tanh"),
        ], name="value_head")

    def _tokenize_bcg(self, bcg, training=False):
        bcg_log = tf.math.log1p(bcg + 1.0)
        t_b2b = self._bcg_proj_b2b(bcg_log[:, 0:1], training=training)
        t_combo = self._bcg_proj_combo(bcg_log[:, 1:2], training=training)
        t_garbage = self._bcg_proj_garbage(bcg_log[:, 2:3], training=training)
        return self._bcg_ln(
            tf.stack([t_b2b, t_combo, t_garbage], axis=1), training=training
        )

    def _encode(self, board, pieces, bcg, training=False):
        patches = self.make_patches(board, training=training)
        board_dec = self.patch_pos_encoding(patches)

        piece_emb = self.piece_embedding(pieces, training=training)
        piece_dec = self.piece_pos_encoding(piece_emb)

        bcg_tokens = self._tokenize_bcg(bcg, training=training)
        piece_dec = tf.concat([piece_dec, bcg_tokens], axis=1)
        board_dec = tf.concat([board_dec, bcg_tokens], axis=1)

        for board_layer, piece_layer in zip(
            self.board_decoder_layers, self.piece_decoder_layers
        ):
            board_dec = board_layer([piece_dec, board_dec], training=training)
            piece_dec = piece_layer([board_dec, piece_dec], training=training)

        # Pool the piece stream — same convention as ValueModel.trunk_bcg input.
        state = self.state_pool(piece_dec)
        return piece_dec, state

    # Inputs are fixed-shape per the canonical action space (board=(_,24,10,1),
    # pieces=(_,7), bcg=(_,3)).  The leading batch dim is polymorphic via
    # `shape=(None, ...)` so a single trace covers single-state inference,
    # batched MCTS leaf eval, and training mini-batches.
    @tf.function(
        jit_compile=True,
        input_signature=[
            (
                tf.TensorSpec(shape=(None, 24, 10, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 7), dtype=tf.int64),
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        ],
        reduce_retracing=True,
    )
    def _forward(self, inputs, training):
        board, pieces, bcg = inputs
        _, state = self._encode(board, pieces, bcg, training=training)
        v = self.value_head(state, training=training)[..., 0]
        logits = self.policy_head(state, training=training)
        return logits, v

    def call(self, inputs, training=False):
        # Promote `training` to a tensor so the @tf.function signature matches.
        training_t = tf.constant(bool(training), dtype=tf.bool)
        return self._forward(inputs, training_t)
