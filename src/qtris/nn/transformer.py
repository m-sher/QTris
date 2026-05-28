import tensorflow as tf
from keras import layers

from qtris.nn.attention import CrossAttention, FeedForward, SelfAttention


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


class CrossAttentionLayer(layers.Layer):
    """Cross-attention + feed-forward block (no self-attention)."""

    def __init__(self, units, num_heads=1, dropout_rate=0.1, name="CrossAttnLayer"):
        super().__init__(name=name)

        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=units, dropout=dropout_rate
        )
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        in_seq, out_seq = inputs

        out_seq, attn_scores = self.cross_attention(out_seq, in_seq, training=training)

        out_seq = self.ff(out_seq, training=training)

        return out_seq, attn_scores


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
