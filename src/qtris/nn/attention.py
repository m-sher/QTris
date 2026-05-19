import tensorflow as tf
from keras import layers


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
