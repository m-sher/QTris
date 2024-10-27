import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def positional_encoding(max_length, depth):
    half_depth = depth / 2
    positions = tf.range(max_length, dtype=tf.float32)[..., None]
    depths = tf.range(half_depth, dtype=tf.float32)[None, ...] / half_depth

    angle_rates = 1.0 / (10000 ** depths)
    angle_rads = positions * angle_rates

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
    return pos_encoding

class SeqEmbedding(layers.Layer):
    def __init__(self, in_dim, depth, max_length, mask_zero):
        super().__init__()
        
        self.pos_embedding = positional_encoding(max_length, depth)
        
        self.seq_emb = tf.keras.layers.Embedding(input_dim=in_dim,
                                                 output_dim=depth,
                                                 mask_zero=mask_zero)
        
        self.add = tf.keras.layers.Add()

    @tf.function
    def call(self, seq):
        seq = self.seq_emb(seq) # (batch, seq, key_emb_dim)

        x = tf.repeat(self.pos_embedding[None, :tf.shape(seq)[1]], repeats=tf.shape(seq)[0], axis=0)
    
        return self.add([seq, x])

class SelfAttention(layers.Layer):
    def __init__(self, causal, **kwargs):
        super().__init__()
        self.causal = causal
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add() 
        self.layernorm = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, x, training=False):
        attn, attention_scores = self.mha(query=x, value=x,
                                          use_causal_mask=self.causal,
                                          return_attention_scores=True)
        x = self.add([x, attn])
        return self.layernorm(x, training=training), attention_scores

class CrossAttention(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add() 
        self.layernorm = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, x, y, training=False, **kwargs):
        attn, attention_scores = self.mha(
                 query=x, value=y,
                 return_attention_scores=True)
        
        x = self.add([x, attn])
        return self.layernorm(x, training=training), attention_scores

class FeedForward(layers.Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(units=2*units, activation='relu'),
            tf.keras.layers.Dense(units=units),
            tf.keras.layers.Dropout(rate=dropout_rate),
        ])
        
        self.layernorm = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, x, training=False):
        x = x + self.seq(x, training=training)
        return self.layernorm(x, training=training)

class DecoderLayer(layers.Layer):
    def __init__(self, units, causal, num_heads=1, dropout_rate=0.1, name='Decoder'):
        super().__init__(name=name)
        
        self.self_attention = SelfAttention(causal=causal,
                                            num_heads=num_heads,
                                            key_dim=units,
                                            dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads,
                                              key_dim=units,
                                              dropout=dropout_rate)
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)
    
    @tf.function
    def call(self, inputs, training=False):
        in_seq, out_seq = inputs
        
        out_seq, _ = self.self_attention(out_seq, training=training)
        
        out_seq, attn_scores = self.cross_attention(out_seq, in_seq, training=training)
        
        out_seq = self.ff(out_seq, training=training)
        
        return out_seq, attn_scores

# class EncoderLayer(layers.Layer):
#     def __init__(self, units, num_heads=1, dropout_rate=0.1, name='Encoder'):
#         super().__init__(name=name)

#         self.self_attention = SelfAttention(causal=False,
#                                             num_heads=num_heads,
#                                             key_dim=units,
#                                             dropout=dropout_rate)

#         self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

#     @tf.function
#     def call(self, inputs, training=False):
#         in_seq = inputs

#         in_seq, attn_scores = self.self_attention(in_seq, training=training)
        
#         in_seq = self.ff(in_seq, training)

#         return in_seq, attn_scores

class TetrisModel(keras.Model):
    def __init__(self, piece_dim, key_dim, depth, num_heads, num_layers, max_length):
        super().__init__()

        self.depth = depth
        
        self.feature_extraction = keras.Sequential([
            layers.Conv2D(filters=depth, kernel_size=3, strides=1, activation='relu', padding='same'),
            layers.Conv2D(filters=depth, kernel_size=3, strides=1, activation='relu', padding='same'),
            layers.Conv2D(filters=depth, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Reshape((-1, depth))
        ])

        self.piece_embedding = SeqEmbedding(
            in_dim=piece_dim,
            depth=depth,
            max_length=7,
            mask_zero=False
        )
        
        self.key_embedding = SeqEmbedding(
            in_dim=key_dim,
            depth=depth,
            max_length=max_length,
            mask_zero=True
        )

        self.piece_decoder_layers = [DecoderLayer(units=depth, causal=False, num_heads=num_heads, dropout_rate=0.1, name=f'pdec_{i}')
                                     for i in range(num_layers)]
        
        self.key_decoder_layers = [DecoderLayer(units=depth, causal=True, num_heads=num_heads, dropout_rate=0.1, name=f'actor_dec_{i}')
                                   for i in range(num_layers)]

        self.val_decoder_layers = [DecoderLayer(units=depth, causal=True, num_heads=num_heads, dropout_rate=0.1, name=f'critic_dec_{i}')
                                   for i in range(num_layers)]
        
        self.actor_top = layers.Dense(key_dim, name='actor_top')
        self.critic_top = layers.Dense(1, name='critic_top')
    
    @tf.function
    def process_board(self, inputs, training=False):
        board, piece = inputs
        
        piece_scores = []
        board_features = self.feature_extraction(board)
        piece_dec = self.piece_embedding(piece)
        
        for piece_dec_layer in self.piece_decoder_layers:
            piece_dec, last_attn = piece_dec_layer([board_features, piece_dec], training=training)
            piece_scores.append(last_attn)
        
        return piece_dec, piece_scores

    @tf.function
    def process_keys(self, inputs, training=False):
        piece_dec, inp_seq = inputs

        key_scores = []
        key_dec = self.key_embedding(inp_seq)
        for dec_layer in self.key_decoder_layers:
            key_dec, last_attn = dec_layer([piece_dec, key_dec], training=training)
            key_scores.append(last_attn)

        logits = self.actor_top(key_dec, training=training)
        
        return logits, key_scores

    @tf.function
    def process_vals(self, inputs, training=False):
        piece_dec, inp_seq = inputs

        val_scores = []
        val_dec = self.key_embedding(inp_seq)
        for dec_layer in self.val_decoder_layers:
            val_dec, last_attn = dec_layer([piece_dec, val_dec], training=training)
            val_scores.append(last_attn)

        values = self.critic_top(val_dec, training=training)
        
        return values, val_scores
    
    @tf.function
    def call(self, inputs, training=False, return_scores=False):
        board, piece, inp_seq = inputs
        
        piece_dec, piece_scores = self.process_board((board, piece), training=training)
        
        logits, key_scores = self.process_keys((piece_dec, inp_seq), training=training)
        
        values, val_scores = self.process_vals((piece_dec, inp_seq), training=training)

        if return_scores:
            return logits, values, piece_scores, key_scores, val_scores
        else:
            return logits, values