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

class EncoderLayer(layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1, name='Encoder'):
        super().__init__(name=name)
        
        self.self_attention = SelfAttention(causal=False,
                                            num_heads=num_heads,
                                            key_dim=units,
                                            dropout=dropout_rate)
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    @tf.function
    def call(self, inputs, training=False):
        in_seq = inputs
        
        out_seq, _ = self.self_attention(in_seq, training=training)
        
        out_seq = self.ff(out_seq, training=training)
        
        return out_seq
    
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

class TetrisModel(keras.Model):
    def __init__(self, piece_dim, key_dim, depth, num_heads, num_layers, max_length, out_dim):
        super().__init__()

        self.depth = depth
        
        self.make_patches = keras.Sequential([
            keras.Input(shape=(28, 10, 1)),
            layers.Rescaling(scale=2.0, offset=-1.0),
            layers.Conv2D(filters=depth, kernel_size=2, strides=2, padding='valid'),
            layers.Reshape((-1, depth))
        ])
        
        num_patches = self.make_patches.output_shape[1]
        self.patch_embedding = layers.Embedding(input_dim=num_patches,
                                                output_dim=self.depth)(tf.range(num_patches)[None, ...])
        
        self.board_encoder_layers = [EncoderLayer(units=depth, num_heads=num_heads, dropout_rate=0.1, name=f'board_enc_{i}')
                                     for i in range(num_layers)]
        
        self.piece_embedding = SeqEmbedding(
            in_dim=piece_dim,
            depth=depth,
            max_length=7,
            mask_zero=False
        )
        
        self.piece_decoder_layers = [DecoderLayer(units=depth, causal=False, num_heads=num_heads, dropout_rate=0.1, name=f'piece_dec_{i}')
                                     for i in range(num_layers)]
        
        self.key_embedding = SeqEmbedding(
            in_dim=key_dim,
            depth=depth,
            max_length=max_length,
            mask_zero=True
        )
        self.key_decoder_layers = [DecoderLayer(units=depth, causal=True, num_heads=num_heads, dropout_rate=0.1, name=f'key_dec_{i}')
                                   for i in range(num_layers)]
        self.model_top = layers.Dense(out_dim, name='model_top')
    
    @tf.function
    def process_obs(self, inputs, training=False):
        board, piece = inputs
        
        piece_scores = []
        board_enc = self.make_patches(board) + self.patch_embedding
        
        for board_enc_layer in self.board_encoder_layers:
            board_enc = board_enc_layer(board_enc, training=training)
        
        piece_dec = self.piece_embedding(piece)
        
        for piece_dec_layer in self.piece_decoder_layers:
            piece_dec, last_attn = piece_dec_layer([board_enc, piece_dec], training=training)
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
        
        key_out = self.model_top(key_dec)
        
        return key_out, key_scores
    
    @tf.function
    def call(self, inputs, training=False, return_scores=False):
        
        board, piece, inp_seq = inputs
        
        piece_dec, piece_scores = self.process_obs((board, piece), training=training)
    
        model_out, key_scores = self.process_keys((piece_dec, inp_seq), training=training)

        if return_scores:
            return model_out, piece_scores, key_scores
        else:
            return model_out