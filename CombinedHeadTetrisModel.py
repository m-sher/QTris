import tensorflow as tf
from tensorflow import keras
from keras import layers

def positional_encoding(max_length, depth):
    half_depth = depth / 2
    positions = tf.range(max_length, dtype=tf.float32)[..., None]
    depths = tf.range(half_depth, dtype=tf.float32)[None, ...] / half_depth

    angle_rates = 1.0 / (10000 ** depths)
    angle_rads = positions * angle_rates

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
    return pos_encoding

class PosEncoding(layers.Layer):
    def __init__(self, depth, max_length):
        super().__init__()
        self.depth = depth
        
        self.pos_encoding = positional_encoding(max_length, depth)
        
        self.add = tf.keras.layers.Add()

    @tf.function
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
    def __init__(self, piece_dim, depth, num_heads, num_layers, dropout_rate, trunk_dim, num_actions):
        super().__init__()

        self._depth = depth
        self._trunk_dim = trunk_dim
        self._num_actions = num_actions
        
        self.make_patches = keras.Sequential([
            keras.Input(shape=(24, 10, 1)),
            layers.Rescaling(scale=2.0, offset=-1.0),
            layers.Conv2D(filters=depth, kernel_size=2, strides=2, padding='valid'),
            layers.Reshape((-1, depth))
        ])
        
        num_patches = self.make_patches.output_shape[1]
        self.patch_pos_encoding = PosEncoding(
            depth=depth,
            max_length=num_patches
        )
        
        self.board_encoder_layers = [EncoderLayer(units=depth, num_heads=num_heads, dropout_rate=dropout_rate, name=f'board_enc_{i}')
                                     for i in range(num_layers)]
        
        self.piece_embedding = layers.Embedding(
            input_dim=piece_dim,
            output_dim=depth,
        )
        
        self.piece_pos_encoding = PosEncoding(
            depth=depth,
            max_length=7,
        )
        
        self.piece_decoder_layers = [DecoderLayer(units=depth, causal=False, num_heads=num_heads, dropout_rate=dropout_rate, name=f'piece_dec_{i}')
                                     for i in range(num_layers)]
        
        self.flatten_rep = layers.Flatten()

        self.trunk = layers.Dense(trunk_dim, activation='relu')

        self.policy_head = layers.Dense(num_actions)
        self.value_head = layers.Dense(1)

    def process_obs(self, inputs, training=False):
        
        board, piece = inputs
        
        piece_scores = []
        patches = self.make_patches(board, training=training)
        board_enc = self.patch_pos_encoding(patches)
        
        for board_enc_layer in self.board_encoder_layers:
            board_enc = board_enc_layer(board_enc, training=training)
        
        piece_embedding = self.piece_embedding(piece, training=training)
        piece_dec = self.piece_pos_encoding(piece_embedding)
        
        for piece_dec_layer in self.piece_decoder_layers:
            piece_dec, last_attn = piece_dec_layer([board_enc, piece_dec], training=training)
            piece_scores.append(last_attn)
        
        latent_state_rep = self.flatten_rep(piece_dec)
        
        return latent_state_rep, piece_scores
    
    @tf.function
    def call(self, inputs, training=False, return_scores=False):
        
        board, piece = inputs
        
        latent_state_rep, piece_scores = self.process_obs((board, piece), training=training)

        trunk_out = self.trunk(latent_state_rep, training=training)

        policy_out = self.policy_head(trunk_out, training=training)
        value_out = tf.squeeze(self.value_head(trunk_out, training=training), axis=-1)

        if return_scores:
            return policy_out, value_out, piece_scores
        else:
            return policy_out, value_out