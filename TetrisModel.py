import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
        
        self.key_decoder_layers = [DecoderLayer(units=depth, causal=True, num_heads=num_heads, dropout_rate=0.1, name=f'kdec_{i}')
                                   for i in range(num_layers)]

        self.val_encoder_layers = [EncoderLayer(units=depth, num_heads=num_heads, dropout_rate=0.1, name=f'venc_{i}')
                                   for i in range(num_layers)]
        
        self.actor_top = layers.Dense(key_dim, name='actor_top')
        self.critic_top = keras.Sequential([layers.Flatten(),
                                            layers.Dense(1)], name='critic_top')
    
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
        val_enc = inputs

        val_scores = []
        for val_layer in self.val_encoder_layers:
            val_enc, last_attn = val_layer(val_enc, training=training)
            val_scores.append(last_attn)

        values = self.critic_top(val_enc, training=training)
        
        return values, val_scores
    
    @tf.function
    def call(self, inputs, training=False, return_scores=False):
        board, piece, inp_seq = inputs
        
        piece_dec, piece_scores = self.process_board((board, piece), training=training)
        
        logits, key_scores = self.process_keys((piece_dec, inp_seq), training=training)
        
        values, val_scores = self.process_vals(piece_dec, training=training)

        if return_scores:
            return logits, values, piece_scores, key_scores, val_scores
        else:
            return logits, values