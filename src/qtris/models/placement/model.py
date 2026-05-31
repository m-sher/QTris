import tensorflow as tf
import keras
from keras import layers
from tensorflow_probability import distributions
from qtris.nn.transformer import CrossAttentionLayer, DecoderLayer, PosEncoding
from qtris.models.encoders import make_patches
from qtris.models.base import QtrisModelBase
from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from TetrisEnv.Moves import Keys

HARD_DROP_ID = Keys.HARD_DROP


class PlacementPolicyModel(QtrisModelBase):
    """Candidate-ranking policy: scores up to 128 placement vectors, each
    conditioned on the shared board latent via cross-attention. Candidates are
    independent (no self-attention across the set)."""

    def __init__(
        self,
        batch_size,
        piece_dim,
        depth,
        num_heads,
        num_layers,
        dropout_rate,
        candidate_capacity=CANDIDATE_CAPACITY,
    ):
        super().__init__()

        self._batch_size = batch_size
        self._depth = depth
        self._candidate_capacity = candidate_capacity

        # Shared board/piece/bcg encoder (process_obs reads these).
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
        self.piece_embedding = layers.Embedding(input_dim=piece_dim, output_dim=depth)
        self.piece_pos_encoding = PosEncoding(depth=depth, max_length=7)
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
        self._bcg_proj_b2b = layers.Dense(depth, activation=None, name="bcg_proj_b2b")
        self._bcg_proj_combo = layers.Dense(
            depth, activation=None, name="bcg_proj_combo"
        )
        self._bcg_proj_garbage = layers.Dense(
            depth, activation=None, name="bcg_proj_garbage"
        )
        self._bcg_ln = layers.LayerNormalization(name="bcg_ln")

        # Candidate head: embed each placement vector, cross-attend to the board.
        self.move_encoder = keras.Sequential(
            [
                layers.Dense(depth, activation="relu"),
                layers.LayerNormalization(),
            ],
            name="move_encoder",
        )
        self.cand_decoder_layers = [
            CrossAttentionLayer(
                units=depth,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                name=f"cand_dec_{i}",
            )
            for i in range(num_layers)
        ]
        self.score_trunk = keras.Sequential(
            [
                layers.Dropout(dropout_rate),
                layers.Dense(depth, activation="relu"),
            ],
            name="score_trunk",
        )
        self.score_top = layers.Dense(1, name="cand_logit")

    @tf.function(jit_compile=True)
    def score_candidates(self, piece_dec, cand_placements, cand_mask, training=False):
        move_emb = self.move_encoder(cand_placements, training=training)  # (B,C,depth)
        cand_dec = move_emb
        for layer in self.cand_decoder_layers:
            cand_dec, _ = layer([piece_dec, cand_dec], training=training)
        logits = tf.squeeze(
            self.score_top(
                self.score_trunk(cand_dec, training=training), training=training
            ),
            axis=-1,
        )  # (B,C)
        return tf.where(cand_mask, logits, tf.constant(-1e9, dtype=tf.float32))

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, return_scores=False):
        board, piece, b2b_combo_garbage, cand_placements, cand_mask = inputs

        piece_dec, piece_scores = self.process_obs(
            (board, piece, b2b_combo_garbage), training=training
        )
        logits = self.score_candidates(
            piece_dec, cand_placements, cand_mask, training=training
        )

        if return_scores:
            return logits, piece_scores
        return logits

    @tf.function(
        jit_compile=True,
        input_signature=[
            (
                tf.TensorSpec(shape=(None, 24, 10, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 7), dtype=tf.int64),
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                tf.TensorSpec(
                    shape=(None, None, PLACEMENT_FEATURE_DIM), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(None, None), dtype=tf.bool),
            ),
            tf.TensorSpec(shape=None, dtype=tf.bool),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.float32),
        ],
    )
    def predict(self, inputs, greedy=False, cand_sequences=None, temperature=1.0):
        board, piece, b2b_combo_garbage, cand_placements, cand_mask = inputs

        piece_dec, piece_scores = self.process_obs(
            (board, piece, b2b_combo_garbage), training=False
        )
        logits = self.score_candidates(
            piece_dec, cand_placements, cand_mask, training=False
        )

        masked_logits = tf.where(
            cand_mask, logits / temperature, tf.constant(-1e9, tf.float32)
        )
        dist = distributions.Categorical(logits=masked_logits, dtype=tf.int64)

        if greedy:
            action_index = tf.argmax(masked_logits, axis=-1, output_type=tf.int64)
        else:
            action_index = dist.sample()

        log_prob = dist.log_prob(action_index)

        batch_indices = tf.range(tf.shape(cand_sequences)[0], dtype=tf.int64)
        selected_sequence = tf.gather_nd(
            cand_sequences,
            tf.stack([batch_indices, action_index], axis=1),
        )

        return selected_sequence, log_prob, action_index, piece_scores
