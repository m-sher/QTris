"""Canonical hyperparameter defaults for QTris.

Overrides go in at construction:
    PretrainConfig(value_anchor_q=0.9, value_anchor_t=0.7)
"""

from pydantic import BaseModel


class ModelConfig(BaseModel):
    piece_dim: int = 8
    depth: int = 64
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.0
    max_len: int = 15
    queue_size: int = 5
    num_row_tiers: int = 2


class EnvConfig(BaseModel):
    max_holes: int = 50
    garbage_chance: float = 0.15
    garbage_min: int = 1
    garbage_max: int = 4
    garbage_push_delay: int = 1


class PretrainConfig(BaseModel):
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 3e-4
    # Placement value label tanh((v - median) / scale): scale puts the value_anchor_q
    # quantile of the oracle score at value_anchor_t.
    value_anchor_q: float = 0.95
    value_anchor_t: float = 0.8


class DataGenConfig(BaseModel):
    search_depth: int = 10
    beam_width: int = 256
    death_trim_count: int = 20
    num_steps: int = 200_000
    seed: int = 0
