"""Canonical hyperparameter defaults for QTris.

Each model carries defaults matching the AR trainer (the "primary" variant).
Flat / 1v1 overrides via construction:
    PPOConfig(num_collection_steps=256, target_kl=0.02, expert_coef=0.1)
"""

from pydantic import BaseModel


class ModelConfig(BaseModel):
    piece_dim: int = 8
    key_dim: int = 12
    depth: int = 64
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.0
    max_len: int = 15
    queue_size: int = 5
    num_row_tiers: int = 2
    num_sequences: int = 320


class EnvConfig(BaseModel):
    max_holes: int = 50
    max_steps: int = 512
    garbage_chance: float = 0.15
    garbage_min: int = 1
    garbage_max: int = 4
    garbage_push_delay: int = 1
    pathfinding: bool = True


class PPOConfig(BaseModel):
    gamma: float = 0.99
    lam: float = 0.95
    ppo_clip: float = 0.2
    value_clip: float = 0.5
    entropy_coef: float = 0.01
    temperature: float = 1.0
    mini_batch_size: int = 512
    num_epochs: int = 4
    num_envs: int = 64
    num_collection_steps: int = 64
    early_stopping: bool = True
    target_kl: float = 0.03
    expert_coef: float = 0.005


class PretrainConfig(BaseModel):
    return_clip_low: float = -150.0
    return_clip_high: float = 100.0
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 3e-4


class DataGenConfig(BaseModel):
    search_depth: int = 16
    beam_width: int = 200
    death_trim_count: int = 20
    num_steps: int = 200_000
    seed: int = 0
