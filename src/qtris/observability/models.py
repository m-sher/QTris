"""Pydantic schemas for run configs + per-step training metrics.

Each run config and each per-step log payload has a typed model here. Trainers
construct an instance per step; the observability backend serializes it and
writes the numpy fields named in `_image_fields` as images.

Hierarchy:
    PPOConfigBase           - 10 shared PPO knobs
      SingleAgentTrainConfig  - adds expert_coef + early_stopping (ar/flat)
      OneVsOneTrainConfig     - adds 1v1 self-play knobs

    PPOLogBase              - 20 shared per-step metrics (incl. board/scores images)
      SingleAgentPPOLog       - adds single-agent reward channels + expert co-train metrics
      OneVsOnePPOLog          - adds win/loss outcomes + derived APP metrics
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel


class PPOConfigBase(BaseModel):
    """Hyperparams logged as the run config; shared across all PPO trainers."""

    num_envs: int
    num_collection_steps: int
    mini_batch_size: int
    num_updates: int

    gamma: float
    lam: float
    ppo_clip: float
    value_clip: float
    entropy_coef: float
    target_kl: float


class SingleAgentTrainConfig(PPOConfigBase):
    """ar/flat trainers."""

    expert_coef: float
    early_stopping: bool = True


class OneVsOneTrainConfig(PPOConfigBase):
    """1v1 self-play trainer."""

    flat: bool
    pool_save_interval: int
    max_pool_size: int


class LogPayloadModel(BaseModel):
    """Base for any per-step log payload model. The backend writes the fields
    named in `_image_fields` as images and the numeric rest as scalars."""

    class Config:
        arbitrary_types_allowed = True

    _image_fields: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return self.dict()


class PPOLogBase(LogPayloadModel):
    """Metrics logged every generation; shared across all PPO trainers.

    `board` and `scores` are numpy arrays; the backend writes them as images
    at log time.
    """

    # PPO optimization
    ppo_loss: float
    entropy: float
    approx_kl: float
    clipped_frac: float
    value_loss: float
    explained_var: float
    return_var: float

    # Action distribution
    avg_probs: float

    # Reward channels (shared subset)
    avg_attacks: float
    avg_clears: float
    avg_attack_reward: float
    avg_total_reward: float
    avg_garbage_pushed: float
    avg_pieces: float

    # Gameplay metrics
    avg_b2b: float
    max_b2b: float
    avg_combo: float
    surge_rate: float

    # Training progress
    updates: int

    # Visualizations (wrapped at log time)
    board: np.ndarray
    scores: np.ndarray

    # Names of fields the backend should write as images instead of scalars.
    # Override in subclasses if they add more image fields.
    _image_fields: tuple[str, ...] = ("board", "scores")


class SingleAgentPPOLog(PPOLogBase):
    """ar/flat per-step metrics."""

    avg_reward: float
    avg_deaths: float

    expert_loss: float
    expert_accuracy: float
    expert_coef: float


class OneVsOnePPOLog(PPOLogBase):
    """1v1 per-step metrics."""

    avg_net_attacks: float
    avg_episodes: float

    # Derived gameplay (attacks/pieces/clears ratios)
    APP_reward: float
    APP_gross: float
    APP_net: float
    reward_per_clear: float
    att_per_clear: float
    cancel_rate: float

    # Outcome counts
    total_wins: int
    total_losses: int
    total_nondec: int
    win_rate: float
    decisive_wr: float
    wr_ema: float


class AlphaZeroTrainConfig(BaseModel):
    """Single-player AlphaZero (MCTS self-play) trainer hyperparams."""

    num_games: int
    horizon: int
    num_simulations: int
    c_puct: float
    gamma: float
    dirichlet_alpha: float
    dirichlet_eps: float
    temp_moves: int
    w_attack: float
    w_b2b: float
    w_death: float
    mini_batch_size: int
    num_epochs: int
    value_coef: float
    learning_rate: float
    replay_capacity: int
    gae_lambda: float


class SingleAgentAZLog(LogPayloadModel):
    """Single-player AlphaZero per-generation metrics."""

    # Optimization
    policy_loss: float
    value_loss: float
    entropy: float
    explained_var: float
    value_mean: float
    return_var: float

    # Reward / gameplay channels
    avg_total_reward: float
    avg_attacks: float
    avg_clears: float
    avg_deaths: float
    avg_pieces: float
    avg_b2b: float
    max_b2b: float
    avg_combo: float
    surge_rate: float

    # Search
    avg_visits: float
    dead_rate: float

    # Training progress
    updates: int
    buffer_size: int

    # Visualization (wrapped at log time)
    board: np.ndarray

    _image_fields: tuple[str, ...] = ("board",)
