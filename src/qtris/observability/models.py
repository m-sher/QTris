"""Pydantic schemas for wandb run configs + per-step training metrics.

Each `wandb.init` config dict and each `wandb.log` payload has a typed model
here. Trainers construct an instance per step; the wandb_backend serializes
it (wrapping numpy image fields in `wandb.Image` at the boundary).

Hierarchy:
    PPOConfigBase           - 10 shared PPO knobs
      SingleAgentTrainConfig  - adds expert_coef + early_stopping (ar/flat)
      OneVsOneTrainConfig     - adds 1v1 self-play knobs

    PPOLogBase              - 20 shared per-step metrics (incl. board/scores images)
      SingleAgentPPOLog       - adds single-agent reward channels + expert co-train metrics
      OneVsOnePPOLog          - adds win/loss outcomes + derived APP metrics

Pydantic version: v1 (pinned because tf-agents 0.19.0 requires typing-extensions==4.5.0,
which is incompatible with pydantic v2). Translate to v2 syntax if tf-agents is bumped.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel


class PPOConfigBase(BaseModel):
    """Hyperparams logged to wandb.config; shared across all PPO trainers."""

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


class PPOLogBase(BaseModel):
    """Metrics logged every generation; shared across all PPO trainers.

    `board` and `scores` are numpy arrays; the wandb_backend wraps them in
    `wandb.Image` at log time so callers don't need to import wandb directly.
    """

    class Config:
        arbitrary_types_allowed = True

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

    # Names of fields the backend should wrap with wandb.Image before logging.
    # Override in subclasses if they add more image fields.
    _image_fields: tuple[str, ...] = ("board", "scores")

    def to_wandb_payload(self) -> dict[str, Any]:
        """Dict shaped for `wandb.log()`, with image fields wrapped."""
        import wandb

        payload = self.dict()  # pydantic v1 API; v2 would be self.model_dump()
        for key in self._image_fields:
            arr = payload.get(key)
            if arr is not None:
                payload[key] = wandb.Image(arr)
        return payload


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
