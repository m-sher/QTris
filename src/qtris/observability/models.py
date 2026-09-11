"""Pydantic schemas for run configs + per-step training metrics.

Each run config and each per-step log payload has a typed model here. Trainers
construct an instance per step; the observability backend serializes it and
writes the numpy fields named in `_image_fields` as images.

`LogPayloadModel` is the base of every per-step payload; the AZ configs are
flat models.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pydantic import BaseModel


class LogPayloadModel(BaseModel):
    """Base for any per-step log payload model. The backend writes the fields
    named in `_image_fields` as images and the numeric rest as scalars, tagged
    `group/field` per `_tag_groups` (how TensorBoard/wandb section the charts)."""

    class Config:
        arbitrary_types_allowed = True

    _image_fields: tuple[str, ...] = ()
    _tag_groups: dict[str, tuple[str, ...]] = {}

    def to_payload(self) -> dict[str, Any]:
        return self.dict()


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
    w_death: float
    mini_batch_size: int
    num_epochs: int
    value_coef: float
    learning_rate: float
    replay_capacity: int
    gae_lambda: float
    garbage_traces: Optional[str] = None
    trace_free_envs: int = 0
    return_scale: float = 0.0
    resumed: bool = False
    checkpoint_dir: str = "checkpoints/placement_az"
    run_name: Optional[str] = None
    harvest: bool = True
    trace_tiers: Optional[str] = None
    seed: Optional[int] = None
    curriculum: bool = False
    curriculum_start: float = 0.0


class OneVsOnePlacementAZConfig(BaseModel):
    """Productive-attack and own-death-risk 1v1 trainer configuration."""

    objective: str = "productive_attack_risk_v1"
    gamma: float = 0.97
    risk_horizon: int = 24
    risk_threshold: float = 0.10
    risk_margin: float = 0.05
    risk_coef: float = 1.0
    w_death: float = 0.0
    q_norm: bool = True
    leaves_per_round: int = 4
    vloss: float = 1.0
    init_checkpoint: Optional[str] = None

    num_games: int
    horizon: int
    max_game_steps: int
    num_simulations: int
    c_puct: float
    dirichlet_alpha: float
    dirichlet_eps: float
    temp_moves: int
    w_attack: float = 0.0
    w_b2b: float = 0.0
    w_height: float = 0.0
    w_bumpiness: float = 0.0
    fpu: float = 0.0
    w_holes: float = 0.0
    w_plain: float = 0.0
    mini_batch_size: int
    num_epochs: int
    value_coef: float
    learning_rate: float
    replay_capacity: int
    max_pool_size: int = 30
    pool_interval: int = 10
    pool_wr_gate: float = 0.55
    eval_interval: int = 20
    eval_games: int = 32
    n_step: int = 14
    resumed: bool = False
    checkpoint_dir: str = "checkpoints/1v1_attack_risk"
    run_name: Optional[str] = None
    seed: Optional[int] = None
    save_states: Optional[str] = None
    # Opponent-pool rating (WHR batch refit)
    elo_enabled: bool = True
    elo_init: float = 1500.0
    whr_drift: float = 8.0
    whr_tie_sigma: float = 70.0


class OneVsOneCollectionLog(LogPayloadModel):
    """Collection diagnostics for generations without an optimizer update."""

    diagnostics: dict[str, float | None]

    def to_payload(self) -> dict[str, Any]:
        return self.diagnostics


class OneVsOneAZLog(LogPayloadModel):
    """1v1 learning, survival, and productive-attack diagnostics."""

    policy_loss: float
    value_loss: float
    entropy: float
    update_kl: float
    explained_var: float
    grad_norm: float
    win_rate_vs_ref: float
    ref_decisive: int
    avg_b2b: float
    b2b_at_death: Optional[float]
    chain_run_len: Optional[float]
    n_deaths: int
    updates: int
    buffer_size: int
    completed_games: int
    diagnostics: dict[str, float | None] = {}

    def to_payload(self) -> dict[str, Any]:
        d = super().to_payload()
        d.update(d.pop("diagnostics", {}))
        return d

    _tag_groups: dict[str, tuple[str, ...]] = {
        "optimization": (
            "policy_loss",
            "value_loss",
            "entropy",
            "update_kl",
            "explained_var",
            "grad_norm",
        ),
        "outcomes": ("win_rate_vs_ref", "ref_decisive"),
        "gameplay": ("avg_b2b", "b2b_at_death", "chain_run_len"),
        "counts": ("n_deaths",),
        "progress": ("updates", "buffer_size", "completed_games"),
    }


class SingleAgentAZLog(LogPayloadModel):
    """Single-player AlphaZero per-generation metrics."""

    # Optimization
    policy_loss: float
    value_loss: float
    entropy: float
    policy_kl: float
    update_kl: float
    explained_var: float
    value_mean: float
    return_var: float
    return_scale: float

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

    # Incoming garbage (trace-replay realism; counters from the env)
    garbage_in_app: float
    garbage_in_rate: float
    garbage_in_chunk: float
    garbage_in_max: float
    garbage_cancel_frac: float

    # Search
    avg_visits: float
    dead_rate: float

    # Training progress
    updates: int
    buffer_size: int
    trace_pool_size: int
    curriculum_d: float

    # Visualization (wrapped at log time)
    board: np.ndarray

    _image_fields: tuple[str, ...] = ("board",)
    _tag_groups: dict[str, tuple[str, ...]] = {
        "optimization": (
            "policy_loss",
            "value_loss",
            "entropy",
            "policy_kl",
            "update_kl",
            "explained_var",
            "value_mean",
            "return_var",
            "return_scale",
        ),
        "rewards": (
            "avg_total_reward",
            "avg_attacks",
            "avg_clears",
            "avg_deaths",
            "avg_pieces",
        ),
        "gameplay": (
            "avg_b2b",
            "max_b2b",
            "avg_combo",
            "surge_rate",
            "garbage_in_app",
            "garbage_in_rate",
            "garbage_in_chunk",
            "garbage_in_max",
            "garbage_cancel_frac",
        ),
        "search": ("avg_visits", "dead_rate"),
        "progress": ("updates", "buffer_size", "trace_pool_size", "curriculum_d"),
    }
