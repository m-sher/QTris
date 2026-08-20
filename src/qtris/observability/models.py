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
    """1v1 opponent-pool AlphaZero trainer hyperparams.

    TD(lambda) value target built from the outcome z in {-1,0,+1}; w_death=1, gamma=1,
    return_scale=1. The learner duels frozen snapshots sampled from a disk pool; both
    players' trajectories train the value head, the learner's also the policy."""

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
    mini_batch_size: int
    num_epochs: int
    value_coef: float
    learning_rate: float
    replay_capacity: int
    max_pool_size: int = 30
    pool_interval: int = 10
    pool_wr_gate: float = 0.55
    eval_interval: int = 10
    eval_games: int = 32
    td_lambda: float = 0.9
    td_blend: float = 0.0
    blend_horizon: int = 16
    resumed: bool = False
    checkpoint_dir: str = "checkpoints/1v1_placement_az"
    run_name: Optional[str] = None
    seed: Optional[int] = None
    save_states: Optional[str] = None
    # Opponent-pool rating (WHR batch refit)
    elo_enabled: bool = True
    elo_init: float = 1500.0
    whr_drift: float = 8.0
    whr_tie_sigma: float = 70.0


class OneVsOneAZLog(LogPayloadModel):
    """1v1 opponent-pool AlphaZero per-generation metrics."""

    # Optimization
    policy_loss: float
    value_loss: float
    entropy: float
    policy_kl: float
    update_kl: float
    explained_var: float
    value_mean: float
    value_target_var: float  # spread of the value target itself; read EV against this
    grad_norm: float  # global grad norm before the optimizer's clipnorm

    # Outcomes / gameplay.
    avg_game_len: float
    win_rate: float  # learner's decisive WR vs the sampled pool opponent
    win_rate_vs_ref: float  # learner's decisive WR vs the frozen gen_0 reference
    draw_rate: float
    app: float  # both players' attack per placement, gross, garbage exchange on
    app_learner: float  # learner-only attack per learner placement
    value_calibration: float
    avg_b2b: float
    max_b2b: float
    avg_combo: float
    surge_rate: float  # share of learner positions SITTING at b2b>=4 (occupancy)

    # Learner b2b economics; None when the gen produced no qualifying event.
    b2b_at_death: Optional[float]  # learner b2b carried into its fatal placement
    b2b_at_cashout: Optional[
        float
    ]  # b2b entering a surge break (trivial clear at b2b>=4); floored at 4
    episode_max_b2b: Optional[float]  # mean per-episode peak b2b
    chain_run_len: Optional[
        float
    ]  # difficult clears in a row; ANY other placement flushes
    bank_run_len: Optional[
        float
    ]  # difficult clears per b2b streak, tolerating stacking between them (the hoard)
    post_break_combo: Optional[
        float
    ]  # peak combo over a surge break, including combo carried into it
    post_break_clears: Optional[float]  # clears chained AFTER a surge break

    # Event counts.
    n_difficult_clears: int
    n_chain_runs: int
    n_breaks: int
    n_cashouts: int
    n_deaths: int
    decisive_games: int

    # Search
    visit_perplexity: (
        float  # exp(H(visit pi)): effective candidates searched; 1 = tunnel vision
    )
    top1_visit_share: float
    visit_coverage: float  # fraction of legal candidates that got >=1 visit
    root_cands_visited: Optional[float]  # mean root candidates receiving visit mass

    # Training progress
    updates: int
    buffer_size: int
    completed_games: int
    pool_size: int

    # Opponent-pool Elo: pre-formatted "elo/..." tags spliced in by to_payload.
    elo: dict[str, float] = {}

    # corr/Brier of the root value against the realized outcome, per steps-to-end bucket.
    grounding: dict[str, float | None] = {}

    # Mean Ahat and its two channels over the generation's rows; None when td_blend is 0.
    ahat_mean: Optional[float] = None
    ahat_b2b: Optional[float] = None
    ahat_atk: Optional[float] = None

    # Visualization (wrapped at log time)
    board: np.ndarray

    def to_payload(self) -> dict[str, Any]:
        d = super().to_payload()
        d.update(d.pop("elo", {}))
        d.update({f"grounding/{k}": v for k, v in d.pop("grounding", {}).items()})
        return d

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
            "value_target_var",
            "grad_norm",
        ),
        "outcomes": (
            "avg_game_len",
            "win_rate",
            "win_rate_vs_ref",
            "draw_rate",
            "app",
            "app_learner",
            "value_calibration",
        ),
        "gameplay": (
            "avg_b2b",
            "max_b2b",
            "avg_combo",
            "surge_rate",
            "b2b_at_death",
            "b2b_at_cashout",
            "episode_max_b2b",
            "chain_run_len",
            "bank_run_len",
            "post_break_combo",
            "post_break_clears",
        ),
        "counts": (
            "n_difficult_clears",
            "n_chain_runs",
            "n_breaks",
            "n_cashouts",
            "n_deaths",
            "decisive_games",
        ),
        "search": (
            "visit_perplexity",
            "top1_visit_share",
            "visit_coverage",
            "root_cands_visited",
        ),
        "blend": ("ahat_mean", "ahat_b2b", "ahat_atk"),
        "progress": ("updates", "buffer_size", "completed_games", "pool_size"),
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
