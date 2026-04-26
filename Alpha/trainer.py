"""Phase 2 driver: oracle-scaffolded self-play + supervised training.

A *generation* is one cycle of:
  1. Anneal the BlendedValuator's oracle weight.
  2. Run N self-play games; transitions go into the replay buffer.
  3. Run K training steps (mini-batches sampled from the buffer).

The loss matches AlphaZero standard, computed in the canonical 320-dim
action space:
  L_P = -Σ π · log softmax_masked(logits)   (mask invalid slots to -∞)
  L_V = (V_θ - z)²
  L   = L_P + L_V                          (+ AdamW weight decay)

All inputs and targets are fixed-shape, so no padding is needed and the
tf.function compiles once.
"""

from __future__ import annotations
import dataclasses
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
import keras

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover — wandb optional
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False

from .replay_buffer import ReplayBuffer, Transition
from .network import AlphaModel
from .neural_valuator import NeuralValuator
from .blended_valuator import BlendedValuator
from .valuator import DecomposeOracle
from .mcts import MCTS
from .self_play import play_game


# ============================================================
# Trainer
# ============================================================

@dataclass
class TrainMetrics:
    loss: float
    policy_loss: float
    value_loss: float


def _stack_batch(transitions: List[Transition]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Stack a list of Transitions into batched arrays.  All inputs are
    fixed-shape — no padding required."""
    boards = np.stack([t.board for t in transitions], axis=0).astype(np.float32)
    pieces = np.stack([t.pieces for t in transitions], axis=0).astype(np.int64)
    bcgs = np.stack([t.bcg for t in transitions], axis=0).astype(np.float32)
    pis = np.stack([t.pi for t in transitions], axis=0).astype(np.float32)
    masks = np.stack([t.valid_mask for t in transitions], axis=0).astype(np.bool_)
    z = np.array([t.z for t in transitions], dtype=np.float32)
    return boards, pieces, bcgs, pis, masks, z


class Trainer:
    def __init__(
        self,
        model: AlphaModel,
        optimizer: keras.optimizers.Optimizer,
        value_loss_weight: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.value_loss_weight = value_loss_weight

    @tf.function
    def _train_step(self, board, pieces, bcg, pi, mask, z):
        with tf.GradientTape() as tape:
            logits, v = self.model((board, pieces, bcg), training=True)
            # Mask invalid slots before softmax so gradients don't try to
            # push down logits the network can't act on anyway.
            mask_f = tf.cast(mask, logits.dtype)
            masked_logits = tf.where(
                mask, logits, tf.constant(-1e9, dtype=logits.dtype)
            )
            log_softmax = tf.nn.log_softmax(masked_logits, axis=-1)
            term = tf.where(pi > 0, pi * log_softmax, tf.zeros_like(pi))
            policy_loss = -tf.reduce_sum(term, axis=-1)
            policy_loss = tf.reduce_mean(policy_loss)

            value_loss = tf.reduce_mean(tf.square(v - z))
            loss = policy_loss + self.value_loss_weight * value_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        return loss, policy_loss, value_loss

    def train_step(
        self, buffer: ReplayBuffer, batch_size: int
    ) -> Optional[TrainMetrics]:
        batch = buffer.sample(batch_size)
        if not batch:
            return None
        board, pieces, bcg, pi, mask, z = _stack_batch(batch)
        loss, p_loss, v_loss = self._train_step(
            tf.constant(board), tf.constant(pieces), tf.constant(bcg),
            tf.constant(pi), tf.constant(mask), tf.constant(z),
        )
        return TrainMetrics(
            loss=float(loss.numpy()),
            policy_loss=float(p_loss.numpy()),
            value_loss=float(v_loss.numpy()),
        )


# ============================================================
# Phase 2 driver
# ============================================================

@dataclass
class Phase2Config:
    num_generations: int = 50
    games_per_generation: int = 16
    train_steps_per_generation: int = 200
    batch_size: int = 256
    num_simulations: int = 100
    num_steps_per_game: int = 200
    buffer_capacity: int = 200_000
    # Oracle-weight schedule: linear anneal from 1.0 → 0.0 over this many gens.
    anneal_generations: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    seed: int = 0
    # --- Network architecture (matches QTris/Autoregressive/TetrisModel defaults) ---
    model_depth: int = 96
    model_num_heads: int = 4
    model_num_layers: int = 4
    model_dropout: float = 0.0
    # --- Solo random-garbage pressure (Block A) ---
    # Probability of injecting a garbage tier after each placement.  Range +
    # delay match the b2b_run_eval_games scheme used elsewhere in the env.
    garbage_chance: float = 0.15
    garbage_min: int = 1
    garbage_max: int = 4
    # --- WandB logging (optional; disabled when use_wandb=False) ---
    use_wandb: bool = True
    wandb_project: str = "Tetris-Alpha"
    wandb_run_name: Optional[str] = None
    wandb_tags: Tuple[str, ...] = ("phase2", "solo")
    # --- Warm-start from an Autoregressive PolicyModel checkpoint ---
    # Path to the directory containing a tf.train.CheckpointManager checkpoint
    # for QTris/Autoregressive/TetrisModel.PolicyModel.  Only encoder weights
    # are transferred; policy/value heads start fresh.  Set None to skip.
    warm_start_ckpt: Optional[str] = None
    warm_start_piece_dim: int = 8
    warm_start_dropout: float = 0.0


def oracle_weight_for_generation(gen: int, anneal_gens: int) -> float:
    if anneal_gens <= 0:
        return 0.0
    return float(np.clip(1.0 - gen / anneal_gens, 0.0, 1.0))


def _init_wandb(cfg: Phase2Config):
    if not cfg.use_wandb:
        return None
    if not _WANDB_AVAILABLE:
        print("[phase2] use_wandb=True but wandb not installed; logging to console only.")
        return None
    return wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        tags=list(cfg.wandb_tags),
        config=dataclasses.asdict(cfg),
    )


def run_phase2(
    cfg: Phase2Config,
    log_fn=print,
) -> Tuple[AlphaModel, ReplayBuffer]:
    rng = np.random.default_rng(cfg.seed)
    tf.random.set_seed(cfg.seed)

    model = AlphaModel(
        depth=cfg.model_depth,
        num_heads=cfg.model_num_heads,
        num_layers=cfg.model_num_layers,
        dropout_rate=cfg.model_dropout,
    )

    if cfg.warm_start_ckpt:
        from .warm_start import warm_start_alpha_from_autoregressive
        warm_start_alpha_from_autoregressive(
            model,
            cfg.warm_start_ckpt,
            piece_dim=cfg.warm_start_piece_dim,
            depth=cfg.model_depth,
            num_heads=cfg.model_num_heads,
            num_layers=cfg.model_num_layers,
            dropout_rate=cfg.warm_start_dropout,
        )

    optimizer = keras.optimizers.AdamW(
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    trainer = Trainer(model, optimizer, value_loss_weight=cfg.value_loss_weight)

    nn_valuator = NeuralValuator(model)
    oracle = DecomposeOracle()
    valuator = BlendedValuator(nn_valuator, oracle, oracle_weight=1.0)
    mcts = MCTS(valuator=valuator)

    buffer = ReplayBuffer(capacity=cfg.buffer_capacity, seed=cfg.seed)
    wandb_run = _init_wandb(cfg)

    # Game seeds: take a chunk per generation so generations don't overlap.
    next_seed = cfg.seed * 1_000_003 + 1

    try:
        for gen in range(cfg.num_generations):
            gen_t0 = time.perf_counter()
            weight = oracle_weight_for_generation(gen, cfg.anneal_generations)
            valuator.set_oracle_weight(weight)

            # ----- Self-play -----
            sp_t0 = time.perf_counter()
            sp_stats = []
            n_games = cfg.games_per_generation
            for game_idx in range(n_games):
                game_t0 = time.perf_counter()
                transitions, stats = play_game(
                    seed=next_seed,
                    num_steps=cfg.num_steps_per_game,
                    mcts=mcts,
                    num_simulations=cfg.num_simulations,
                    rng=rng,
                    garbage_chance=cfg.garbage_chance,
                    garbage_min=cfg.garbage_min,
                    garbage_max=cfg.garbage_max,
                )
                buffer.push_many(transitions)
                sp_stats.append(stats)
                next_seed += 1
                game_secs = time.perf_counter() - game_t0
                log_fn(
                    f"  gen={gen}  game={game_idx+1:>3}/{n_games}  "
                    f"steps={stats['steps']:>4}  surv={stats['survived']}  "
                    f"app={stats['app']:>5.3f}  b2b={stats['max_b2b']:>3}  "
                    f"atk={stats['total_attack']:>4.0f}  "
                    f"t={game_secs:>5.1f}s"
                )
            sp_seconds = time.perf_counter() - sp_t0

            avg_app = float(np.mean([s["app"] for s in sp_stats]))
            avg_steps = float(np.mean([s["steps"] for s in sp_stats]))
            survival = float(np.mean([s["survived"] for s in sp_stats]))
            avg_b2b = float(np.mean([s["max_b2b"] for s in sp_stats]))
            avg_attack = float(np.mean([s["total_attack"] for s in sp_stats]))
            max_steps_in_gen = int(np.max([s["steps"] for s in sp_stats]))
            max_b2b_in_gen = int(np.max([s["max_b2b"] for s in sp_stats]))

            # ----- Training -----
            tr_t0 = time.perf_counter()
            train_losses, train_p, train_v = [], [], []
            last_metrics = None
            n_train = cfg.train_steps_per_generation
            # Print ~10 progress lines per training phase, regardless of size.
            log_every = max(n_train // 10, 1)
            for step_idx in range(n_train):
                m = trainer.train_step(buffer, cfg.batch_size)
                if m is not None:
                    train_losses.append(m.loss)
                    train_p.append(m.policy_loss)
                    train_v.append(m.value_loss)
                    last_metrics = m
                if (step_idx + 1) % log_every == 0 and last_metrics is not None:
                    elapsed = time.perf_counter() - tr_t0
                    log_fn(
                        f"  gen={gen}  train={step_idx+1:>4}/{n_train}  "
                        f"L={last_metrics.loss:.3f}  "
                        f"(P={last_metrics.policy_loss:.3f} "
                        f"V={last_metrics.value_loss:.3f})  "
                        f"t={elapsed:>5.1f}s"
                    )
            tr_seconds = time.perf_counter() - tr_t0

            gen_seconds = time.perf_counter() - gen_t0

            line = (
                f"gen={gen:>3}  α={weight:.2f}  buf={len(buffer):>5}  "
                f"app={avg_app:>5.3f}  steps={avg_steps:>5.1f}  "
                f"surv={survival*100:>4.0f}%  b2b={avg_b2b:>4.1f}/{max_b2b_in_gen}  "
            )
            if last_metrics is not None:
                line += (
                    f"L={last_metrics.loss:.3f} (P={last_metrics.policy_loss:.3f} "
                    f"V={last_metrics.value_loss:.3f})  "
                )
            line += (
                f"sp={sp_seconds:>5.1f}s  tr={tr_seconds:>5.1f}s  total={gen_seconds:>5.1f}s"
            )
            log_fn(line)

            if wandb_run is not None:
                metrics = {
                    "gen": gen,
                    "oracle_weight": weight,
                    "buffer_size": len(buffer),
                    # Gameplay (avg over games in this gen)
                    "play/app": avg_app,
                    "play/steps": avg_steps,
                    "play/max_steps": max_steps_in_gen,
                    "play/survival": survival,
                    "play/avg_max_b2b": avg_b2b,
                    "play/max_b2b": max_b2b_in_gen,
                    "play/avg_total_attack": avg_attack,
                    # Timing
                    "time/self_play_s": sp_seconds,
                    "time/train_s": tr_seconds,
                    "time/total_s": gen_seconds,
                }
                if train_losses:
                    metrics.update({
                        "train/loss": float(np.mean(train_losses)),
                        "train/loss_last": float(train_losses[-1]),
                        "train/policy_loss": float(np.mean(train_p)),
                        "train/value_loss": float(np.mean(train_v)),
                        "train/value_loss_last": float(train_v[-1]),
                        "train/steps": len(train_losses),
                    })
                wandb.log(metrics)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return model, buffer
