"""Self-play game driver.

For each move:
  1. Capture NN inputs (fixed shapes) at the current state.
  2. Run MCTS (BlendedValuator) for `num_simulations`.
  3. Scatter MCTS visit counts into the canonical 320-dim action space → π_target.
  4. Sample (or argmax) a placement, apply it, record the immediate reward.
  5. Reuse the child subtree as the new root.

When the game ends, compute discounted Monte Carlo returns z_t and emit one
Transition per step.  Terminal states contribute V=-1 to the bootstrap return;
if the game runs out of allotted steps the bootstrap is 0 (treat as truncation).
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np

from TetrisEnv.CTetrisCore import TetrisCore
from .tree import Node
from .mcts import MCTS, REWARD_SCALE, GAMMA
from .action_space import NUM_ACTIONS
from .neural_valuator import state_to_inputs
from .replay_buffer import Transition
from . import config as cfg


def _temperature_for_move(move_idx: int) -> float:
    if move_idx < cfg.TEMP_HIGH_MOVES:
        return 1.0
    if move_idx < cfg.TEMP_HIGH_MOVES + cfg.TEMP_DECAY_MOVES:
        # Linear decay from 1.0 → 0.3 across TEMP_DECAY_MOVES.
        frac = (move_idx - cfg.TEMP_HIGH_MOVES) / float(cfg.TEMP_DECAY_MOVES)
        return float(1.0 - 0.7 * frac)
    return 0.3


def _sample_action(visit_counts: np.ndarray, temperature: float,
                   rng: np.random.Generator) -> int:
    if visit_counts.sum() == 0:
        return -1
    if temperature <= 1e-6:
        return int(np.argmax(visit_counts))
    counts = visit_counts.astype(np.float64)
    probs = counts ** (1.0 / temperature)
    s = probs.sum()
    if s <= 0:
        return int(np.argmax(visit_counts))
    probs = probs / s
    return int(rng.choice(len(probs), p=probs))


def play_game(
    seed: int,
    num_steps: int,
    mcts: MCTS,
    num_simulations: int,
    rng: np.random.Generator,
    gamma: float = GAMMA,
    reward_scale: float = REWARD_SCALE,
    use_dirichlet_noise: bool = True,
    garbage_chance: float = 0.0,
    garbage_min: int = 0,
    garbage_max: int = 0,
) -> Tuple[List[Transition], dict]:
    """Run a single solo game; return (transitions, stats).

    If `garbage_chance > 0`, ambient garbage is rolled after each placement
    using the state's deterministic RNG (same scheme as b2b_run_eval_games),
    creating real survival pressure for the value head to learn against.
    """

    ts = TetrisCore(
        seed=seed, queue_size=cfg.QUEUE_SIZE,
        garbage_push_delay=cfg.GARBAGE_PUSH_DELAY,
    )
    root = Node(ts)

    captured = []   # tuples of (board, pieces, bcg, pi_320, valid_mask_320)
    rewards = []    # scaled per-step rewards (attack / reward_scale)

    total_attack = 0.0
    max_b2b = 0
    died = False

    for move_idx in range(num_steps):
        # Capture state inputs (fixed shapes).
        board, pieces, bcg = state_to_inputs(root.state)

        # Run MCTS.  Adding Dirichlet noise after the root has been expanded.
        mcts.run(root, num_simulations)
        if use_dirichlet_noise and mcts.dirichlet_eps > 0:
            mcts.add_root_noise(root)
            extra = max(num_simulations // 4, 8)
            mcts.run(root, extra)

        if root.is_terminal or root.num_actions == 0:
            died = True
            break

        N = root.N
        # Scatter visit counts into the canonical action space for π target.
        pi_320 = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if N.sum() > 0:
            np.add.at(pi_320, root.canonical_idx, N.astype(np.float32))
            pi_320 = pi_320 / pi_320.sum()
        else:
            # No visits — use the prior projected into 320 space.
            np.add.at(pi_320, root.canonical_idx, root.P.astype(np.float32))
            s = pi_320.sum()
            if s > 0:
                pi_320 = pi_320 / s

        valid_mask_320 = root.valid_mask_320.copy()

        # Sample / argmax via temperature schedule (over the per-placement N).
        tau = _temperature_for_move(move_idx)
        a = _sample_action(N, tau, rng)
        if a < 0:
            died = True
            break

        chosen = root.placements[a]
        new_state = root.state.clone()
        ev = new_state.apply_placement(
            int(chosen[0]), int(chosen[1]), int(chosen[2]),
            int(chosen[3]), int(chosen[4]),
        )

        captured.append((board, pieces, bcg, pi_320, valid_mask_320))
        rewards.append(float(ev.attack) / reward_scale)
        total_attack += float(ev.attack)
        if new_state.b2b > max_b2b:
            max_b2b = new_state.b2b

        if ev.terminal:
            died = True
            break

        # Inject ambient garbage AFTER the apply (which already ticked timers
        # and pushed any ready garbage).  The new entry sits with timer =
        # garbage_push_delay, getting decremented on the next non-clear move.
        if garbage_chance > 0.0 and garbage_max > 0:
            new_state.inject_random_garbage(garbage_chance, garbage_min, garbage_max)

        child = root.children[a] if root.children else None
        if child is None:
            child = Node(new_state, is_terminal=False)
        else:
            # Subtree was built before garbage injection — its state is now
            # out-of-sync.  Drop the subtree if we injected, since the cached
            # priors/Q values reflect a different future.
            if garbage_chance > 0.0 and new_state.total_garbage != child.state.total_garbage:
                child = Node(new_state, is_terminal=False)
        root = child

    # Compute discounted MC returns.
    bootstrap = -1.0 if died else 0.0
    G = bootstrap
    z_targets: List[float] = [0.0] * len(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        G = rewards[i] + gamma * G
        z_targets[i] = G

    transitions = [
        Transition(board, pieces, bcg, pi, valid, z)
        for (board, pieces, bcg, pi, valid), z in zip(captured, z_targets)
    ]

    stats = {
        "steps": len(captured),
        "survived": 0 if died else 1,
        "total_attack": total_attack,
        "max_b2b": max_b2b,
        "app": total_attack / max(len(captured), 1),
    }
    return transitions, stats
