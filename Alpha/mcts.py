"""PUCT MCTS for solo Tetris.

Phase 1 runs simulations sequentially with a uniform-policy oracle valuator.
The valuator interface is batch-aware so a Phase 2 NN backend slots in
without rewriting the search loop.

Tree-wide Q-min/Q-max normalization keeps the PUCT exploitation term on the
same scale as the exploration term across heuristic-V ranges.

Backup uses discounted returns G_t = r_{t+1} + γ·G_{t+1} (n-step bootstrap
with γ ∈ (0,1]) so per-step rewards (Tetris attacks) are credited along
with the leaf V.  AlphaZero's chess/go formulation drops r and uses only
terminal V — that's wrong for MDPs with intermediate rewards.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np

from TetrisEnv.CTetrisCore import TetrisCore
from .tree import Node
from .valuator import Valuator
from .action_space import canonical_indices, NUM_ACTIONS
from . import config as cfg


REWARD_SCALE = 5.0   # attack / 5 → unit-scale reward (a Tetris contributes ~1.0)
GAMMA = 0.99


class MCTS:
    def __init__(
        self,
        valuator: Valuator,
        c_puct: float = cfg.C_PUCT,
        dirichlet_alpha: float = cfg.DIRICHLET_ALPHA,
        dirichlet_eps: float = cfg.DIRICHLET_EPS,
        gamma: float = GAMMA,
        reward_scale: float = REWARD_SCALE,
    ) -> None:
        self.valuator = valuator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.q_min: float = 0.0
        self.q_max: float = 0.0
        self._q_seen: bool = False

    def _update_q_range(self, v: float) -> None:
        if not self._q_seen:
            self.q_min = v
            self.q_max = v
            self._q_seen = True
        else:
            if v < self.q_min: self.q_min = v
            if v > self.q_max: self.q_max = v

    def _normalized_Q(self, node: Node) -> np.ndarray:
        Q = node.Q()
        if not self._q_seen or self.q_max <= self.q_min:
            return Q
        return (Q - self.q_min) / (self.q_max - self.q_min)

    def _puct_select(self, node: Node) -> int:
        Q_norm = self._normalized_Q(node)
        sqrt_total = math.sqrt(max(node.total_visits(), 1))
        U = self.c_puct * node.P * sqrt_total / (1.0 + node.N)
        scores = Q_norm + U
        # Unvisited actions get a default optimistic Q (mid-range) so they're
        # explored before saturating any single visited path.
        if self._q_seen and self.q_max > self.q_min:
            unvisited = node.N == 0
            scores[unvisited] = 0.5 + U[unvisited]
        return int(np.argmax(scores))

    def _materialize_child(self, parent: Node, action_idx: int) -> Node:
        is_hold, rot, col, lr, sp = parent.placements[action_idx]
        child_state = parent.state.clone()
        ev = child_state.apply_placement(int(is_hold), int(rot), int(col), int(lr), int(sp))
        # Convert raw attack into a unit-scale reward.  Death contributes a
        # large negative, but only via the leaf V (-1) — the immediate reward
        # for the death-causing action is just its (zero) attack.
        r = float(ev.attack) / self.reward_scale
        child = Node(child_state, is_terminal=bool(ev.terminal), incoming_reward=r)
        parent.children[action_idx] = child
        return child

    def _expand(self, node: Node) -> float:
        """Enumerate placements, filter to canonical-kept, query valuator,
        scatter P_320 onto per-placement priors, fill node, return V."""
        all_placements = node.state.enumerate_placements(include_hold=True)
        if len(all_placements) == 0:
            node.is_terminal = True
            return -1.0

        action_idx_full, kept_full, valid_mask = canonical_indices(
            all_placements, node.state
        )
        keep = np.flatnonzero(kept_full)
        n = len(keep)
        if n == 0:
            # No placements survived the canonical mapping — treat as terminal.
            node.is_terminal = True
            return -1.0

        placements = all_placements[keep]
        canonical_idx = action_idx_full[keep]

        P_320, V = self.valuator.evaluate(node.state)
        if P_320 is None:
            P = np.full(n, 1.0 / n, dtype=np.float32)
        elif len(P_320) != NUM_ACTIONS:
            raise RuntimeError(
                f"Valuator returned P of size {len(P_320)}, expected {NUM_ACTIONS}"
            )
        else:
            P = P_320[canonical_idx].astype(np.float32)
            s = P.sum()
            if s > 0:
                P = P / s
            else:
                P = np.full(n, 1.0 / n, dtype=np.float32)

        node.expand(placements, canonical_idx, valid_mask, P, V)
        return V

    def add_root_noise(self, root: Node) -> None:
        """Inject Dirichlet exploration noise at the root for self-play."""
        if self.dirichlet_eps <= 0 or root.P is None:
            return
        n = len(root.P)
        if n == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * n).astype(np.float32)
        root.P[:] = (1.0 - self.dirichlet_eps) * root.P + self.dirichlet_eps * noise

    def run(self, root: Node, num_simulations: int) -> None:
        """Run num_simulations on the existing tree rooted at `root`."""
        if not root.is_expanded:
            v = self._expand(root)
            self._update_q_range(v)

        for _ in range(num_simulations):
            self._simulate(root)

    def _simulate(self, root: Node) -> None:
        # path entries are (parent, action, immediate_reward_of_action)
        path = []
        node = root

        # Descend until a leaf or terminal
        while node.is_expanded and not node.is_terminal:
            a = self._puct_select(node)
            if node.children[a] is None:
                child = self._materialize_child(node, a)
                path.append((node, a, child.incoming_reward))
                node = child
                break
            child = node.children[a]
            path.append((node, a, child.incoming_reward))
            node = child

        # Evaluate the leaf
        if node.is_terminal:
            v = -1.0
        elif not node.is_expanded:
            v = self._expand(node)
        else:
            v = node.v_estimate

        # Backprop discounted returns.  G_T = V(leaf); G_t = r_{t+1} + γ·G_{t+1}.
        self._update_q_range(v)
        G = v
        for parent, a, r in reversed(path):
            G = r + self.gamma * G
            parent.N[a] += 1
            parent.W[a] += G
            self._update_q_range(G)


def select_action(root: Node, temperature: float = 0.0) -> int:
    """Sample (or argmax) an action from root.N visit counts."""
    if root.N is None or len(root.N) == 0:
        return -1
    if temperature <= 0.0:
        return int(np.argmax(root.N))
    # Temperature-scaled visit-count sampling.
    counts = root.N.astype(np.float64)
    if counts.sum() == 0:
        # No simulations ran — fall back to prior.
        probs = root.P.astype(np.float64)
    else:
        probs = counts ** (1.0 / temperature)
        probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def reuse_subtree(root: Node, action_idx: int) -> Optional[Node]:
    """After picking action_idx at root, return the corresponding child as
    the new root (preserving its accumulated subtree)."""
    if root.children is None or action_idx < 0 or action_idx >= len(root.children):
        return None
    return root.children[action_idx]
