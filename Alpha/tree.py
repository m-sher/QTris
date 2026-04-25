"""MCTS tree node.

A Node owns a TetrisCore state and (after expansion) numpy-backed visit/value/
prior arrays for its actions.  Children are lazy: a child node materializes
the first time its action is descended into.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from TetrisEnv.CTetrisCore import TetrisCore


class Node:
    __slots__ = (
        "state", "is_terminal", "is_expanded",
        "placements", "canonical_idx", "valid_mask_320",
        "P", "N", "W", "children",
        "v_estimate", "incoming_reward",
    )

    def __init__(self, state: TetrisCore, is_terminal: bool = False,
                 incoming_reward: float = 0.0) -> None:
        self.state: TetrisCore = state
        self.is_terminal: bool = is_terminal
        self.is_expanded: bool = False
        # Per-placement (post-canonical-filter):
        self.placements: Optional[np.ndarray] = None       # (n, 5) int32
        self.canonical_idx: Optional[np.ndarray] = None    # (n,) int32 — slot index in [0, 320)
        self.P: Optional[np.ndarray] = None                # (n,) float32 — prior at the placement level
        self.N: Optional[np.ndarray] = None                # (n,) int32
        self.W: Optional[np.ndarray] = None                # (n,) float32
        self.children: Optional[List[Optional[Node]]] = None
        # Snapshot of the 320-dim valid mask (used when scattering pi targets at training time).
        self.valid_mask_320: Optional[np.ndarray] = None   # (320,) bool
        self.v_estimate: float = 0.0
        self.incoming_reward: float = float(incoming_reward)

    @property
    def num_actions(self) -> int:
        return 0 if self.placements is None else len(self.placements)

    def expand(
        self,
        placements: np.ndarray,
        canonical_idx: np.ndarray,
        valid_mask_320: np.ndarray,
        P: np.ndarray,
        V: float,
    ) -> None:
        n = len(placements)
        self.placements = np.ascontiguousarray(placements, dtype=np.int32)
        self.canonical_idx = np.ascontiguousarray(canonical_idx, dtype=np.int32)
        self.valid_mask_320 = np.ascontiguousarray(valid_mask_320, dtype=np.bool_)
        self.P = np.ascontiguousarray(P, dtype=np.float32)
        self.N = np.zeros(n, dtype=np.int32)
        self.W = np.zeros(n, dtype=np.float32)
        self.children = [None] * n
        self.v_estimate = float(V)
        self.is_expanded = True

    def total_visits(self) -> int:
        return 0 if self.N is None else int(self.N.sum())

    def Q(self) -> np.ndarray:
        """Per-action Q values (W/N), 0 where unvisited."""
        n = self.N
        out = np.zeros_like(self.W)
        mask = n > 0
        out[mask] = self.W[mask] / n[mask]
        return out
