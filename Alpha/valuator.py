"""Valuator: state -> (action_priors P_320, scalar value V).

P is always a fixed-size 320-dim vector aligned with the canonical action
space (see Alpha/action_space.py).  Padding-slot probabilities are not
required to be zero — the consumer applies a valid-mask before normalizing.

Two oracle valuators ship for Phase 1 / scaffolding:

  - `CheapStateValuator` — V from b2b counter and stack height only; P is
    None (sentinel = uniform over valid slots).  ~µs per call.
  - `DecomposeOracle` — V = max over placements of (21-component decompose
    sum); P also returned via softmax over decompose-scores assigned to the
    matching canonical slots, others 0.  ~3 ms per call.

Both follow the same interface as `NeuralValuator`, so MCTS treats them
interchangeably.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from TetrisEnv.CTetrisCore import TetrisCore
from .action_space import canonical_indices, NUM_ACTIONS


class Valuator(ABC):
    """state -> (P_320, V).

    P_320 may be None to indicate "uniform over valid slots" — the search
    will fill it.  When non-None, P_320.shape == (NUM_ACTIONS,) — invalid
    slots may have nonzero values; the consumer masks them.
    """

    @abstractmethod
    def evaluate(self, state: TetrisCore) -> Tuple[Optional[np.ndarray], float]:
        ...

    def evaluate_many(self, states: List[TetrisCore]) -> List[Tuple[Optional[np.ndarray], float]]:
        return [self.evaluate(s) for s in states]


def _max_height(board: np.ndarray, board_height: int) -> int:
    nz = np.flatnonzero(board[:board_height])
    if len(nz) == 0:
        return 0
    return board_height - int(nz[0])


class CheapStateValuator(Valuator):
    """V from b2b and stack height only.  No per-placement work; ~µs per call.

    Defaults are tuned so V lands in roughly [-1, 1] across normal play:
        V = clip( w_b2b·b2b / 30  -  w_height·h / 24,  -1, 1 )
    """

    def __init__(self, w_b2b: float = 1.0, w_height: float = 0.5) -> None:
        self.w_b2b = w_b2b
        self.w_height = w_height

    def evaluate(self, state: TetrisCore) -> Tuple[Optional[np.ndarray], float]:
        h = _max_height(state.board, state.board_height)
        v = self.w_b2b * (state.b2b / 30.0) - self.w_height * (h / 24.0)
        return None, float(np.clip(v, -1.0, 1.0))


class DecomposeOracle(Valuator):
    """V from the 21-component decomposition AND informative action priors,
    delivered in canonical 320-dim format.

    For each placement we compute its decompose-sum.  We then:
      V = max(score) / V_SCALE
      P_320 = softmax-by-canonical-slot:  for each kept placement, the slot
              gets exp((score - max) / P_TEMP); the rest stay 0.  Renormalized.
    """

    V_SCALE = 100.0
    P_TEMP = 8.0

    def evaluate(self, state: TetrisCore) -> Tuple[Optional[np.ndarray], float]:
        components = state.decompose()
        if len(components) == 0:
            return None, -1.0
        scores = components.sum(axis=1)
        v = float(np.clip(scores.max() / self.V_SCALE, -1.0, 1.0))

        placements = state.enumerate_placements(include_hold=True)
        if len(placements) != len(scores):
            # Defensive — shouldn't happen, but if enumerate / decompose disagree,
            # don't trust the prior; let the consumer fall back to uniform.
            return None, v
        action_idx, kept, _valid = canonical_indices(placements, state)

        P = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if not kept.any():
            return None, v
        kept_idx = action_idx[kept]
        kept_scores = scores[kept]
        shifted = (kept_scores - kept_scores.max()) / self.P_TEMP
        np.clip(shifted, -50.0, 0.0, out=shifted)
        weights = np.exp(shifted).astype(np.float32)
        P[kept_idx] = weights
        s = P.sum()
        if s > 0:
            P = P / s
        return P, v
