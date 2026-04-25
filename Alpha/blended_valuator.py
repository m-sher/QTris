"""Convex blend of two Valuators — the scaffolding mechanism for Path B.

  V_blended  = (1-α)·V_nn + α·V_oracle
  P_blended  = renorm( (1-α)·P_nn + α·P_oracle )

Annealing α from 1.0 → 0.0 across early generations gives the NN a curriculum:
oracle-dominated MCTS produces high-quality trajectories while the NN learns,
and once the NN matches the oracle the scaffolding is dropped.

If either side returns P=None, that side's prior is treated as uniform over
the placement count from the oracle (which always knows the true N because
it enumerates).
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

from TetrisEnv.CTetrisCore import TetrisCore
from .valuator import Valuator


class BlendedValuator(Valuator):
    def __init__(
        self,
        nn: Valuator,
        oracle: Valuator,
        oracle_weight: float = 1.0,
    ) -> None:
        self.nn = nn
        self.oracle = oracle
        self.oracle_weight = float(oracle_weight)

    def set_oracle_weight(self, w: float) -> None:
        self.oracle_weight = float(np.clip(w, 0.0, 1.0))

    def _blend(
        self,
        nn_out: Tuple[Optional[np.ndarray], float],
        ora_out: Tuple[Optional[np.ndarray], float],
    ) -> Tuple[Optional[np.ndarray], float]:
        P_nn, V_nn = nn_out
        P_ora, V_ora = ora_out
        a = self.oracle_weight

        # Determine N for any uniform fallback.
        n = None
        if P_nn is not None:
            n = len(P_nn)
        elif P_ora is not None:
            n = len(P_ora)

        if n is None:
            return None, (1.0 - a) * V_nn + a * V_ora

        if P_nn is None:
            P_nn = np.full(n, 1.0 / n, dtype=np.float32)
        if P_ora is None:
            P_ora = np.full(n, 1.0 / n, dtype=np.float32)

        # Defensive: if P sizes ever disagree, fall back to oracle (it's the
        # source of truth on placement counts).
        if len(P_nn) != n or len(P_ora) != n:
            P_blend = P_ora if len(P_ora) == n else P_nn
        else:
            P_blend = (1.0 - a) * P_nn + a * P_ora
            s = P_blend.sum()
            if s > 0:
                P_blend = (P_blend / s).astype(np.float32)
            else:
                P_blend = np.full(n, 1.0 / n, dtype=np.float32)

        V_blend = (1.0 - a) * V_nn + a * V_ora
        return P_blend, float(V_blend)

    def evaluate(
        self, state: TetrisCore
    ) -> Tuple[Optional[np.ndarray], float]:
        # Short-circuits when the blend is degenerate.
        if self.oracle_weight >= 1.0:
            return self.oracle.evaluate(state)
        if self.oracle_weight <= 0.0:
            return self.nn.evaluate(state)
        nn_out = self.nn.evaluate(state)
        ora_out = self.oracle.evaluate(state)
        return self._blend(nn_out, ora_out)

    def evaluate_many(
        self, states: List[TetrisCore]
    ) -> List[Tuple[Optional[np.ndarray], float]]:
        if not states:
            return []
        if self.oracle_weight >= 1.0:
            return self.oracle.evaluate_many(states)
        if self.oracle_weight <= 0.0:
            return self.nn.evaluate_many(states)
        nn_out = self.nn.evaluate_many(states)
        ora_out = self.oracle.evaluate_many(states)
        return [self._blend(a, b) for a, b in zip(nn_out, ora_out)]
