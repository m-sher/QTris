"""Wrap an AlphaModel into the Valuator interface.

State → NN inputs conversion lives here so MCTS / self-play don't need
to know about tensor shapes.  The network outputs fixed (B, NUM_ACTIONS)
logits + (B,) values; we softmax over the union with the per-state valid
mask before returning P_320.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

from TetrisEnv.CTetrisCore import TetrisCore
from .network import AlphaModel
from .action_space import canonical_indices, NUM_ACTIONS


# ============================================================
# State → tensor conversion
# ============================================================

BOARD_HEIGHT = 24
BOARD_COLS = 10
QUEUE_TOKENS = 7  # active + hold + 5 queue


def _bitmask_board_to_cells(mask_rows: np.ndarray, board_height: int) -> np.ndarray:
    """Convert TetrisCore (40,) uint16 bitmask to (board_height, 10) float32 cell array."""
    out = np.zeros((board_height, BOARD_COLS), dtype=np.float32)
    bits = (np.arange(BOARD_COLS, dtype=np.uint16))
    rows = mask_rows[:board_height].astype(np.uint16)
    for c in range(BOARD_COLS):
        col_bit = np.uint16(1) << bits[c]
        out[:, c] = ((rows & col_bit) != 0).astype(np.float32)
    return out


def state_to_inputs(state: TetrisCore) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (board, pieces, bcg) numpy arrays for ONE state.  Fixed shapes.

      board:  (24, 10, 1) float32
      pieces: (7,) int64                  (active, hold, q0..q4)
      bcg:    (3,) float32                (b2b, combo, total_garbage)
    """
    board = _bitmask_board_to_cells(state.board, state.board_height)[..., None]

    queue = state.queue[:5]
    queue = queue + [0] * (5 - len(queue))
    pieces = np.array(
        [state.active_piece, state.hold_piece] + queue,
        dtype=np.int64,
    )

    bcg = np.array(
        [float(state.b2b), float(state.combo), float(state.total_garbage)],
        dtype=np.float32,
    )
    return board, pieces, bcg


def batch_inputs(
    boards: List[np.ndarray],
    pieces_list: List[np.ndarray],
    bcgs: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack into a uniform batch (no padding — all inputs are fixed-shape)."""
    return (
        np.stack(boards, axis=0).astype(np.float32),
        np.stack(pieces_list, axis=0).astype(np.int64),
        np.stack(bcgs, axis=0).astype(np.float32),
    )


# ============================================================
# Valuator implementation
# ============================================================

from .valuator import Valuator


class NeuralValuator(Valuator):
    """Wraps an AlphaModel.  evaluate() runs a 1-element batch; prefer
    evaluate_many() for batched GPU throughput.

    The model emits raw 320-dim logits regardless of state.  The valid mask
    (which slots are reachable from this state) is applied here before the
    softmax so the returned P_320 only has mass on reachable slots."""

    def __init__(self, model: AlphaModel) -> None:
        self.model = model

    def _forward_batch(
        self,
        board: np.ndarray,
        pieces: np.ndarray,
        bcg: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        board_t = tf.convert_to_tensor(board, dtype=tf.float32)
        pieces_t = tf.convert_to_tensor(pieces, dtype=tf.int64)
        bcg_t = tf.convert_to_tensor(bcg, dtype=tf.float32)
        logits, v = self.model((board_t, pieces_t, bcg_t), training=False)
        return logits.numpy(), v.numpy()

    def evaluate(self, state: TetrisCore) -> Tuple[Optional[np.ndarray], float]:
        return self.evaluate_many([state])[0]

    def evaluate_many(
        self, states: List[TetrisCore]
    ) -> List[Tuple[Optional[np.ndarray], float]]:
        if not states:
            return []

        boards, pieces_list, bcgs = [], [], []
        valid_masks: List[np.ndarray] = []
        for s in states:
            b, p, c = state_to_inputs(s)
            boards.append(b)
            pieces_list.append(p)
            bcgs.append(c)
            placements = s.enumerate_placements(include_hold=True)
            _, _, mask = canonical_indices(placements, s)
            valid_masks.append(mask)

        board_b, pieces_b, bcg_b = batch_inputs(boards, pieces_list, bcgs)
        logits, v = self._forward_batch(board_b, pieces_b, bcg_b)

        out: List[Tuple[Optional[np.ndarray], float]] = []
        for i, mask in enumerate(valid_masks):
            if not mask.any():
                out.append((None, -1.0))
                continue
            row = np.where(mask, logits[i], -1e9)
            row = row - row.max()
            P = np.exp(row, dtype=np.float64)
            P = np.where(mask, P, 0.0)
            s = P.sum()
            P = (P / s).astype(np.float32) if s > 0 else None
            out.append((P, float(v[i])))
        return out
