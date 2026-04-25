"""In-memory ring buffer for self-play transitions.

One transition = (NN inputs at state s, π_320 target, valid_mask_320, z target):
  - NN inputs from neural_valuator.state_to_inputs(s) — fixed shapes
  - π_320 is the MCTS-improved visit distribution scattered into the
    canonical action space; sums to 1, zero on invalid slots
  - valid_mask_320 marks which slots had reachable placements
    (kept for masking during loss computation)
  - z is the discounted Monte Carlo return observed during the game

All inputs and targets are now fixed-shape, so the trainer can stack
without padding.
"""

from __future__ import annotations
from collections import deque
from typing import Iterator, List

import numpy as np


class Transition:
    __slots__ = ("board", "pieces", "bcg", "pi", "valid_mask", "z")

    def __init__(
        self,
        board: np.ndarray,        # (24, 10, 1) float32
        pieces: np.ndarray,       # (7,) int64
        bcg: np.ndarray,          # (3,) float32
        pi: np.ndarray,           # (320,) float32 — MCTS visit distribution, sums to 1
        valid_mask: np.ndarray,   # (320,) bool   — true where reachable
        z: float,                 # scalar return target
    ) -> None:
        self.board = board
        self.pieces = pieces
        self.bcg = bcg
        self.pi = pi
        self.valid_mask = valid_mask
        self.z = float(z)


class ReplayBuffer:
    """FIFO ring with random-sample mini-batches for training."""

    def __init__(self, capacity: int = 1_000_000, seed: int = 0) -> None:
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def push_many(self, ts: List[Transition]) -> None:
        self._buf.extend(ts)

    def sample(self, batch_size: int) -> List[Transition]:
        n = len(self._buf)
        if n == 0:
            return []
        idx = self._rng.integers(0, n, size=min(batch_size, n))
        return [self._buf[int(i)] for i in idx]

    def iter_recent(self, n: int) -> Iterator[Transition]:
        n = min(n, len(self._buf))
        for i in range(len(self._buf) - n, len(self._buf)):
            yield self._buf[i]
