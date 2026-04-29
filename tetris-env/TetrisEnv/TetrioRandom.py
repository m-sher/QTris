from .Pieces import PieceType
from typing import List
import random

TETROMINOS = [
    PieceType.Z,
    PieceType.L,
    PieceType.O,
    PieceType.S,
    PieceType.I,
    PieceType.J,
    PieceType.T,
]


class TetrioRNG:
    def __init__(self, seed: int | None = None):
        self._seed = seed
        self.reset()

    def next_int(self) -> int:
        self._t = (16807 * self._t) % 2147483647
        return int(self._t)

    def next_float(self) -> float:
        return (self.next_int() - 1) / 2147483646.0

    def next_bag(self) -> List[PieceType]:
        bag = TETROMINOS[:]
        i = len(bag) - 1
        while i > 0:
            j = int(self.next_float() * (i + 1))
            bag[i], bag[j] = bag[j], bag[i]
            i -= 1
        return bag

    def reset(self) -> None:
        seed = self._seed if self._seed is not None else random.randint(0, 2147483646)
        t = seed % 2147483647
        if t <= 0:
            t += 2147483646
        self._t = t
