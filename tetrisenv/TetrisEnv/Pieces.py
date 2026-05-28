import numpy as np
from enum import Enum


# ruff: noqa: E741
class PieceType(Enum):
    N = 0  # NULL - empty and only used for initial hold
    I = 1
    J = 2
    L = 3
    O = 4
    S = 5
    T = 6
    Z = 7
    G = 8  # Garbage


class Piece:
    def __init__(
        self,
        piece_type: PieceType = PieceType.N,
        loc: np.ndarray = np.zeros((2,), dtype=np.int32),
        r: int = 0,
        cells: np.ndarray = np.zeros((4, 2), dtype=np.int32),
    ) -> None:
        self.piece_type = piece_type

        self.loc = loc
        self.delta_loc = np.zeros((2,), np.int32)

        self.r = r
        self.delta_r = 0

        self.cells = cells
