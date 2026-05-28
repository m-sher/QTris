from .Pieces import PieceType, Piece
from .helpers import overlaps
import numpy as np
from typing import Dict, Tuple


class RotationSystem:
    def __init__(self) -> None:
        self.orientations = {
            PieceType.I: np.array(
                [
                    [[1, 0], [1, 1], [1, 2], [1, 3]],
                    [[0, 2], [1, 2], [2, 2], [3, 2]],
                    [[2, 0], [2, 1], [2, 2], [2, 3]],
                    [[0, 1], [1, 1], [2, 1], [3, 1]],
                ],
                dtype=np.int32,
            ),
            PieceType.J: np.array(
                [
                    [[0, 0], [1, 0], [1, 1], [1, 2]],
                    [[0, 1], [0, 2], [1, 1], [2, 1]],
                    [[1, 0], [1, 1], [1, 2], [2, 2]],
                    [[0, 1], [1, 1], [2, 0], [2, 1]],
                ],
                dtype=np.int32,
            ),
            PieceType.L: np.array(
                [
                    [[0, 2], [1, 0], [1, 1], [1, 2]],
                    [[0, 1], [1, 1], [2, 1], [2, 2]],
                    [[1, 0], [1, 1], [1, 2], [2, 0]],
                    [[0, 0], [0, 1], [1, 1], [2, 1]],
                ],
                dtype=np.int32,
            ),
            PieceType.O: np.array(
                [
                    [[0, 1], [0, 2], [1, 1], [1, 2]],
                    [[0, 1], [0, 2], [1, 1], [1, 2]],
                    [[0, 1], [0, 2], [1, 1], [1, 2]],
                    [[0, 1], [0, 2], [1, 1], [1, 2]],
                ],
                dtype=np.int32,
            ),
            PieceType.S: np.array(
                [
                    [[0, 1], [0, 2], [1, 0], [1, 1]],
                    [[0, 1], [1, 1], [1, 2], [2, 2]],
                    [[1, 1], [1, 2], [2, 0], [2, 1]],
                    [[0, 0], [1, 0], [1, 1], [2, 1]],
                ],
                dtype=np.int32,
            ),
            PieceType.T: np.array(
                [
                    [[0, 1], [1, 0], [1, 1], [1, 2]],
                    [[0, 1], [1, 1], [1, 2], [2, 1]],
                    [[1, 0], [1, 1], [1, 2], [2, 1]],
                    [[0, 1], [1, 0], [1, 1], [2, 1]],
                ],
                dtype=np.int32,
            ),
            PieceType.Z: np.array(
                [
                    [[0, 0], [0, 1], [1, 1], [1, 2]],
                    [[0, 2], [1, 1], [1, 2], [2, 1]],
                    [[1, 0], [1, 1], [2, 1], [2, 2]],
                    [[0, 1], [1, 0], [1, 1], [2, 0]],
                ],
                dtype=np.int32,
            ),
        }

        self.kicks = {
            # (kick_from, kick_to): (offset1, offset2, offset3, offset4)
            (0, 1): np.array(
                [[+0, -1], [-1, -1], [+2, +0], [+2, -1]], dtype=np.int32
            ),  # 0 -> R | CW
            (0, 3): np.array(
                [[+0, +1], [-1, +1], [+2, +0], [+2, +1]], dtype=np.int32
            ),  # 0 -> L | CCW
            (1, 0): np.array(
                [[+0, +1], [+1, +1], [-2, +0], [-2, +1]], dtype=np.int32
            ),  # R -> 0 | CCW
            (1, 2): np.array(
                [[+0, +1], [+1, +1], [-2, +0], [-2, +1]], dtype=np.int32
            ),  # R -> 2 | CW
            (2, 1): np.array(
                [[+0, -1], [-1, -1], [+2, +0], [+2, -1]], dtype=np.int32
            ),  # 2 -> R | CCW
            (2, 3): np.array(
                [[+0, +1], [-1, +1], [+2, +0], [+2, +1]], dtype=np.int32
            ),  # 2 -> L | CW
            (3, 0): np.array(
                [[+0, -1], [+1, -1], [-2, +0], [-2, -1]], dtype=np.int32
            ),  # L -> 0 | CW
            (3, 2): np.array(
                [[+0, -1], [+1, -1], [-2, +0], [-2, -1]], dtype=np.int32
            ),  # L -> 2 | CCW
            (0, 2): np.array(
                [[-1, +0], [-1, +1], [-1, -1], [+0, +1], [+0, -1]], dtype=np.int32
            ),  # 0 -> 2
            (1, 3): np.array(
                [[+0, +1], [-2, +1], [-1, +1], [-2, +0], [-1, +0]], dtype=np.int32
            ),  # R -> L
            (2, 0): np.array(
                [[+1, +0], [+1, -1], [+1, +1], [+0, -1], [+0, +1]], dtype=np.int32
            ),  # 2 -> 0
            (3, 1): np.array(
                [[+0, -1], [-2, -1], [-1, -1], [-2, +0], [-1, +0]], dtype=np.int32
            ),  # L -> R
        }

        self.i_kicks = {
            # (kick_from, kick_to): (offset1, offset2, offset3, offset4)
            (0, 1): np.array(
                [[+0, +1], [+0, -2], [+1, -2], [-2, +1]], dtype=np.int32
            ),  # 0 -> R | CW
            (0, 3): np.array(
                [[+0, -1], [+0, +2], [+1, +2], [-2, -1]], dtype=np.int32
            ),  # 0 -> L | CCW
            (1, 0): np.array(
                [[+0, -1], [+0, +2], [+2, -1], [-1, +2]], dtype=np.int32
            ),  # R -> 0 | CCW
            (1, 2): np.array(
                [[+0, -1], [+0, +2], [-2, -1], [+1, +2]], dtype=np.int32
            ),  # R -> 2 | CW
            (2, 1): np.array(
                [[+0, -2], [+0, +1], [-1, -2], [+2, +1]], dtype=np.int32
            ),  # 2 -> R | CCW
            (2, 3): np.array(
                [[+0, +2], [+0, -1], [-1, +2], [+2, -1]], dtype=np.int32
            ),  # 2 -> L | CW
            (3, 0): np.array(
                [[+0, +1], [+0, -2], [+2, +1], [-1, -2]], dtype=np.int32
            ),  # L -> 0 | CW
            (3, 2): np.array(
                [[+0, +1], [+0, -2], [-2, +1], [+1, -2]], dtype=np.int32
            ),  # L -> 2 | CCW
            (0, 2): np.array(
                [[-1, +0], [-1, +1], [-1, -1], [+0, +1], [+0, -1]], dtype=np.int32
            ),  # 0 -> 2
            (1, 3): np.array(
                [[+0, +1], [-2, +1], [-1, +1], [-2, +0], [-1, +0]], dtype=np.int32
            ),  # R -> L
            (2, 0): np.array(
                [[+1, +0], [+1, -1], [+1, +1], [+0, -1], [+0, +1]], dtype=np.int32
            ),  # 2 -> 0
            (3, 1): np.array(
                [[+0, -1], [-2, -1], [-1, -1], [-2, +0], [-1, +0]], dtype=np.int32
            ),  # L -> R
        }

    def kick_piece(
        self,
        kick_table: Dict[Tuple[int, int], np.ndarray],
        piece: Piece,
        cells: np.ndarray,
        new_r: int,
        delta_r: int,
        board: np.ndarray,
    ) -> Tuple[bool, np.ndarray, np.ndarray, int, int, np.ndarray]:
        # Do I ever need to check this?
        if (piece.r, new_r) not in kick_table.keys():
            # Rotation not possible, raise error
            raise ValueError(f"Rotation not possible from {piece.r} to {new_r}")

        kicked = False
        for delta_loc in kick_table[(piece.r, new_r)]:
            # Check each kick and perform the first valid
            if not overlaps(cells=cells, loc=piece.loc + delta_loc, board=board):
                # Kick doesn't overlap, so apply it and break
                piece.r = new_r
                piece.delta_r = delta_r

                piece.loc += delta_loc
                piece.delta_loc = delta_loc

                piece.cells = cells
                kicked = True
                break

        if not kicked:
            piece.delta_loc = np.zeros((2,), dtype=np.int32)
            piece.delta_r = 0

        return kicked, piece.cells, piece.loc, piece.r, piece.delta_r, piece.delta_loc
