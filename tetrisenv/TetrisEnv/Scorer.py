from enum import Enum
from .Pieces import Piece, PieceType
from .helpers import overlaps
import numpy as np
from typing import Tuple


class Spins(Enum):
    # TODO
    # More spins
    NO_SPIN = 0
    T_SPIN_MINI = 1
    T_SPIN = 2
    ALL_MINI = 3


class Scorer:
    def __init__(self) -> None:
        self._b2b = -1
        self._combo = -1

    def reset(self) -> None:
        self._b2b = -1
        self._combo = -1

    def judge(self, piece: Piece, board: np.ndarray, post_lock_board: np.ndarray, clears: int) -> Tuple[float, bool]:
        attack = 0
        spin = Spins.NO_SPIN
        surge = 0
        new_b2b = self._b2b
        new_combo = self._combo

        if piece.delta_r != 0:
            if piece.piece_type == PieceType.T:
                # Check for corners clockwise from top-left
                # Corners are filled if cell is nonzero or out of board
                corner_cells = np.array(
                    [[0, 0], [0, 2], [2, 2], [2, 0]], dtype=np.int32
                )

                # Check out of bounds corners
                corner_inds = piece.loc + corner_cells  # 4, 2
                corners = np.any(
                    [
                        np.any(corner_inds >= board.shape, axis=-1),
                        np.any(corner_inds < 0, axis=-1),
                    ],
                    axis=0,
                )

                # Check in bounds corners
                corner_inds = np.maximum(
                    0, np.minimum(corner_inds, [board.shape[0] - 1, board.shape[1] - 1])
                )
                corners = np.any(
                    [corners, board[corner_inds[:, 0], corner_inds[:, 1]] != 0], axis=0
                )

                # Find back cell in same order as corners. Corners[back] is
                # is the corner anticlockwise relative to the cell
                back = None
                for i, cell in enumerate(
                    np.array([[0, 1], [1, 2], [2, 1], [1, 0]], np.int32)
                ):
                    if not np.any(np.all(piece.cells == cell, axis=-1)):
                        back = i
                        break

                front_corners = np.sum(
                    [corners[(back + 2) % 4], corners[(back + 3) % 4]]
                )
                back_corners = np.sum(
                    [corners[(back + 0) % 4], corners[(back + 1) % 4]]
                )

                if front_corners == 2 and back_corners >= 1:
                    # Proper T spin
                    spin = Spins.T_SPIN

                elif front_corners == 1 and back_corners == 2:
                    if np.sum(np.abs(piece.delta_loc)) > 2:
                        # Piece was kicked far enough, so this is also a T spin
                        spin = Spins.T_SPIN
                    else:
                        # Not kicked far, T spin mini
                        spin = Spins.T_SPIN_MINI
                else:
                    # Fails to pass corner rules, not a T spin
                    spin = Spins.NO_SPIN
            else:
                for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if not overlaps(
                        cells=piece.cells, loc=piece.loc + direction, board=board
                    ):
                        spin = Spins.NO_SPIN
                        break
                else:
                    spin = Spins.ALL_MINI

        perfect_clear = np.all(post_lock_board == 0)

        if clears:
            if spin != Spins.NO_SPIN or clears == 4 or perfect_clear:
                new_b2b += 1
            else:
                # Compute surge
                if self._b2b >= 4:
                    surge = new_b2b

                new_b2b = -1

            new_combo += 1

            # Compute base attack
            if perfect_clear:
                attack += [0, 5, 6, 7, 9][clears]

            elif spin == Spins.T_SPIN:
                attack += [0, 2, 4, 6, 0][clears]

            elif spin == Spins.T_SPIN_MINI:
                attack += [0, 0, 1, 2, 0][clears]

            else:
                attack += [0, 0, 1, 2, 4][clears]

            # Compute b2b and combo bonuses
            if self._b2b > -1:
                attack += 1

            if self._combo > 0:
                if attack > 0:
                    attack = np.floor(attack * (1 + 0.25 * self._combo))
                else:
                    attack = np.floor(np.log(1 + 1.25 * self._combo))

            # Send surge
            attack += surge

        else:
            new_combo = -1

        self._b2b = new_b2b
        self._combo = new_combo

        return attack, spin != Spins.NO_SPIN
