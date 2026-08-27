from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from .Pieces import PieceType
from .RotationSystem import RotationSystem


@dataclass(frozen=True)
class Placement:
    rotation: int
    row: int
    col: int
    cells: np.ndarray

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.rotation, self.row, self.col)


class PlacementFinder:
    def __init__(self, rotation_system: RotationSystem | None = None) -> None:
        self._rotation_system = rotation_system or RotationSystem()

    def find(self, board: np.ndarray, piece_type: PieceType) -> List[Placement]:
        occupied = self._coerce_board(board)
        reachable = self._reachable_from_top(occupied)

        placements: List[Placement] = []
        for rotation_idx, cells in self._unique_orientations(piece_type):
            row_range, col_range = self._anchor_ranges(cells, occupied.shape)
            if row_range is None or col_range is None:
                continue

            rel_cols, bottom_rows = self._bottom_profile(cells)

            for anchor_col in col_range:
                for anchor_row in row_range:
                    abs_rows = cells[:, 0] + anchor_row
                    abs_cols = cells[:, 1] + anchor_col

                    if np.any(occupied[abs_rows, abs_cols]):
                        continue

                    if not self._has_support(
                        anchor_row, anchor_col, rel_cols, bottom_rows, occupied
                    ):
                        continue

                    if not np.all(reachable[abs_rows, abs_cols]):
                        continue

                    placements.append(
                        Placement(
                            rotation=rotation_idx,
                            row=anchor_row,
                            col=anchor_col,
                            cells=np.ascontiguousarray(
                                np.stack([abs_rows, abs_cols], axis=1), dtype=np.int32
                            ),
                        )
                    )

        return placements

    def _unique_orientations(
        self, piece_type: PieceType
    ) -> Iterator[Tuple[int, np.ndarray]]:
        orientations = self._rotation_system.orientations[piece_type]

        seen: set[bytes] = set()
        for idx, cells in enumerate(orientations):
            key = cells.tobytes()
            if key in seen:
                continue
            seen.add(key)
            yield idx, np.array(cells, dtype=np.int32, copy=True)

    @staticmethod
    def _coerce_board(board: np.ndarray) -> np.ndarray:
        if board.ndim != 2:
            raise ValueError("Board must be a 2-D array.")

        occupied = board != 0
        occupied.setflags(write=False)
        return occupied

    @staticmethod
    def _anchor_ranges(
        cells: np.ndarray, board_shape: Tuple[int, int]
    ) -> Tuple[range | None, range | None]:
        min_row = int(cells[:, 0].min())
        max_row = int(cells[:, 0].max())
        min_col = int(cells[:, 1].min())
        max_col = int(cells[:, 1].max())

        rows, cols = board_shape
        row_start = -min_row
        row_end = rows - 1 - max_row
        col_start = -min_col
        col_end = cols - 1 - max_col

        if row_start > row_end or col_start > col_end:
            return None, None

        return range(row_start, row_end + 1), range(col_start, col_end + 1)

    @staticmethod
    def _reachable_from_top(occupied: np.ndarray) -> np.ndarray:
        rows, cols = occupied.shape
        reachable = np.zeros_like(occupied, dtype=bool)
        queue: deque[Tuple[int, int]] = deque()

        top_open = np.flatnonzero(~occupied[0])
        for col in top_open:
            reachable[0, col] = True
            queue.append((0, col))

        directions = ((1, 0), (-1, 0), (0, -1), (0, 1))
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if occupied[nr, nc] or reachable[nr, nc]:
                    continue
                reachable[nr, nc] = True
                queue.append((nr, nc))

        return reachable

    @staticmethod
    def _bottom_profile(cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rel_cols = cells[:, 1]
        unique_cols = np.unique(rel_cols)
        bottom_rows = np.empty_like(unique_cols)
        for idx, col in enumerate(unique_cols):
            bottom_rows[idx] = cells[rel_cols == col][:, 0].max()
        return unique_cols, bottom_rows

    @staticmethod
    def _has_support(
        anchor_row: int,
        anchor_col: int,
        rel_cols: Sequence[int],
        bottom_rows: Sequence[int],
        occupied: np.ndarray,
    ) -> bool:
        rows = occupied.shape[0]
        for rel_col, bottom_row in zip(rel_cols, bottom_rows):
            r = anchor_row + bottom_row
            c = anchor_col + rel_col
            if r == rows - 1 or occupied[r + 1, c]:
                return True
        return False
