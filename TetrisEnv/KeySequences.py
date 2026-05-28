from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .Moves import Keys
from .Pieces import Piece, PieceType
from .Placements import Placement, PlacementFinder
from .RotationSystem import RotationSystem
from .helpers import overlaps


@dataclass(frozen=True)
class Goal:
    """Desired final placement (rotation index, resting row, column)."""

    rotation: int
    row: int
    col: int


@dataclass(frozen=True)
class State:
    """Piece configuration used during BFS while the piece is floating."""

    row: int
    col: int
    rotation: int


class KeySequenceFinder:
    """Compute minimal key sequences that reach specified resting placements."""

    _ALLOWED_KEYS: Tuple[int, ...] = (
        Keys.TAP_LEFT,
        Keys.TAP_RIGHT,
        Keys.DAS_LEFT,
        Keys.DAS_RIGHT,
        Keys.CLOCKWISE,
        Keys.ANTICLOCKWISE,
        Keys.ROTATE_180,
        Keys.SOFT_DROP,
    )

    def __init__(self, rotation_system: RotationSystem | None = None) -> None:
        self._rotation_system = rotation_system or RotationSystem()
        self._placement_finder = PlacementFinder(self._rotation_system)

    def find_sequences(
        self,
        board: np.ndarray,
        piece: Piece,
        goals: Sequence[Goal],
        max_len: int,
        is_hold: bool,
    ) -> List[Optional[List[int]]]:
        self._validate_board(board)
        if not goals:
            return []

        rotation_system = self._rotation_system
        start_piece = self._copy_piece(piece)
        start_piece.cells = np.array(
            rotation_system.orientations[piece.piece_type][piece.r],
            dtype=np.int32,
            copy=True,
        )

        if overlaps(start_piece.cells, start_piece.loc, board):
            return [None for _ in goals]

        goal_map = self._normalize_goals(goals)
        results: List[Optional[List[int]]] = [None] * len(goals)

        queue: deque[State] = deque()
        start_state = self._state_from_piece(start_piece)
        queue.append(start_state)
        parents: MutableMapping[State, Tuple[Optional[State], Optional[int]]] = {
            start_state: (None, None)
        }
        depths: Dict[State, int] = {start_state: 0}
        max_path_length = max_len - 2

        remaining = set(goal_map.keys())
        while queue and remaining:
            current_state = queue.popleft()
            current_piece = self._piece_from_state(
                current_state, start_piece.piece_type, rotation_system
            )
            current_depth = depths[current_state]

            drop_row = self._hard_drop_row(current_piece, board)
            goal_key = (current_state.rotation % 4, drop_row, current_state.col)
            if goal_key in remaining:
                sequence = (
                    [Keys.START]
                    + ([Keys.HOLD] if is_hold else [])
                    + self._reconstruct_sequence(current_state, parents)
                    + [Keys.HARD_DROP]
                )
                if len(sequence) <= max_len:
                    results[goal_map[goal_key]] = sequence
                    remaining.remove(goal_key)
                    if not remaining:
                        break

            if current_depth >= max_path_length:
                continue

            for key in self._ALLOWED_KEYS:
                next_piece = self._apply_key(current_piece, key, board, rotation_system)
                if next_piece is None:
                    continue
                next_state = self._state_from_piece(next_piece)
                if next_state in parents:
                    continue
                next_depth = current_depth + 1
                if next_depth > max_path_length:
                    continue
                parents[next_state] = (current_state, key)
                depths[next_state] = next_depth
                queue.append(next_state)

        return results

    def find_all(
        self,
        board: np.ndarray,
        piece: Piece,
        max_len: int,
        is_hold: bool,
        return_timing: bool = False,
    ) -> (
        Tuple[np.ndarray, List[Placement], Dict[str, float]]
        | Tuple[np.ndarray, List[Placement]]
    ):
        t0 = time.perf_counter()
        placements = self._placement_finder.find(board, piece.piece_type)
        t1 = time.perf_counter()

        goals = [
            Goal(rotation=placement.rotation, row=placement.row, col=placement.col)
            for placement in placements
        ]
        raw_sequences = self.find_sequences(
            board, piece, goals, max_len=max_len, is_hold=is_hold
        )
        t2 = time.perf_counter()

        board_rows, board_cols = board.shape
        visible_rows = 20
        rotations = 4

        if max_len < 2:
            raise ValueError(
                "max_len must be at least 2 to include START and HARD_DROP keys."
            )
        if board_cols != 10:
            raise ValueError("Board must have exactly 10 columns.")
        if board_rows < visible_rows:
            raise ValueError("Board must have at least 20 rows.")

        visible_start = board_rows - visible_rows
        total_positions = rotations * visible_rows * board_cols
        sequences_array = np.full((total_positions, max_len), Keys.PAD, dtype=np.int32)

        def placement_index(rotation: int, row: int, display_col: int) -> Optional[int]:
            rotation_idx = rotation % rotations
            if display_col < 0 or display_col >= board_cols:
                return None
            row_idx = row - visible_start
            if row_idx < 0 or row_idx >= visible_rows:
                return None
            return (
                rotation_idx * visible_rows * board_cols
                + row_idx * board_cols
                + display_col
            )

        for placement, sequence in zip(placements, raw_sequences):
            display_col = int(placement.cells[:, 1].min())
            idx = placement_index(placement.rotation, placement.row, display_col)
            if idx is None or sequence is None:
                continue

            seq_len = len(sequence)
            if seq_len > max_len:
                continue

            padded = np.full(max_len, Keys.PAD, dtype=np.int32)
            padded[:seq_len] = sequence
            sequences_array[idx] = padded

        if return_timing:
            timing = {
                "placements": t1 - t0,
                "paths": t2 - t1,
                "total": t2 - t0,
            }
            return sequences_array, timing

        return sequences_array

    @staticmethod
    def _validate_board(board: np.ndarray) -> None:
        if board.ndim != 2:
            raise ValueError("Board must be a 2-D array.")

    @staticmethod
    def _copy_piece(piece: Piece) -> Piece:
        return Piece(
            piece_type=piece.piece_type,
            loc=np.array(piece.loc, dtype=np.int32, copy=True),
            r=int(piece.r),
            cells=np.array(piece.cells, dtype=np.int32, copy=True),
        )

    @staticmethod
    def _apply_rotation(
        piece: Piece,
        delta_r: int,
        board: np.ndarray,
        rotation_system: RotationSystem,
    ) -> Optional[Piece]:
        new_r = (piece.r + delta_r) % 4
        cells = rotation_system.orientations[piece.piece_type][new_r]
        loc = piece.loc

        rotated = KeySequenceFinder._copy_piece(piece)
        if not overlaps(cells=cells, loc=loc, board=board):
            rotated.r = new_r
            rotated.cells = cells
            return rotated

        kick_table = (
            rotation_system.i_kicks
            if piece.piece_type == PieceType.I
            else rotation_system.kicks
        )
        kicked_piece = KeySequenceFinder._copy_piece(piece)
        kicked, cells, loc, r, _, _ = rotation_system.kick_piece(
            kick_table=kick_table,
            piece=kicked_piece,
            cells=cells,
            new_r=new_r,
            delta_r=delta_r,
            board=board,
        )
        if not kicked:
            return None

        kicked_piece.cells = cells
        kicked_piece.loc = loc
        kicked_piece.r = r
        return kicked_piece

    @staticmethod
    def _apply_translation(
        piece: Piece,
        delta_col: int,
        board: np.ndarray,
    ) -> Optional[Piece]:
        if delta_col == 0:
            return KeySequenceFinder._copy_piece(piece)

        direction = -1 if delta_col < 0 else 1
        limit = abs(delta_col)
        last_valid_loc = None
        for step in range(1, limit + 1):
            try_loc = piece.loc + np.array([0, direction * step], dtype=np.int32)
            if overlaps(piece.cells, try_loc, board):
                break
            last_valid_loc = try_loc

        if last_valid_loc is None:
            return None

        shifted = KeySequenceFinder._copy_piece(piece)
        shifted.loc = last_valid_loc
        return shifted

    @classmethod
    def _apply_key(
        cls,
        piece: Piece,
        key: int,
        board: np.ndarray,
        rotation_system: RotationSystem,
    ) -> Optional[Piece]:
        if key == Keys.CLOCKWISE:
            return cls._apply_rotation(piece, +1, board, rotation_system)
        if key == Keys.ANTICLOCKWISE:
            return cls._apply_rotation(piece, -1, board, rotation_system)
        if key == Keys.ROTATE_180:
            return cls._apply_rotation(piece, +2, board, rotation_system)
        if key == Keys.TAP_LEFT:
            return cls._apply_translation(piece, -1, board)
        if key == Keys.TAP_RIGHT:
            return cls._apply_translation(piece, +1, board)
        if key == Keys.DAS_LEFT:
            return cls._apply_translation(piece, -100, board)
        if key == Keys.DAS_RIGHT:
            return cls._apply_translation(piece, +100, board)
        if key == Keys.SOFT_DROP:
            dropped = cls._copy_piece(piece)
            loc = np.array(piece.loc, dtype=np.int32, copy=True)
            while True:
                next_loc = loc + np.array([1, 0], dtype=np.int32)
                if overlaps(dropped.cells, next_loc, board):
                    break
                loc = next_loc
            if np.array_equal(loc, dropped.loc):
                return None
            dropped.loc = loc
            return dropped
        if key == Keys.START:
            return cls._copy_piece(piece)
        if key == Keys.HARD_DROP:
            raise ValueError("Hard drop should not be applied during BFS exploration.")
        if key == Keys.HOLD:
            raise ValueError("Hold key is not permitted for active piece movement.")
        raise ValueError(f"Unsupported key: {key}")

    @staticmethod
    def _state_from_piece(piece: Piece) -> State:
        return State(int(piece.loc[0]), int(piece.loc[1]), int(piece.r) % 4)

    @staticmethod
    def _piece_from_state(
        state: State,
        piece_type: PieceType,
        rotation_system: RotationSystem,
    ) -> Piece:
        cells = rotation_system.orientations[piece_type][state.rotation]
        return Piece(
            piece_type=piece_type,
            loc=np.array([state.row, state.col], dtype=np.int32),
            r=state.rotation,
            cells=np.array(cells, dtype=np.int32, copy=True),
        )

    @staticmethod
    def _hard_drop_row(piece: Piece, board: np.ndarray) -> int:
        loc = np.array(piece.loc, dtype=np.int32, copy=True)
        while True:
            next_loc = loc + np.array([1, 0], dtype=np.int32)
            if overlaps(piece.cells, next_loc, board):
                break
            loc = next_loc
        return int(loc[0])

    @staticmethod
    def _reconstruct_sequence(
        end_state: State,
        parents: Mapping[State, Tuple[Optional[State], Optional[int]]],
    ) -> List[int]:
        keys: List[int] = []
        state = end_state
        while True:
            parent_state, key = parents[state]
            if parent_state is None or key is None:
                break
            keys.append(key)
            state = parent_state
        keys.reverse()
        return keys

    @staticmethod
    def _normalize_goals(goals: Sequence[Goal]) -> Dict[Tuple[int, int, int], int]:
        mapping: Dict[Tuple[int, int, int], int] = {}
        for idx, goal in enumerate(goals):
            key = (goal.rotation % 4, int(goal.row), int(goal.col))
            mapping[key] = idx
        return mapping
