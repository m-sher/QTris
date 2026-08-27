import ctypes
import numpy as np
import os
import glob
import time
from typing import Tuple, Dict, Optional

from .KeySequences import KeySequenceFinder
from .Pieces import Piece
from .Moves import Keys

class CKeySequenceFinder(KeySequenceFinder):
    def __init__(self, rotation_system=None, num_row_tiers: int = 1):
        super().__init__(rotation_system)
        self._num_row_tiers = num_row_tiers
        
        # Load Library
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Search for pathfinder*.so (handling various platform suffixes)
        candidates = glob.glob(os.path.join(curr_dir, "pathfinder*.so")) + \
                     glob.glob(os.path.join(curr_dir, "..", "pathfinder*.so")) + \
                     glob.glob(os.path.join(curr_dir, "pathfinder.so")) # Explicit fallback
                     
        if not candidates:
             lib_path = os.path.join(curr_dir, "pathfinder.so")
        else:
            lib_path = candidates[0]
            
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            # We allow instantiation but methods will fail.
            print(f"Warning: Could not load C pathfinding library at {lib_path}: {e}")
            self._lib = None
            
        if self._lib:
            self._lib.find_sequences_c.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS')
            ]
            self._lib.find_sequences_c.restype = None

            self._lib.find_placement_candidates_c.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'),
                ctypes.c_int,  # board_height
                ctypes.c_int,  # piece_type
                ctypes.c_int,  # start_row
                ctypes.c_int,  # start_col
                ctypes.c_int,  # start_rot
                ctypes.c_int,  # max_len
                ctypes.c_int,  # is_hold
                np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
            ]
            self._lib.find_placement_candidates_c.restype = None

            self._lib.find_unique_placements_c.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags="C_CONTIGUOUS"),
                ctypes.c_int,  # board_height
                ctypes.c_int,  # piece_type
                ctypes.c_int,  # start_row
                ctypes.c_int,  # start_col
                ctypes.c_int,  # start_rot
                ctypes.c_int,  # max_len
                ctypes.c_int,  # is_hold
                ctypes.c_int,  # max_out
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
                ctypes.c_void_p,  # out_sequences nullable
            ]
            self._lib.find_unique_placements_c.restype = ctypes.c_int

        self._col_bits = (
            np.uint16(1) << np.arange(10, dtype=np.uint16)
        ).astype(np.uint16)

    def find_all(
        self,
        board: np.ndarray,
        piece: Piece,
        max_len: int,
        is_hold: bool,
        return_timing: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
        if not self._lib:
            raise RuntimeError("C Library not loaded")

        t_start = time.perf_counter()
        
        # Prepare Board
        # Pack each row into a uint16 column bitmask
        occupied = (board != 0).astype(np.uint16)
        mask_rows = (occupied * self._col_bits).sum(axis=1, dtype=np.uint16)
        if not mask_rows.flags['C_CONTIGUOUS']:
            mask_rows = np.ascontiguousarray(mask_rows)
            
        board_height = board.shape[0]
        
        total_positions = 80 * self._num_row_tiers
        output_buffer = np.full(total_positions * max_len, Keys.PAD, dtype=np.int64)
        
        self._lib.find_sequences_c(
            mask_rows,
            board_height,
            piece.piece_type.value,
            int(piece.loc[0]),
            int(piece.loc[1]),
            int(piece.r),
            max_len,
            int(is_hold),
            self._num_row_tiers,
            output_buffer
        )
        
        sequences_array = output_buffer.reshape((total_positions, max_len))
        
        if return_timing:
             duration = time.perf_counter() - t_start
             return sequences_array, {"placements": 0.0, "paths": duration, "total": duration}

        return sequences_array

    def find_placement_candidates(
        self, board: np.ndarray, piece: Piece, max_len: int, is_hold: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dense per-branch placement candidates: every reachable resting placement keyed by
        rot*40 + norm_col*4 + spin_type. Returns (sequences[160, max_len] PAD-filled,
        landing_rows[160], -1 = empty slot). No death-pruning, so any spawnable board yields >=1."""
        if not self._lib:
            raise RuntimeError("C Library not loaded")

        occupied = (board != 0).astype(np.uint16)
        mask_rows = (occupied * self._col_bits).sum(axis=1, dtype=np.uint16)
        if not mask_rows.flags["C_CONTIGUOUS"]:
            mask_rows = np.ascontiguousarray(mask_rows)
        board_height = board.shape[0]

        sequences = np.full(160 * max_len, Keys.PAD, dtype=np.int64)
        landing_rows = np.full(160, -1, dtype=np.int32)

        self._lib.find_placement_candidates_c(
            mask_rows,
            board_height,
            piece.piece_type.value,
            int(piece.loc[0]),
            int(piece.loc[1]),
            int(piece.r),
            max_len,
            int(is_hold),
            sequences,
            landing_rows,
        )
        return sequences.reshape((160, max_len)), landing_rows

    def find_unique_placements(
        self,
        board: np.ndarray,
        piece: Piece,
        max_len: int,
        is_hold: bool,
        max_out: int = 512,
        with_sequences: bool = False,
    ):
        """Every unique resting placement: (rot, norm_col, landing_row, spin)[, sequences].

        Distinct landing rows are separate entries; BFS rules match
        find_placement_candidates.
        """
        if not self._lib:
            raise RuntimeError("C Library not loaded")

        occupied = (board != 0).astype(np.uint16)
        mask_rows = (occupied * self._col_bits).sum(axis=1, dtype=np.uint16)
        if not mask_rows.flags["C_CONTIGUOUS"]:
            mask_rows = np.ascontiguousarray(mask_rows)
        board_height = board.shape[0]

        out_rot = np.zeros(max_out, dtype=np.int32)
        out_ncol = np.zeros(max_out, dtype=np.int32)
        out_lr = np.zeros(max_out, dtype=np.int32)
        out_spin = np.zeros(max_out, dtype=np.int32)
        seqs = np.full(max_out * max_len, Keys.PAD, dtype=np.int64) if with_sequences else None
        seq_arg = (
            seqs.ctypes.data_as(ctypes.c_void_p) if with_sequences else ctypes.c_void_p(0)
        )

        n = int(
            self._lib.find_unique_placements_c(
                mask_rows,
                board_height,
                piece.piece_type.value,
                int(piece.loc[0]),
                int(piece.loc[1]),
                int(piece.r),
                max_len,
                int(is_hold),
                max_out,
                out_rot,
                out_ncol,
                out_lr,
                out_spin,
                seq_arg,
            )
        )
        if with_sequences:
            return (
                out_rot[:n].copy(),
                out_ncol[:n].copy(),
                out_lr[:n].copy(),
                out_spin[:n].copy(),
                seqs.reshape(max_out, max_len)[:n].copy(),
            )
        return (
            out_rot[:n].copy(),
            out_ncol[:n].copy(),
            out_lr[:n].copy(),
            out_spin[:n].copy(),
        )
