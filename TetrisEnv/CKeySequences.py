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
        # Also check parent dir in case of build layout quirks
        candidates = glob.glob(os.path.join(curr_dir, "pathfinder*.so")) + \
                     glob.glob(os.path.join(curr_dir, "..", "pathfinder*.so")) + \
                     glob.glob(os.path.join(curr_dir, "pathfinder.so")) # Explicit fallback
                     
        if not candidates:
             # If running from source and not yet built, this might fail.
             # But usually setup.py build_ext --inplace puts it in source tree.
             lib_path = os.path.join(curr_dir, "pathfinder.so")
        else:
            lib_path = candidates[0]
            
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            # If checking during build process or CI, this might happen.
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
        # Convert to bitmasks (same logic as BitboardKeySequenceFinder)
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
