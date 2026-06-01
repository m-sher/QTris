import ctypes
import numpy as np
import os
import glob
from typing import Optional

class CHoleFinder:
    def __init__(self):
        # Load Library
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Search for hole_finder*.so
        candidates = glob.glob(os.path.join(curr_dir, "hole_finder*.so")) + \
                     glob.glob(os.path.join(curr_dir, "..", "hole_finder*.so")) + \
                     glob.glob(os.path.join(curr_dir, "hole_finder.so"))
                     
        if not candidates:
             lib_path = os.path.join(curr_dir, "hole_finder.so")
        else:
            lib_path = candidates[0]
            
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            print(f"Warning: Could not load C hole_finder library at {lib_path}: {e}")
            self._lib = None
            
        if self._lib:
            # int count_enclosed_holes(const uint16_t* board_rows, int board_height)
            self._lib.count_enclosed_holes.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'),
                ctypes.c_int
            ]
            self._lib.count_enclosed_holes.restype = ctypes.c_int
        
        self._col_bits = (
            np.uint16(1) << np.arange(10, dtype=np.uint16)
        ).astype(np.uint16)

    def count_holes(self, board: np.ndarray) -> int:
        if not self._lib:
            return 0
            
        # Convert to bitmasks
        # Board is expected to be float or int where !=0 is occupied.
        occupied = (board != 0).astype(np.uint16)
        
        # Calculate row masks
        mask_rows = (occupied * self._col_bits).sum(axis=1, dtype=np.uint16)
        
        if not mask_rows.flags['C_CONTIGUOUS']:
            mask_rows = np.ascontiguousarray(mask_rows)
            
        board_height = board.shape[0]
        
        return self._lib.count_enclosed_holes(mask_rows, board_height)
