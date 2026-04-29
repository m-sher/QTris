import ctypes
import numpy as np
import os
import glob
from typing import Tuple, List


class GameConfig(ctypes.Structure):
    _fields_ = [
        ('seed', ctypes.c_int),
        ('garbage_chance', ctypes.c_float),
        ('garbage_min', ctypes.c_int),
        ('garbage_max', ctypes.c_int),
        ('garbage_push_delay', ctypes.c_int),
    ]


class GameResult(ctypes.Structure):
    _fields_ = [
        ('steps_completed', ctypes.c_int),
        ('survived', ctypes.c_int),
        ('total_attack', ctypes.c_float),
        ('max_b2b', ctypes.c_int),
        ('end_height', ctypes.c_int),
        ('avg_height', ctypes.c_float),
        ('max_height', ctypes.c_int),
    ]


class CB2BSearch:
    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))

        candidates = (
            glob.glob(os.path.join(curr_dir, "b2b_search*.so"))
            + glob.glob(os.path.join(curr_dir, "..", "b2b_search*.so"))
            + glob.glob(os.path.join(curr_dir, "b2b_search.so"))
        )

        if not candidates:
            lib_path = os.path.join(curr_dir, "b2b_search.so")
        else:
            lib_path = candidates[0]

        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            print(f"Warning: Could not load b2b_search library at {lib_path}: {e}")
            self._lib = None

        if self._lib:
            self._lib.b2b_search_c.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags="C_CONTIGUOUS"),
                ctypes.c_int,  # board_height
                ctypes.c_int,  # active_piece
                ctypes.c_int,  # hold_piece
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
                ctypes.c_int,  # queue_len
                ctypes.c_int,  # b2b
                ctypes.c_int,  # combo
                ctypes.c_int,  # total_garbage
                ctypes.c_int,  # garbage_push_delay
                ctypes.c_int,  # bag_seen_init
                ctypes.c_int,  # search_depth
                ctypes.c_int,  # beam_width
                ctypes.c_int,  # max_len
                ctypes.POINTER(ctypes.c_int),  # out_action_index
                np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
            ]
            self._lib.b2b_search_c.restype = None

            # --- decompose function ---
            self._lib.b2b_decompose_c.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags="C_CONTIGUOUS"),
                ctypes.c_int,  # board_height
                ctypes.c_int,  # active_piece
                ctypes.c_int,  # hold_piece
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
                ctypes.c_int,  # queue_len
                ctypes.c_int,  # b2b
                ctypes.c_int,  # combo
                ctypes.c_int,  # total_garbage
                ctypes.c_int,  # garbage_push_delay
                np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                ctypes.c_int,  # max_placements
            ]
            self._lib.b2b_decompose_c.restype = ctypes.c_int

            self._lib.b2b_get_num_decompose.argtypes = []
            self._lib.b2b_get_num_decompose.restype = ctypes.c_int

            # --- game-loop function ---
            self._lib.b2b_run_eval_games.argtypes = [
                ctypes.c_int,   # num_games
                ctypes.c_void_p,  # configs (GameConfig*)
                ctypes.c_int,   # num_steps
                ctypes.c_int,   # search_depth
                ctypes.c_int,   # beam_width
                ctypes.c_int,   # queue_size
                ctypes.c_void_p,  # results (GameResult*)
            ]
            self._lib.b2b_run_eval_games.restype = None

        self._col_bits = (
            np.uint16(1) << np.arange(10, dtype=np.uint16)
        ).astype(np.uint16)

        self.NUM_DECOMPOSE = 21
        self.COMPONENT_NAMES = [
            "height", "near_death", "bumpiness",
            "holes", "wasted_holes", "hole_ceiling", "hole_forgive", "well",
            "b2b_flat", "b2b_sqrt", "combo", "downstack",
            "tslot", "immobile_clear",
            "max_single", "attack",
            "cascade", "surge_pot", "app", "break_ready", "b2b_linear",
        ]

    def search(
        self,
        board: np.ndarray,
        active_piece: int,
        hold_piece: int,
        queue: np.ndarray,
        b2b: int,
        combo: int,
        total_garbage: int,
        garbage_push_delay: int = 1,
        bag_seen: int = 0,
        search_depth: int = 7,
        beam_width: int = 128,
        max_len: int = 15,
    ) -> Tuple[int, np.ndarray]:
        if not self._lib:
            raise RuntimeError("b2b_search C library not loaded")

        # Convert board to uint16 bitmasks
        occupied = (board != 0).astype(np.uint16)
        mask_rows = (occupied * self._col_bits).sum(axis=1, dtype=np.uint16)
        if not mask_rows.flags["C_CONTIGUOUS"]:
            mask_rows = np.ascontiguousarray(mask_rows)

        board_height = board.shape[0]

        # Prepare queue as int32 array
        queue_arr = np.asarray(queue, dtype=np.int32)
        if not queue_arr.flags["C_CONTIGUOUS"]:
            queue_arr = np.ascontiguousarray(queue_arr)

        # Output buffers
        out_action = ctypes.c_int(-1)
        out_sequence = np.full(max_len, 11, dtype=np.int64)  # PAD = 11

        self._lib.b2b_search_c(
            mask_rows,
            board_height,
            active_piece,
            hold_piece,
            queue_arr,
            len(queue_arr),
            b2b,
            combo,
            total_garbage,
            garbage_push_delay,
            bag_seen,
            search_depth,
            beam_width,
            max_len,
            ctypes.byref(out_action),
            out_sequence,
        )

        return out_action.value, out_sequence

    def decompose(
        self,
        board: np.ndarray,
        active_piece: int,
        hold_piece: int,
        queue: np.ndarray,
        b2b: int,
        combo: int,
        total_garbage: int,
        garbage_push_delay: int = 1,
        max_placements: int = 512,
    ) -> np.ndarray:
        """Decompose depth-0 scores into per-component terms.

        Returns (num_placements, NUM_DECOMPOSE) float32 array.
        Each row is one placement, each column is a heuristic term.
        """
        if not self._lib:
            raise RuntimeError("b2b_search C library not loaded")

        occupied = (board != 0).astype(np.uint16)
        mask_rows = (occupied * self._col_bits).sum(axis=1, dtype=np.uint16)
        if not mask_rows.flags["C_CONTIGUOUS"]:
            mask_rows = np.ascontiguousarray(mask_rows)

        board_height = board.shape[0]
        queue_arr = np.asarray(queue, dtype=np.int32)
        if not queue_arr.flags["C_CONTIGUOUS"]:
            queue_arr = np.ascontiguousarray(queue_arr)

        buf = np.zeros(max_placements * self.NUM_DECOMPOSE, dtype=np.float32)

        n = self._lib.b2b_decompose_c(
            mask_rows, board_height,
            active_piece, hold_piece,
            queue_arr, len(queue_arr),
            b2b, combo, total_garbage,
            garbage_push_delay,
            buf, max_placements,
        )

        return buf[: n * self.NUM_DECOMPOSE].reshape(n, self.NUM_DECOMPOSE)

    def run_eval_games(
        self,
        configs: List[GameConfig],
        num_steps: int = 64,
        search_depth: int = 7,
        beam_width: int = 128,
        queue_size: int = 5,
    ) -> List[GameResult]:
        """Run multiple evaluation games entirely in C.

        Each GameConfig specifies seed + garbage parameters.
        Returns a list of GameResult structs with stats.
        """
        if not self._lib:
            raise RuntimeError("b2b_search C library not loaded")

        n = len(configs)
        ConfigArray = GameConfig * n
        ResultArray = GameResult * n
        cfgs = ConfigArray(*configs)
        results = ResultArray()

        self._lib.b2b_run_eval_games(
            n,
            ctypes.cast(cfgs, ctypes.c_void_p),
            num_steps,
            search_depth,
            beam_width,
            queue_size,
            ctypes.cast(results, ctypes.c_void_p),
        )

        return list(results)
