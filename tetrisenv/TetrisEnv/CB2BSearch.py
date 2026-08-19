import ctypes
import numpy as np
import os
import glob
from typing import Tuple


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
                ctypes.c_int,  # max_roots
                ctypes.POINTER(ctypes.c_int),    # out_num_roots
                ctypes.POINTER(ctypes.c_int),    # out_root_action_indices
                ctypes.POINTER(ctypes.c_float),  # out_root_scores
                ctypes.POINTER(ctypes.c_int64),  # out_root_sequences
                ctypes.POINTER(ctypes.c_int),    # out_root_landing_rows
            ]
            self._lib.b2b_search_c.restype = None

            # --- single placement lock + score (MCTS placement step) ---
            self._lib.b2b_lock_score_c.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags="C_CONTIGUOUS"),
                ctypes.c_int,  # board_height
                ctypes.c_int,  # piece_type
                ctypes.c_int,  # rot
                ctypes.c_int,  # norm_col
                ctypes.c_int,  # landing_row
                ctypes.c_int,  # spin_type
                ctypes.c_int,  # b2b
                ctypes.c_int,  # combo
                ctypes.POINTER(ctypes.c_int),    # out_clears
                ctypes.POINTER(ctypes.c_float),  # out_attack
                ctypes.POINTER(ctypes.c_int),    # out_new_b2b
                ctypes.POINTER(ctypes.c_int),    # out_new_combo
            ]
            self._lib.b2b_lock_score_c.restype = None

        self._col_bits = (
            np.uint16(1) << np.arange(10, dtype=np.uint16)
        ).astype(np.uint16)

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
            0, None, None, None, None, None,  # no per-root output
        )

        return out_action.value, out_sequence

    def search_with_scores(
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
        max_roots: int = 512,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the search and also return per-root candidates.

        Returns (best_action_idx, best_sequence, cand_action_indices,
        cand_scores, cand_sequences, cand_landing_rows) where the candidate arrays
        cover every root placement that survived to the final beam. `cand_scores`
        are RAW search scores (no softmax). The value target is `cand_scores.max()`.
        `cand_landing_rows` is each placement's BFS lock row (0..board_height-1).
        `best_sequence` is the chosen move's key sequence (PAD-filled to max_len).
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

        out_action = ctypes.c_int(-1)
        out_sequence = np.full(max_len, 11, dtype=np.int64)

        num_roots = ctypes.c_int(0)
        root_actions = np.zeros(max_roots, dtype=np.int32)
        root_scores = np.zeros(max_roots, dtype=np.float32)
        root_sequences = np.full(max_roots * max_len, 11, dtype=np.int64)
        root_landing_rows = np.zeros(max_roots, dtype=np.int32)

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
            max_roots,
            ctypes.byref(num_roots),
            root_actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            root_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            root_sequences.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            root_landing_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )

        n = num_roots.value
        return (
            out_action.value,
            out_sequence,
            root_actions[:n],
            root_scores[:n],
            root_sequences[: n * max_len].reshape(n, max_len),
            root_landing_rows[:n],
        )

    def lock_score(
        self,
        board: np.ndarray,
        piece_type: int,
        rot: int,
        norm_col: int,
        landing_row: int,
        spin_type: int,
        b2b: int,
        combo: int,
    ) -> Tuple[np.ndarray, int, float, int, int]:
        """Lock one placement on `board` (24x10 occupancy, garbage as plain cells) and
        score it - the C core of `_lock_piece` + `Scorer.judge`. Returns (new_board,
        clears, attack, new_b2b, new_combo). Garbage / stats / reward stay in Python."""
        if not self._lib:
            raise RuntimeError("b2b_search C library not loaded")

        occupied = (board != 0).astype(np.uint16)
        mask_rows = np.ascontiguousarray(
            (occupied * self._col_bits).sum(axis=1, dtype=np.uint16)
        )
        board_height = board.shape[0]

        out_clears = ctypes.c_int(0)
        out_attack = ctypes.c_float(0.0)
        out_new_b2b = ctypes.c_int(0)
        out_new_combo = ctypes.c_int(0)

        self._lib.b2b_lock_score_c(
            mask_rows, board_height, piece_type, rot, norm_col, landing_row,
            spin_type, b2b, combo,
            ctypes.byref(out_clears), ctypes.byref(out_attack),
            ctypes.byref(out_new_b2b), ctypes.byref(out_new_combo),
        )

        new_board = ((mask_rows[:, None] & self._col_bits) > 0).astype(np.float32)
        return (
            new_board,
            out_clears.value,
            out_attack.value,
            out_new_b2b.value,
            out_new_combo.value,
        )
