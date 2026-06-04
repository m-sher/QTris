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
        ('max_combo', ctypes.c_int),
        ('avg_combo', ctypes.c_float),
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
                ctypes.c_int,  # max_roots
                ctypes.POINTER(ctypes.c_int),    # out_num_roots
                ctypes.POINTER(ctypes.c_int),    # out_root_action_indices
                ctypes.POINTER(ctypes.c_float),  # out_root_scores
                ctypes.POINTER(ctypes.c_int64),  # out_root_sequences
                ctypes.POINTER(ctypes.c_int),    # out_root_landing_rows
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

            # --- weight override / introspection ---
            self._lib.b2b_set_weight.argtypes = [ctypes.c_char_p, ctypes.c_float]
            self._lib.b2b_set_weight.restype = ctypes.c_int

            self._lib.b2b_get_weight.argtypes = [ctypes.c_char_p]
            self._lib.b2b_get_weight.restype = ctypes.c_float

            self._lib.b2b_reset_weights.argtypes = []
            self._lib.b2b_reset_weights.restype = None

            self._lib.b2b_get_weight_count.argtypes = []
            self._lib.b2b_get_weight_count.restype = ctypes.c_int

            self._lib.b2b_get_weight_name.argtypes = [ctypes.c_int]
            self._lib.b2b_get_weight_name.restype = ctypes.c_char_p

        self._col_bits = (
            np.uint16(1) << np.arange(10, dtype=np.uint16)
        ).astype(np.uint16)

        # Read the component count from the C side so it can't drift out of sync.
        self.NUM_DECOMPOSE = self._lib.b2b_get_num_decompose() if self._lib else 13
        self.COMPONENT_NAMES = [
            "height", "near_death", "bumpiness", "holes", "hole_ceiling",
            "b2b_flat", "b2b_sqrt", "b2b_linear", "attack", "app",
            "tslot", "immobile_clear", "garbage_prevent",
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

    def set_weight(self, name: str, value: float) -> bool:
        if not self._lib:
            raise RuntimeError("b2b_search C library not loaded")
        return bool(self._lib.b2b_set_weight(name.encode("ascii"), float(value)))

    def get_weight(self, name: str) -> float:
        if not self._lib:
            raise RuntimeError("b2b_search C library not loaded")
        return float(self._lib.b2b_get_weight(name.encode("ascii")))

    def reset_weights(self) -> None:
        if not self._lib:
            raise RuntimeError("b2b_search C library not loaded")
        self._lib.b2b_reset_weights()

    def weight_names(self) -> List[str]:
        if not self._lib:
            raise RuntimeError("b2b_search C library not loaded")
        n = self._lib.b2b_get_weight_count()
        out = []
        for i in range(n):
            raw = self._lib.b2b_get_weight_name(i)
            out.append(raw.decode("ascii") if raw else "")
        return out
