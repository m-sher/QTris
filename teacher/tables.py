"""Piece, kick and Zobrist tables built to match tetrisenv/TetrisEnv/b2b_search.c."""

import numpy as np

from teacher.constants import (
    BOARD_COLS,
    BOARD_ROWS,
    PIECE_I,
    PIECE_J,
    PIECE_L,
    PIECE_O,
    PIECE_S,
    PIECE_T,
    PIECE_Z,
    ROTATIONS,
    Z_B2B_SLOTS,
    Z_BAG_SLOTS,
    Z_COMBO_SLOTS,
    Z_GARB_REM_SLOTS,
    Z_GARB_T_SLOTS,
    Z_HOLD_SLOTS,
    Z_QIDX_SLOTS,
)

NUM_PIECES = 8
MASK64 = (1 << 64) - 1

# Piece tables
# _ORIENTATIONS[piece][rot] = (row_masks, min_col, max_col, min_row, max_row).
# PIECE_N is absent and stays all-zero, as b2b_init_pieces memsets it.
_ORIENTATIONS = {
    PIECE_I: (  # b2b_search.c:442
        ((0, 15, 0, 0), 0, 3, 1, 1),
        ((4, 4, 4, 4), 2, 2, 0, 3),
        ((0, 0, 15, 0), 0, 3, 2, 2),
        ((2, 2, 2, 2), 1, 1, 0, 3),
    ),
    PIECE_J: (  # b2b_search.c:448
        ((1, 7, 0, 0), 0, 2, 0, 1),
        ((6, 2, 2, 0), 1, 2, 0, 2),
        ((0, 7, 4, 0), 0, 2, 1, 2),
        ((2, 2, 3, 0), 0, 1, 0, 2),
    ),
    PIECE_L: (  # b2b_search.c:454
        ((4, 7, 0, 0), 0, 2, 0, 1),
        ((2, 2, 6, 0), 1, 2, 0, 2),
        ((0, 7, 1, 0), 0, 2, 1, 2),
        ((3, 2, 2, 0), 0, 1, 0, 2),
    ),
    PIECE_O: (  # b2b_search.c:460
        ((6, 6, 0, 0), 1, 2, 0, 1),
        ((6, 6, 0, 0), 1, 2, 0, 1),
        ((6, 6, 0, 0), 1, 2, 0, 1),
        ((6, 6, 0, 0), 1, 2, 0, 1),
    ),
    PIECE_S: (  # b2b_search.c:464
        ((6, 3, 0, 0), 0, 2, 0, 1),
        ((2, 6, 4, 0), 1, 2, 0, 2),
        ((0, 6, 3, 0), 0, 2, 1, 2),
        ((1, 3, 2, 0), 0, 1, 0, 2),
    ),
    PIECE_T: (  # b2b_search.c:470
        ((2, 7, 0, 0), 0, 2, 0, 1),
        ((2, 6, 2, 0), 1, 2, 0, 2),
        ((0, 7, 2, 0), 0, 2, 1, 2),
        ((2, 3, 2, 0), 0, 1, 0, 2),
    ),
    PIECE_Z: (  # b2b_search.c:476
        ((3, 6, 0, 0), 0, 2, 0, 1),
        ((4, 6, 2, 0), 1, 2, 0, 2),
        ((0, 3, 6, 0), 0, 2, 1, 2),
        ((2, 3, 1, 0), 0, 1, 0, 2),
    ),
}

PIECE_ROW_MASKS = np.zeros((NUM_PIECES, ROTATIONS, 4), dtype=np.uint16)
PIECE_MIN_COL = np.zeros((NUM_PIECES, ROTATIONS), dtype=np.int8)
PIECE_MAX_COL = np.zeros((NUM_PIECES, ROTATIONS), dtype=np.int8)
PIECE_MIN_ROW = np.zeros((NUM_PIECES, ROTATIONS), dtype=np.int8)
PIECE_MAX_ROW = np.zeros((NUM_PIECES, ROTATIONS), dtype=np.int8)

for _piece, _orients in _ORIENTATIONS.items():
    for _rot, (_masks, _min_col, _max_col, _min_row, _max_row) in enumerate(_orients):
        PIECE_ROW_MASKS[_piece, _rot] = _masks
        PIECE_MIN_COL[_piece, _rot] = _min_col
        PIECE_MAX_COL[_piece, _rot] = _max_col
        PIECE_MIN_ROW[_piece, _rot] = _min_row
        PIECE_MAX_ROW[_piece, _rot] = _max_row


def _shape_key(
    row_masks: tuple[int, ...], min_col: int, min_row: int, max_row: int
) -> int:
    """Row masks shifted to the bounding box origin, packed 4 bits per row."""
    key = 0
    j = 0
    while j + min_row <= max_row and j < 4:
        key |= ((row_masks[min_row + j] >> min_col) & 0xF) << (4 * j)
        j += 1
    return key


PIECE_SHAPE_KEY = np.zeros((NUM_PIECES, ROTATIONS), dtype=np.uint16)

for _piece in range(NUM_PIECES):
    for _rot in range(ROTATIONS):
        PIECE_SHAPE_KEY[_piece, _rot] = _shape_key(
            tuple(int(m) for m in PIECE_ROW_MASKS[_piece, _rot]),
            int(PIECE_MIN_COL[_piece, _rot]),
            int(PIECE_MIN_ROW[_piece, _rot]),
            int(PIECE_MAX_ROW[_piece, _rot]),
        )

# Kick tables
# _KICKS_*_SRC[(from_rot, to_rot)] = tests as (dr, dc), applied to (row, col).
# Slots past the listed tests stay (0, 0), as b2b_init_pieces memsets the arrays.
_KICKS_JLOSTZ_SRC = {
    (0, 1): ((0, -1), (-1, -1), (2, 0), (2, -1)),  # b2b_search.c:482
    (0, 3): ((0, 1), (-1, 1), (2, 0), (2, 1)),  # b2b_search.c:485
    (1, 0): ((0, 1), (1, 1), (-2, 0), (-2, 1)),  # b2b_search.c:488
    (1, 2): ((0, 1), (1, 1), (-2, 0), (-2, 1)),  # b2b_search.c:491
    (2, 1): ((0, -1), (-1, -1), (2, 0), (2, -1)),  # b2b_search.c:494
    (2, 3): ((0, 1), (-1, 1), (2, 0), (2, 1)),  # b2b_search.c:497
    (3, 0): ((0, -1), (1, -1), (-2, 0), (-2, -1)),  # b2b_search.c:500
    (3, 2): ((0, -1), (1, -1), (-2, 0), (-2, -1)),  # b2b_search.c:503
    (0, 2): ((-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1)),  # b2b_search.c:507
    (1, 3): ((0, 1), (-2, 1), (-1, 1), (-2, 0), (-1, 0)),  # b2b_search.c:510
    (2, 0): ((1, 0), (1, -1), (1, 1), (0, -1), (0, 1)),  # b2b_search.c:513
    (3, 1): ((0, -1), (-2, -1), (-1, -1), (-2, 0), (-1, 0)),  # b2b_search.c:516
}

_KICKS_I_SRC = {
    (0, 1): ((0, 1), (0, -2), (1, -2), (-2, 1)),  # b2b_search.c:520
    (0, 3): ((0, -1), (0, 2), (1, 2), (-2, -1)),  # b2b_search.c:523
    (1, 0): ((0, -1), (0, 2), (2, -1), (-1, 2)),  # b2b_search.c:526
    (1, 2): ((0, -1), (0, 2), (-2, -1), (1, 2)),  # b2b_search.c:529
    (2, 1): ((0, -2), (0, 1), (-1, -2), (2, 1)),  # b2b_search.c:532
    (2, 3): ((0, 2), (0, -1), (-1, 2), (2, -1)),  # b2b_search.c:535
    (3, 0): ((0, 1), (0, -2), (2, 1), (-1, -2)),  # b2b_search.c:538
    (3, 2): ((0, 1), (0, -2), (-2, 1), (1, -2)),  # b2b_search.c:541
    (0, 2): ((-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1)),  # b2b_search.c:545
    (1, 3): ((0, 1), (-2, 1), (-1, 1), (-2, 0), (-1, 0)),  # b2b_search.c:548
    (2, 0): ((1, 0), (1, -1), (1, 1), (0, -1), (0, 1)),  # b2b_search.c:551
    (3, 1): ((0, -1), (-2, -1), (-1, -1), (-2, 0), (-1, 0)),  # b2b_search.c:554
}

MAX_KICKS = 5


def _build_kick_table(
    src: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> np.ndarray:
    """Kick array (ROTATIONS, ROTATIONS, MAX_KICKS, 2) from a (dr, dc) source dict."""
    table = np.zeros((ROTATIONS, ROTATIONS, MAX_KICKS, 2), dtype=np.int8)
    for (from_rot, to_rot), tests in src.items():
        for k, (dr, dc) in enumerate(tests):
            table[from_rot, to_rot, k] = (dr, dc)
    return table


KICKS_JLOSTZ = _build_kick_table(_KICKS_JLOSTZ_SRC)
KICKS_I = _build_kick_table(_KICKS_I_SRC)

KICKS = np.zeros((NUM_PIECES, ROTATIONS, ROTATIONS, MAX_KICKS, 2), dtype=np.int8)
KICKS[:] = KICKS_JLOSTZ
KICKS[PIECE_I] = KICKS_I  # b2b_search.c:1020

# 180 turns read 5 tests, quarter turns 4; b2b_search.c:1023. The 5-test path skips
# any (0, 0) slot (b2b_search.c:1028) and no 180 row holds one.
KICK_COUNT = np.zeros((ROTATIONS, ROTATIONS), dtype=np.int8)

for _from_rot in range(ROTATIONS):
    for _to_rot in range(ROTATIONS):
        if _from_rot == _to_rot:
            continue
        KICK_COUNT[_from_rot, _to_rot] = 5 if (_to_rot - _from_rot) % 4 == 2 else 4

# Zobrist tables
ZOBRIST_SEED = 0xB2BC0DE51CE5EED  # b2b_search.c:246

# Draw order and shapes of zobrist_init, b2b_search.c:245.
_ZOBRIST_LAYOUT = (
    ("Z_BOARD", (BOARD_ROWS, BOARD_COLS)),
    ("Z_GARB_ROW", (BOARD_ROWS,)),
    ("Z_B2B", (Z_B2B_SLOTS,)),
    ("Z_COMBO", (Z_COMBO_SLOTS,)),
    ("Z_HOLD", (Z_HOLD_SLOTS,)),
    ("Z_QIDX", (Z_QIDX_SLOTS,)),
    ("Z_BAG", (Z_BAG_SLOTS,)),
    ("Z_GARB_REM", (Z_GARB_REM_SLOTS,)),
    ("Z_GARB_T", (Z_GARB_T_SLOTS,)),
)


def splitmix64(state: int) -> tuple[int, int]:
    """Next splitmix64 draw and the advanced state, as (value, new_state)."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31), state


def zobrist_tables() -> dict[str, np.ndarray]:
    """Every Zobrist table, drawn from ZOBRIST_SEED in _ZOBRIST_LAYOUT order."""
    state = ZOBRIST_SEED
    tables = {}
    for name, shape in _ZOBRIST_LAYOUT:
        count = 1
        for dim in shape:
            count *= dim
        values = np.empty(count, dtype=np.uint64)
        for i in range(count):
            values[i], state = splitmix64(state)
        tables[name] = values.reshape(shape)
    return tables


_ZOBRIST = zobrist_tables()
Z_BOARD = _ZOBRIST["Z_BOARD"]
Z_GARB_ROW = _ZOBRIST["Z_GARB_ROW"]
Z_B2B = _ZOBRIST["Z_B2B"]
Z_COMBO = _ZOBRIST["Z_COMBO"]
Z_HOLD = _ZOBRIST["Z_HOLD"]
Z_QIDX = _ZOBRIST["Z_QIDX"]
Z_BAG = _ZOBRIST["Z_BAG"]
Z_GARB_REM = _ZOBRIST["Z_GARB_REM"]
Z_GARB_T = _ZOBRIST["Z_GARB_T"]


def device_tables() -> dict[str, np.ndarray]:
    """Every table the search kernels read, keyed by name, as host numpy arrays."""
    return {
        "PIECE_ROW_MASKS": PIECE_ROW_MASKS,
        "PIECE_MIN_COL": PIECE_MIN_COL,
        "PIECE_MAX_COL": PIECE_MAX_COL,
        "PIECE_MIN_ROW": PIECE_MIN_ROW,
        "PIECE_MAX_ROW": PIECE_MAX_ROW,
        "PIECE_SHAPE_KEY": PIECE_SHAPE_KEY,
        "KICKS": KICKS,
        "KICK_COUNT": KICK_COUNT,
        "Z_BOARD": Z_BOARD,
        "Z_GARB_ROW": Z_GARB_ROW,
        "Z_B2B": Z_B2B,
        "Z_COMBO": Z_COMBO,
        "Z_HOLD": Z_HOLD,
        "Z_QIDX": Z_QIDX,
        "Z_BAG": Z_BAG,
        "Z_GARB_REM": Z_GARB_REM,
        "Z_GARB_T": Z_GARB_T,
    }
