"""Constants mirrored from the beam search in tetrisenv/TetrisEnv/b2b_search.c."""

import math

# Geometry
BOARD_ROWS = 40  # b2b_search.c:17
VISIBLE_ROWS = 20  # b2b_search.c:18
BOARD_COLS = 10  # b2b_search.c:19
ROTATIONS = 4  # b2b_search.c:20
SPAWN_ROW = 17  # b2b_search.c:25
SPAWN_COL = 3  # b2b_search.c:886
DEATH_HEIGHT_CAP = 35  # b2b_search.c:26
PERFECT_CLEAR_ATTACK = 5  # b2b_search.c:27
FULL_ROW = 0x3FF  # b2b_search.c:754
GARB_ROW_MARKER = 0x400  # bit 10 marks an unclearable row; b2b_search.c:69
SPAWN_BOX_ROW0_MASK = 0x38  # row 17 cols 3-5; b2b_search.c:185
SPAWN_BOX_ROW1_MASK = 0x78  # row 18 cols 3-6; b2b_search.c:185

# Piece ids
PIECE_N = 0  # b2b_search.c:51
PIECE_I = 1  # b2b_search.c:52
PIECE_J = 2  # b2b_search.c:53
PIECE_L = 3  # b2b_search.c:54
PIECE_O = 4  # b2b_search.c:55
PIECE_S = 5  # b2b_search.c:56
PIECE_T = 6  # b2b_search.c:57
PIECE_Z = 7  # b2b_search.c:58
PIECE_NAMES = ("N", "I", "J", "L", "O", "S", "T", "Z")  # b2b_search.c:51-58

# Spin ids
SPIN_NONE = 0  # b2b_search.c:61
SPIN_T_MINI = 1  # b2b_search.c:62
SPIN_T_FULL = 2  # b2b_search.c:63
SPIN_ALL_MINI = 3  # b2b_search.c:64

# Key ids
KEY_START = 0  # b2b_search.c:37
KEY_HOLD = 1  # b2b_search.c:38
KEY_TAP_LEFT = 2  # b2b_search.c:39
KEY_TAP_RIGHT = 3  # b2b_search.c:40
KEY_DAS_LEFT = 4  # b2b_search.c:41
KEY_DAS_RIGHT = 5  # b2b_search.c:42
KEY_CLOCKWISE = 6  # b2b_search.c:43
KEY_ANTICLOCKWISE = 7  # b2b_search.c:44
KEY_ROTATE_180 = 8  # b2b_search.c:45
KEY_SOFT_DROP = 9  # b2b_search.c:46
KEY_HARD_DROP = 10  # b2b_search.c:47
KEY_PAD = 11  # b2b_search.c:48

# Search caps
MAX_PLACEMENTS = 512  # b2b_search.c:76
MAX_BEAM_WIDTH = 2048  # b2b_search.c:79
MAX_SEARCH_DEPTH = 16  # b2b_search.c:80
BFS_QUEUE_CAPACITY = 8192  # b2b_search.c:72
BFS_STATE_SPACE = BOARD_ROWS * BOARD_COLS * ROTATIONS  # b2b_search.c:73
BFS_DEPTH_CAP = 12  # a state at this depth is not expanded; b2b_search.c:970
BEAM_STRATA = 8  # b2b_search.c:1621
ROOT_CAPACITY = 2 * MAX_PLACEMENTS  # b2b_search.c:1959
ROOT_SCORE_INIT = -1e30  # unseeded root slot, never emitted; b2b_search.c:1961

# Zobrist slot counts
Z_B2B_SLOTS = 64  # b2b_search.c:218
Z_COMBO_SLOTS = 64  # b2b_search.c:219
Z_HOLD_SLOTS = 8  # b2b_search.c:220
Z_QIDX_SLOTS = 32  # b2b_search.c:221
Z_BAG_SLOTS = 256  # b2b_search.c:222
Z_GARB_REM_SLOTS = 64  # b2b_search.c:223
Z_GARB_T_SLOTS = 32  # b2b_search.c:224

# Transposition cache, carried for reference; the GPU port has no such table
TT_SIZE = 1 << 16  # b2b_search.c:362
TT_MASK = TT_SIZE - 1  # b2b_search.c:363
TT_GENERATION_EXPIRY = 4  # b2b_search.c:364

# Heuristic weights, all in attack lines; b2b_search.c:398-428
W_RISK = 300.0  # b2b_search.c:404
RISK_H0 = 18.0  # b2b_search.c:405
RISK_TAU = 1.5  # b2b_search.c:406
# RISK_TABLE[h] for effective heights below DEATH_HEIGHT_CAP; b2b_search.c:569-571
RISK_TABLE = tuple(
    W_RISK * math.exp((h - RISK_H0) / RISK_TAU) for h in range(DEATH_HEIGHT_CAP)
)
W_AVG_HEIGHT = 4.0  # b2b_search.c:409
W_BUMPINESS = 0.125  # b2b_search.c:410
W_HOLES = 0.5  # b2b_search.c:412
HOLE_HEIGHT_SCALE = 32.0  # b2b_search.c:413
W_HOLE_CEILING = 0.125  # b2b_search.c:414
W_B2B_FLAT = 1.0  # b2b_search.c:417
W_B2B_LINEAR = 2.0  # b2b_search.c:419
W_ATTACK_H = 10.0  # b2b_search.c:421
W_GARBAGE_PREVENT = 0.5  # b2b_search.c:422
W_IMMOBILE_CLEAR = 0.5  # b2b_search.c:425
W_IMMOBILE_LINES = 0.125  # b2b_search.c:426
W_EXEC_RAMP = 18.0  # b2b_search.c:427
W_CHAIN = (0.0, 0.0, 3.5, 9.0, 16.0)  # by spin clears; b2b_search.c:428
# Root reporting scale: each frontier maps to [-1, 0] and a root keeps the best value
# it reached plus one step per depth; b2b_search.c:1596-1619.
ROOT_DEPTH_STEP = 1.125  # b2b_search.c:1599
ROOT_FLOOR = -2.0  # every child dead, so the root was never raised; b2b_search.c:1600
DEATH_SCORE = -1e6  # b2b_search.c:1679

# Attack base tables, indexed by min(clears, 4). A perfect clear takes
# PERFECT_CLEAR_ATTACK in place of any table; b2b_search.c:829-841.
ATTACK_TS_TABLE = (0, 2, 4, 6, 0)  # SPIN_T_FULL; b2b_search.c:833
ATTACK_TM_TABLE = (0, 0, 1, 2, 0)  # SPIN_T_MINI; b2b_search.c:836
ATTACK_PLAIN_TABLE = (0, 0, 1, 2, 4)  # SPIN_NONE and SPIN_ALL_MINI; b2b_search.c:839

# Action encoding
ACTION_HOLD_STRIDE = 160  # b2b_search.c:2411
ACTION_ROT_STRIDE = 40  # b2b_search.c:2411
ACTION_COL_STRIDE = 4  # b2b_search.c:2411
ACTION_SPACE = 2 * ACTION_HOLD_STRIDE


def action_index(is_hold: int, rot: int, norm_col: int, spin: int) -> int:
    """Action id for a placement, is_hold*160 + rot*40 + norm_col*4 + spin."""
    return (
        is_hold * ACTION_HOLD_STRIDE
        + rot * ACTION_ROT_STRIDE
        + norm_col * ACTION_COL_STRIDE
        + spin
    )


def decode_action(action: int) -> tuple[int, int, int, int]:
    """Inverse of action_index, as (is_hold, rot, norm_col, spin)."""
    is_hold, rest = divmod(action, ACTION_HOLD_STRIDE)
    rot, rest = divmod(rest, ACTION_ROT_STRIDE)
    norm_col, spin = divmod(rest, ACTION_COL_STRIDE)
    return is_hold, rot, norm_col, spin
