"""Device functions every search kernel calls, ported from b2b_search.c."""

# The teacher import runs the CUDA library bootstrap and must precede numba.cuda.
from teacher.constants import (
    ATTACK_PLAIN_TABLE,
    ATTACK_TM_TABLE,
    ATTACK_TS_TABLE,
    BOARD_COLS,
    BOARD_ROWS,
    DEATH_HEIGHT_CAP,
    FULL_ROW,
    GARB_ROW_MARKER,
    PERFECT_CLEAR_ATTACK,
    PIECE_I,
    PIECE_Z,
    SPAWN_BOX_ROW0_MASK,
    SPAWN_BOX_ROW1_MASK,
    SPAWN_ROW,
    SPIN_NONE,
    SPIN_T_FULL,
    SPIN_T_MINI,
    Z_B2B_SLOTS,
    Z_COMBO_SLOTS,
    Z_GARB_REM_SLOTS,
    Z_GARB_T_SLOTS,
    Z_HOLD_SLOTS,
    Z_QIDX_SLOTS,
)
from teacher.tables import (
    PIECE_MAX_COL,
    PIECE_MAX_ROW,
    PIECE_MIN_COL,
    PIECE_MIN_ROW,
    PIECE_ROW_MASKS,
    Z_B2B,
    Z_BAG,
    Z_BOARD,
    Z_COMBO,
    Z_GARB_REM,
    Z_GARB_ROW,
    Z_GARB_T,
    Z_HOLD,
    Z_QIDX,
)

from numba import cuda

import math
import numpy as np

# Attack base tables, indexed by min(clears, 4); b2b_search.c:833-839.
_ATTACK_TS = np.asarray(ATTACK_TS_TABLE, dtype=np.int32)
_ATTACK_TM = np.asarray(ATTACK_TM_TABLE, dtype=np.int32)
_ATTACK_PLAIN = np.asarray(ATTACK_PLAIN_TABLE, dtype=np.int32)


# Bit scan
@cuda.jit(device=True, inline=True)
def _ctz(mask):
    """Index of the lowest set bit of a nonzero mask, as __builtin_ctz."""
    n = 0
    m = np.uint32(mask)
    while (m & np.uint32(1)) == np.uint32(0):
        m = m >> np.uint32(1)
        n += 1
    return n


# Geometry and collision
@cuda.jit(device=True, inline=True)
def collides(board, board_height, piece, rot, r, c):
    """Whether the piece at (rot, r, c) hits a wall or a cell; b2b_search.c:580."""
    if c + PIECE_MIN_COL[piece, rot] < 0 or c + PIECE_MAX_COL[piece, rot] >= BOARD_COLS:
        return True
    if r + PIECE_MIN_ROW[piece, rot] < 0:
        return True
    if r + PIECE_MAX_ROW[piece, rot] >= board_height:
        return True
    for i in range(4):
        board_row = r + i
        if board_row < 0 or board_row >= board_height:
            continue
        mask = PIECE_ROW_MASKS[piece, rot, i]
        if c >= 0:
            shifted = mask << c
        else:
            shifted = mask >> (-c)
        if (board[board_row] & shifted) != 0:
            return True
    return False


@cuda.jit(device=True, inline=True)
def hard_drop_row(board, board_height, piece, rot, r, c):
    """Row the piece rests on when dropped from (r, c); b2b_search.c:598."""
    curr = r
    while not collides(board, board_height, piece, rot, curr + 1, c):
        curr += 1
    return curr


@cuda.jit(device=True, inline=True)
def encode_state(r, c, rot, piece):
    """BFS state index of (r, c, rot), -1 when out of range; b2b_search.c:607."""
    norm_col = c + PIECE_MIN_COL[piece, rot]
    if norm_col < 0 or norm_col >= BOARD_COLS:
        return -1
    if r < 0 or r >= BOARD_ROWS:
        return -1
    return ((r * BOARD_COLS) + norm_col) * 4 + rot


@cuda.jit(device=True, inline=True)
def decode_rot(state):
    """Rotation held by a BFS state index; b2b_search.c:615."""
    return state % 4


@cuda.jit(device=True, inline=True)
def decode_row(state, piece):
    """Row held by a BFS state index; b2b_search.c:615."""
    return (state // 4) // BOARD_COLS


@cuda.jit(device=True, inline=True)
def decode_col(state, piece):
    """Column of a BFS state index, denormalised by min_col; b2b_search.c:615."""
    rot = state % 4
    norm_col = (state // 4) % BOARD_COLS
    return norm_col - PIECE_MIN_COL[piece, rot]


# Board mutation
@cuda.jit(device=True, inline=True)
def lock_piece(board, board_height, piece, rot, r, c):
    """Set the piece cells on the board; b2b_search.c:737."""
    for i in range(4):
        board_row = r + i
        if board_row < 0 or board_row >= board_height:
            continue
        mask = PIECE_ROW_MASKS[piece, rot, i]
        if mask == 0:
            continue
        if c >= 0:
            shifted = mask << c
        else:
            shifted = mask >> (-c)
        board[board_row] = np.uint16(board[board_row] | shifted)


@cuda.jit(device=True, inline=True)
def clear_lines(board, board_height):
    """Drop the full unmarked rows and return how many; b2b_search.c:753."""
    clears = 0
    write = board_height - 1
    for read in range(board_height - 1, -1, -1):
        row = board[read]
        if (row & FULL_ROW) == FULL_ROW and (row & GARB_ROW_MARKER) == 0:
            clears += 1
        else:
            board[write] = row
            write -= 1
    for i in range(write, -1, -1):
        board[i] = np.uint16(0)
    return clears


@cuda.jit(device=True, inline=True)
def is_perfect_clear(board, board_height):
    """Whether every row is empty; b2b_search.c:1802."""
    for r in range(board_height):
        if board[r] != 0:
            return False
    return True


@cuda.jit(device=True, inline=True)
def push_garbage(board, board_height, rows):
    """Push rows of unclearable garbage in at the bottom; b2b_search.c:780."""
    if rows <= 0:
        return 0
    if rows > board_height:
        rows = board_height
    garb_row = np.uint16(FULL_ROW | GARB_ROW_MARKER)
    for r in range(board_height - rows):
        board[r] = board[r + rows]
    for r in range(board_height - rows, board_height):
        board[r] = garb_row
    return rows


# Heights
@cuda.jit(device=True, inline=True)
def compute_col_heights(board, board_height, out_heights):
    """Fill out_heights by scanning every column; b2b_search.c:153."""
    for c in range(BOARD_COLS):
        out_heights[c] = np.int8(0)
        bit = np.uint16(1) << c
        for r in range(board_height):
            if (board[r] & bit) != 0:
                out_heights[c] = np.int8(board_height - r)
                break


@cuda.jit(device=True, inline=True)
def patch_col_heights(
    parent_heights, piece, rot, land_r, col, board_height, out_heights
):
    """Parent heights raised by the placed piece; b2b_search.c:628.

    Valid only when the placement cleared no lines and pushed no garbage.
    """
    for c in range(BOARD_COLS):
        out_heights[c] = parent_heights[c]
    for dr in range(4):
        m = np.int64(PIECE_ROW_MASKS[piece, rot, dr])
        if m == 0:
            continue
        col_height = board_height - (land_r + dr)
        while m != 0:
            c = col + _ctz(m)
            if c >= 0 and c < BOARD_COLS and col_height > out_heights[c]:
                out_heights[c] = np.int8(col_height)
            m &= m - 1


@cuda.jit(device=True, inline=True)
def mean_col_heights(heights):
    """Mean of the ten column heights; b2b_search.c:164."""
    total = 0
    for c in range(BOARD_COLS):
        total += heights[c]
    return np.float32(total) / np.float32(BOARD_COLS)


@cuda.jit(device=True, inline=True)
def max_stack_height(board, board_height):
    """Height of the topmost occupied row; b2b_search.c:188."""
    for r in range(board_height):
        if board[r] != 0:
            return board_height - r
    return 0


# Death
@cuda.jit(device=True, inline=True)
def spawn_envelope_blocked(board):
    """Whether the 7-cell spawn box holds a cell; b2b_search.c:184."""
    return (board[SPAWN_ROW] & SPAWN_BOX_ROW0_MASK) != 0 or (
        board[SPAWN_ROW + 1] & SPAWN_BOX_ROW1_MASK
    ) != 0


@cuda.jit(device=True, inline=True)
def placement_is_dead(board, col_heights, garbage_remaining):
    """Whether spawn is blocked or the stack hits the cap; b2b_search.c:200."""
    mh = 0
    for c in range(BOARD_COLS):
        if col_heights[c] > mh:
            mh = col_heights[c]
    return spawn_envelope_blocked(board) or (mh + garbage_remaining) >= DEATH_HEIGHT_CAP


# Spin classification
@cuda.jit(device=True, inline=True)
def _corner_filled(board, board_height, cr, cc):
    """Whether a corner counts as filled, off-board included; b2b_search.c:665."""
    if cr >= board_height or cc < 0 or cc >= BOARD_COLS or cr < 0:
        return 1
    if (board[cr] & (1 << cc)) != 0:
        return 1
    return 0


@cuda.jit(device=True, inline=True)
def detect_t_spin(board, board_height, land_r, c, rot, dloc_sum):
    """T-spin type of a T locked at (rot, land_r, c); b2b_search.c:656."""
    tl = _corner_filled(board, board_height, land_r + 0, c + 0)
    tr = _corner_filled(board, board_height, land_r + 0, c + 2)
    br = _corner_filled(board, board_height, land_r + 2, c + 2)
    bl = _corner_filled(board, board_height, land_r + 2, c + 0)

    if tl + tr + br + bl < 3:
        return SPIN_NONE

    if rot == 0:
        front_filled = tl + tr
        back_filled = br + bl
    elif rot == 1:
        front_filled = tr + br
        back_filled = tl + bl
    elif rot == 2:
        front_filled = br + bl
        back_filled = tl + tr
    else:
        front_filled = tl + bl
        back_filled = tr + br

    if front_filled == 2 and back_filled >= 1:
        return SPIN_T_FULL
    elif front_filled == 1 and back_filled == 2:
        if dloc_sum > 2:
            return SPIN_T_FULL
        else:
            return SPIN_T_MINI
    return SPIN_NONE


@cuda.jit(device=True, inline=True)
def check_immobility(board, board_height, piece, rot, r, c):
    """Whether the piece is blocked in all four directions; b2b_search.c:720."""
    if not collides(board, board_height, piece, rot, r + 1, c):
        return False
    if not collides(board, board_height, piece, rot, r - 1, c):
        return False
    if not collides(board, board_height, piece, rot, r, c + 1):
        return False
    if not collides(board, board_height, piece, rot, r, c - 1):
        return False
    return True


# Scoring
@cuda.jit(device=True, inline=True)
def compute_attack(clears, spin_type, b2b, combo, perfect_clear, out):
    """Attack of one lock; b2b_search.c:805.

    out is float32 (attack, new_b2b, new_combo, b2b_maintaining, surge).
    """
    attack = np.float32(0.0)
    new_b2b = b2b
    new_combo = combo
    maintaining = np.float32(0.0)
    surge = np.float32(0.0)

    if clears > 0:
        if spin_type != SPIN_NONE or clears == 4 or perfect_clear:
            new_b2b = b2b + 1
            maintaining = np.float32(1.0)
        else:
            if b2b >= 4:
                surge = np.float32(b2b)
            new_b2b = -1

        new_combo = combo + 1

        idx = clears if clears < 5 else 4
        if perfect_clear:
            attack += np.float32(PERFECT_CLEAR_ATTACK)
        elif spin_type == SPIN_T_FULL:
            attack += np.float32(_ATTACK_TS[idx])
        elif spin_type == SPIN_T_MINI:
            attack += np.float32(_ATTACK_TM[idx])
        else:
            attack += np.float32(_ATTACK_PLAIN[idx])

        if b2b > -1:
            attack += np.float32(1.0)

        if combo > 0:
            if attack > np.float32(0.0):
                scale = np.float32(1.0) + np.float32(0.25) * np.float32(combo)
                attack = np.float32(math.floor(attack * scale))
            else:
                arg = np.float32(1.0) + np.float32(1.25) * np.float32(combo)
                logged = np.float32(math.log(arg))
                attack = np.float32(math.floor(logged))

        attack += surge
    else:
        new_combo = -1

    out[0] = attack
    out[1] = np.float32(new_b2b)
    out[2] = np.float32(new_combo)
    out[3] = maintaining
    out[4] = surge


@cuda.jit(device=True, inline=True)
def latch_unlicensed_cash(clears, spin_type, perfect_clear, parent_combo):
    """Whether this lock's attack is unlicensed cash; b2b_search.c:172."""
    d_cash = clears > 0 and spin_type == SPIN_NONE and (clears == 4 or perfect_clear)
    if d_cash and parent_combo < 0:
        return 1
    return 0


# Bag tracking
@cuda.jit(device=True, inline=True)
def bag_consume_piece(bag_seen, piece):
    """Bag mask after a piece is taken, 0 once all seven are; b2b_search.c:299."""
    seen = bag_seen | (1 << piece)
    if (seen & 0xFE) == 0xFE:
        seen = 0
    return np.uint8(seen)


@cuda.jit(device=True, inline=True)
def bag_remaining_count(bag_seen):
    """Count of pieces left in the current bag; b2b_search.c:308."""
    count = 0
    for p in range(PIECE_I, PIECE_Z + 1):
        if (bag_seen & (1 << p)) == 0:
            count += 1
    return count


@cuda.jit(device=True, inline=True)
def bag_nth_remaining(bag_seen, k):
    """The k-th piece left in the bag, -1 past the last; b2b_search.c:308."""
    count = 0
    for p in range(PIECE_I, PIECE_Z + 1):
        if (bag_seen & (1 << p)) == 0:
            if count == k:
                return p
            count += 1
    return -1


# Hashing
@cuda.jit(device=True, inline=True)
def state_hash(
    board,
    board_height,
    b2b,
    combo,
    hold_piece,
    next_queue_idx,
    bag_seen,
    garbage_remaining,
    garbage_timer,
):
    """Zobrist hash of the beam dedupe key; b2b_search.c:260."""
    h = np.uint64(0)
    for r in range(board_height):
        row = board[r]
        play = row & FULL_ROW
        while play != 0:
            h ^= Z_BOARD[r, _ctz(play)]
            play &= play - 1
        if (row & GARB_ROW_MARKER) != 0:
            h ^= Z_GARB_ROW[r]

    if b2b < 0:
        b2b_i = Z_B2B_SLOTS - 1
    else:
        b2b_i = b2b % (Z_B2B_SLOTS - 1)
    if combo < 0:
        combo_i = Z_COMBO_SLOTS - 1
    else:
        combo_i = combo % (Z_COMBO_SLOTS - 1)

    h ^= Z_B2B[b2b_i]
    h ^= Z_COMBO[combo_i]
    h ^= Z_HOLD[hold_piece & (Z_HOLD_SLOTS - 1)]
    h ^= Z_QIDX[next_queue_idx & (Z_QIDX_SLOTS - 1)]
    h ^= Z_BAG[bag_seen]

    if garbage_remaining < 0:
        gr_i = 0
    elif garbage_remaining >= Z_GARB_REM_SLOTS:
        gr_i = Z_GARB_REM_SLOTS - 1
    else:
        gr_i = garbage_remaining
    if garbage_timer < 0:
        gt_i = 0
    elif garbage_timer >= Z_GARB_T_SLOTS:
        gt_i = Z_GARB_T_SLOTS - 1
    else:
        gt_i = garbage_timer

    h ^= Z_GARB_REM[gr_i]
    h ^= Z_GARB_T[gt_i]
    return h
