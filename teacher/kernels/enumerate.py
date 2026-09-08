"""BFS placement enumerator, one CUDA thread per work item; b2b_search.c:872."""

# The teacher import runs the CUDA library bootstrap and must precede numba.cuda.
from teacher.constants import (
    BFS_DEPTH_CAP,
    BFS_STATE_SPACE,
    KEY_ANTICLOCKWISE,
    KEY_CLOCKWISE,
    KEY_DAS_LEFT,
    KEY_DAS_RIGHT,
    KEY_ROTATE_180,
    KEY_SOFT_DROP,
    KEY_TAP_LEFT,
    KEY_TAP_RIGHT,
    PIECE_T,
    SPAWN_COL,
    SPAWN_ROW,
    SPIN_ALL_MINI,
    SPIN_NONE,
)
from teacher.kernels.rules import (
    check_immobility,
    collides,
    decode_col,
    decode_rot,
    decode_row,
    detect_t_spin,
    encode_state,
    hard_drop_row,
)
from teacher.tables import (
    KICKS,
    PIECE_MAX_ROW,
    PIECE_MIN_COL,
    PIECE_MIN_ROW,
    PIECE_SHAPE_KEY,
)

from numba import cuda

import numpy as np

# The eight moves tried per state, in order; b2b_search.c:972.
_MOVES = np.array(
    [
        KEY_TAP_LEFT,
        KEY_TAP_RIGHT,
        KEY_DAS_LEFT,
        KEY_DAS_RIGHT,
        KEY_CLOCKWISE,
        KEY_ANTICLOCKWISE,
        KEY_ROTATE_180,
        KEY_SOFT_DROP,
    ],
    dtype=np.int8,
)

# Meta byte per BFS state: depth 0..BFS_DEPTH_CAP in bits 0-3, delta_r != 0 in
# bit 4, abs(delta_row) + abs(delta_col) in bits 5-7. Bits 5-7 hold a kick offset
# only, so they stay under 4; every other move stores 0 there and leaves bit 4
# clear. A depth nibble of 15 is unreachable, so 0xFF marks an unvisited state.
_META_DELTA_R = 0x10
_META_DLOC_SHIFT = 5
_META_DEPTH_MASK = 0x0F
_META_UNVISITED = 0xFF


@cuda.jit
def enumerate_kernel(
    n_items,
    wi_parent,
    wi_piece,
    wi_carry,
    boards,
    board_height,
    pl_rot,
    pl_col,
    pl_row,
    pl_spin,
    wi_count,
    placement_overflow,
):
    """Write every unique placement of one work item per thread; b2b_search.c:872."""
    i = cuda.grid(1)
    if i >= n_items:
        return

    wi_count[i] = 0

    # A carry item copies its parent forward and places nothing; b2b_search.c:2283.
    if wi_carry[i] != 0:
        return

    board = boards[wi_parent[i]]
    piece = wi_piece[i]
    max_out = pl_rot.shape[1]

    # Spawn
    start_state = encode_state(SPAWN_ROW, SPAWN_COL, 0, piece)
    if start_state == -1 or collides(
        board, board_height, piece, 0, SPAWN_ROW, SPAWN_COL
    ):
        return

    # cuda.local.array takes a literal; this is teacher.constants BFS_STATE_SPACE.
    queue = cuda.local.array(1600, np.int16)
    meta = cuda.local.array(1600, np.uint8)
    for s in range(BFS_STATE_SPACE):
        meta[s] = np.uint8(_META_UNVISITED)

    head = 0
    tail = 0
    queue[tail] = np.int16(start_state)
    tail += 1
    meta[start_state] = np.uint8(0)

    count = 0

    while head != tail:
        curr_state = queue[head]
        head += 1

        rot = decode_rot(curr_state)
        r = decode_row(curr_state, piece)
        c = decode_col(curr_state, piece)
        mb = meta[curr_state]
        depth = mb & _META_DEPTH_MASK

        # Emit, only from a state that is its own landing state; b2b_search.c:924.
        land_r = hard_drop_row(board, board_height, piece, rot, r, c)
        if r == land_r and land_r >= SPAWN_ROW:
            if count >= max_out:
                cuda.atomic.max(placement_overflow, 0, np.int32(1))
            else:
                spin = SPIN_NONE
                if (mb & _META_DELTA_R) != 0:
                    dloc_sum = mb >> _META_DLOC_SHIFT
                    if piece == PIECE_T:
                        spin = detect_t_spin(
                            board, board_height, land_r, c, rot, dloc_sum
                        )
                    elif check_immobility(board, board_height, piece, rot, land_r, c):
                        spin = SPIN_ALL_MINI

                # Dedupe on spin, normalised anchor and shape key; b2b_search.c:939.
                norm_col_k = c + PIECE_MIN_COL[piece, rot]
                norm_row_k = land_r + PIECE_MIN_ROW[piece, rot]
                shape_key = PIECE_SHAPE_KEY[piece, rot]
                dup = False
                for j in range(count):
                    rot_j = pl_rot[i, j]
                    if (
                        pl_spin[i, j] == spin
                        and pl_col[i, j] + PIECE_MIN_COL[piece, rot_j] == norm_col_k
                        and pl_row[i, j] + PIECE_MIN_ROW[piece, rot_j] == norm_row_k
                        and PIECE_SHAPE_KEY[piece, rot_j] == shape_key
                    ):
                        dup = True
                        break

                if not dup:
                    pl_rot[i, count] = np.int8(rot)
                    pl_col[i, count] = np.int8(c)
                    pl_row[i, count] = np.int8(land_r)
                    pl_spin[i, count] = np.int8(spin)
                    count += 1

        if depth >= BFS_DEPTH_CAP:
            continue

        for m in range(8):
            key = _MOVES[m]
            nr = r
            nc = c
            nrot = rot
            dr = 0
            drow = 0
            dcol = 0
            valid = False

            if key == KEY_TAP_LEFT:
                if not collides(board, board_height, piece, rot, r, c - 1):
                    nc = c - 1
                    valid = True
                    dcol = -1
            elif key == KEY_TAP_RIGHT:
                if not collides(board, board_height, piece, rot, r, c + 1):
                    nc = c + 1
                    valid = True
                    dcol = 1
            elif key == KEY_DAS_LEFT:
                tmp = c
                while not collides(board, board_height, piece, rot, r, tmp - 1):
                    tmp -= 1
                if tmp != c:
                    nc = tmp
                    valid = True
                    dcol = nc - c
            elif key == KEY_DAS_RIGHT:
                tmp = c
                while not collides(board, board_height, piece, rot, r, tmp + 1):
                    tmp += 1
                if tmp != c:
                    nc = tmp
                    valid = True
                    dcol = nc - c
            elif key == KEY_SOFT_DROP:
                tmp = r
                max_row = PIECE_MAX_ROW[piece, rot]
                while not collides(board, board_height, piece, rot, tmp + 1, c):
                    tmp += 1
                    if tmp + max_row >= board_height - 1:
                        break
                if tmp != r:
                    nr = tmp
                    valid = True
                    drow = nr - r
            else:
                # Rotation, un-kicked target first; b2b_search.c:1015.
                if key == KEY_CLOCKWISE:
                    delta = 1
                elif key == KEY_ANTICLOCKWISE:
                    delta = 3
                else:
                    delta = 2
                next_rot = (rot + delta) % 4

                if not collides(board, board_height, piece, next_rot, r, c):
                    nrot = next_rot
                    valid = True
                    dr = -1 if delta == 3 else delta
                else:
                    # 5 tests for a 180, 4 otherwise; b2b_search.c:1023.
                    kick_count = 5 if key == KEY_ROTATE_180 else 4
                    for k in range(kick_count):
                        kdr = KICKS[piece, rot, next_rot, k, 0]
                        kdc = KICKS[piece, rot, next_rot, k, 1]
                        if kdr == 0 and kdc == 0 and kick_count == 5:
                            continue
                        if not collides(
                            board, board_height, piece, next_rot, r + kdr, c + kdc
                        ):
                            nr = r + kdr
                            nc = c + kdc
                            nrot = next_rot
                            valid = True
                            dr = -1 if delta == 3 else delta
                            drow = kdr
                            dcol = kdc
                            break

            if valid:
                next_s = encode_state(nr, nc, nrot, piece)
                if next_s != -1 and meta[next_s] == _META_UNVISITED:
                    packed = depth + 1
                    if dr != 0:
                        dloc = abs(drow) + abs(dcol)
                        packed |= _META_DELTA_R | (dloc << _META_DLOC_SHIFT)
                    meta[next_s] = np.uint8(packed)
                    queue[tail] = np.int16(next_s)
                    tail += 1

    wi_count[i] = count


def debug_enumerate(board_masks, piece, board_height=40, max_placements=512, carry=0):
    """Placements of one board and piece, as numpy int8 (rot, col, row, spin)."""
    boards = np.ascontiguousarray(board_masks, dtype=np.uint16).reshape(1, -1)
    d_boards = cuda.to_device(boards)
    d_parent = cuda.to_device(np.zeros(1, dtype=np.int32))
    d_piece = cuda.to_device(np.full(1, piece, dtype=np.int32))
    d_carry = cuda.to_device(np.full(1, carry, dtype=np.uint8))
    d_rot = cuda.to_device(np.zeros((1, max_placements), dtype=np.int8))
    d_col = cuda.to_device(np.zeros((1, max_placements), dtype=np.int8))
    d_row = cuda.to_device(np.zeros((1, max_placements), dtype=np.int8))
    d_spin = cuda.to_device(np.zeros((1, max_placements), dtype=np.int8))
    d_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    d_overflow = cuda.to_device(np.zeros(1, dtype=np.int32))

    enumerate_kernel[1, 1](
        1,
        d_parent,
        d_piece,
        d_carry,
        d_boards,
        board_height,
        d_rot,
        d_col,
        d_row,
        d_spin,
        d_count,
        d_overflow,
    )

    n = int(d_count.copy_to_host()[0])
    rot = d_rot.copy_to_host()[0, :n]
    col = d_col.copy_to_host()[0, :n]
    row = d_row.copy_to_host()[0, :n]
    spin = d_spin.copy_to_host()[0, :n]
    return rot, col, row, spin
