"""Board geometry device functions the leaf evaluation reads, from b2b_search.c."""

# The teacher import runs the CUDA library bootstrap and must precede numba.cuda.
from teacher.constants import (
    BOARD_COLS,
    BOARD_ROWS,
    FULL_ROW,
    PIECE_I,
    PIECE_N,
    PIECE_O,
    PIECE_T,
    PIECE_Z,
    ROTATIONS,
    SPIN_NONE,
)
from teacher.kernels.rules import check_immobility, collides, detect_t_spin
from teacher.tables import (
    KICKS,
    PIECE_MAX_COL,
    PIECE_MAX_ROW,
    PIECE_MIN_COL,
    PIECE_MIN_ROW,
    PIECE_ROW_MASKS,
)

from numba import cuda

import numpy as np

# Rotation deltas tried by the kick walk; b2b_search.c:1256.
_KICK_DELTAS = np.array((1, 3, 2), dtype=np.int32)

# Sweep cap for both flood fills, one sweep per board cell.
_FILL_SWEEPS = BOARD_ROWS * BOARD_COLS


# Flood fill
@cuda.jit(device=True, inline=True)
def _row_spread(cells, empty):
    """Cells grown left and right through empty until saturated."""
    x = cells
    for _ in range(BOARD_COLS):
        grown = np.uint16((x | (x << 1) | (x >> 1)) & empty)
        if grown == x:
            break
        x = grown
    return x


@cuda.jit(device=True, inline=True)
def _spread_into(cells, up, down, empty):
    """Cells merged with the rows above and below, saturated within empty."""
    return _row_spread(np.uint16((cells | up | down) & empty), empty)


# Board scans
@cuda.jit(device=True, inline=True)
def top_filled_row(board, board_height):
    """First non-empty row, board_height when the board is empty; b2b_search.c:1316."""
    for r in range(board_height):
        if board[r] != 0:
            return r
    return board_height


@cuda.jit(device=True, inline=True)
def compute_reachability(board, board_height, top_filled, reach):
    """Fill reach with the empty cells joined to the top row; b2b_search.c:1079."""
    for r in range(top_filled):
        reach[r] = np.uint16(FULL_ROW)
    for r in range(top_filled, board_height):
        reach[r] = np.uint16(0)
    if top_filled >= board_height:
        return

    # Seed every empty cell of the first non-empty row; b2b_search.c:1090.
    reach[top_filled] = np.uint16((~board[top_filled]) & FULL_ROW)

    for _ in range(_FILL_SWEEPS):
        changed = False
        for r in range(top_filled, board_height):
            empty = np.uint16((~board[r]) & FULL_ROW)
            up = np.uint16(0)
            if r > top_filled:
                up = reach[r - 1]
            down = np.uint16(0)
            if r + 1 < board_height:
                down = reach[r + 1]
            x = _spread_into(reach[r], up, down, empty)
            if x != reach[r]:
                reach[r] = x
                changed = True
        for r in range(board_height - 2, top_filled - 1, -1):
            empty = np.uint16((~board[r]) & FULL_ROW)
            up = np.uint16(0)
            if r > top_filled:
                up = reach[r - 1]
            x = _spread_into(reach[r], up, reach[r + 1], empty)
            if x != reach[r]:
                reach[r] = x
                changed = True
        if not changed:
            break


@cuda.jit(device=True, inline=True)
def count_hole_sections(board, board_height, reach):
    """Count of 4-connected enclosed hole components; b2b_search.c:1121."""
    rem = cuda.local.array(BOARD_ROWS, np.uint16)
    comp = cuda.local.array(BOARD_ROWS, np.uint16)
    for r in range(board_height):
        empty = np.uint16((~board[r]) & FULL_ROW)
        rem[r] = np.uint16(empty & (~reach[r]))

    sections = 0
    for r in range(board_height):
        while rem[r] != 0:
            sections += 1
            for k in range(r, board_height):
                comp[k] = np.uint16(0)
            # Lowest set bit of the row, the C's __builtin_ctz column.
            comp[r] = np.uint16(rem[r] ^ (rem[r] & (rem[r] - 1)))

            for _ in range(_FILL_SWEEPS):
                changed = False
                for k in range(r, board_height):
                    up = np.uint16(0)
                    if k > r:
                        up = comp[k - 1]
                    down = np.uint16(0)
                    if k + 1 < board_height:
                        down = comp[k + 1]
                    x = _spread_into(comp[k], up, down, rem[k])
                    if x != comp[k]:
                        comp[k] = x
                        changed = True
                for k in range(board_height - 2, r - 1, -1):
                    up = np.uint16(0)
                    if k > r:
                        up = comp[k - 1]
                    x = _spread_into(comp[k], up, comp[k + 1], rem[k])
                    if x != comp[k]:
                        comp[k] = x
                        changed = True
                if not changed:
                    break

            for k in range(r, board_height):
                rem[k] = np.uint16(rem[k] & (~comp[k]))
    return sections


@cuda.jit(device=True, inline=True)
def compute_hole_ceiling_weight(board, board_height, top_filled, reach):
    """Filled cells over enclosed holes, weighted by height; b2b_search.c:1192."""
    total = np.int32(0)
    for c in range(BOARD_COLS):
        bit = np.uint16(1) << c
        filled_above = 0
        for r in range(top_filled, board_height):
            if (board[r] & bit) != 0:
                filled_above += 1
            else:
                enclosed = (reach[r] & bit) == 0
                if enclosed and filled_above > 0:
                    hole_height = board_height - r
                    total += np.int32(filled_above * (board_height + hole_height))
    return np.float32(total) / np.float32(board_height)


# Spin placements
@cuda.jit(device=True, inline=True)
def _piece_cells(mask, c):
    """Piece row mask at column c, truncated to uint16; b2b_search.c:1370.

    A column below 0 yields an empty mask.
    """
    if c < 0:
        return np.uint16(0)
    return np.uint16(mask << c)


@cuda.jit(device=True, inline=True)
def clear_sky(board, piece, rot, r, c):
    """Whether every piece cell has empty sky above it; b2b_search.c:1240."""
    for i in range(4):
        mask = PIECE_ROW_MASKS[piece, rot, i]
        if mask == 0:
            continue
        cells = _piece_cells(mask, c)
        for row in range(r + i):
            if (board[row] & cells) != 0:
                return False
    return True


@cuda.jit(device=True, inline=True)
def kick_reachable(board, board_height, piece, rot, r, c):
    """Whether a rotation lands the piece on (rot, r, c); b2b_search.c:1254."""
    for d in range(3):
        delta = _KICK_DELTAS[d]
        from_rot = (rot + 4 - delta) % 4
        if (
            not collides(board, board_height, piece, from_rot, r, c)
            and collides(board, board_height, piece, from_rot, r + 1, c)
            and clear_sky(board, piece, from_rot, r, c)
        ):
            return True

        count = 5 if delta == 2 else 4
        for k in range(count):
            kdr = KICKS[piece, from_rot, rot, k, 0]
            kdc = KICKS[piece, from_rot, rot, k, 1]
            if kdr == 0 and kdc == 0 and count == 5:
                continue
            pr = r - kdr
            pc = c - kdc
            if collides(board, board_height, piece, from_rot, pr, pc):
                continue
            if not collides(board, board_height, piece, rot, pr, pc):
                continue
            earlier_fits = False
            for j in range(k):
                jr = KICKS[piece, from_rot, rot, j, 0]
                jc = KICKS[piece, from_rot, rot, j, 1]
                if jr == 0 and jc == 0 and count == 5:
                    continue
                if not collides(board, board_height, piece, rot, pr + jr, pc + jc):
                    earlier_fits = True
                    break
            if earlier_fits:
                continue
            if not collides(board, board_height, piece, from_rot, pr + 1, pc):
                continue
            if clear_sky(board, piece, from_rot, pr, pc):
                return True
    return False


@cuda.jit(device=True, inline=True)
def count_immobile_placements(
    board, board_height, reach, upcoming, num_upcoming, piece_queue_count, out
):
    """Queue-weighted spin-clear placements and the best one; b2b_search.c:1298.

    out is float32 (clearing, lines, best_pt, best_rot, best_r, best_c, best_lines).
    """
    for i in range(7):
        out[i] = np.float32(0.0)
    out[2] = np.float32(PIECE_N)
    if num_upcoming <= 0:
        return

    top_filled = top_filled_row(board, board_height)
    scan_start = top_filled - 3
    if scan_start < 0:
        scan_start = 0
    scan_end = top_filled + 5
    if scan_end > board_height:
        scan_end = board_height

    # Best queue position of each piece type, weight 1 / (position + 1); piece_pos
    # keeps the position for the exact division in the weighted sums.
    piece_weight = cuda.local.array(8, np.float32)
    piece_pos = cuda.local.array(8, np.int32)
    for p in range(8):
        piece_weight[p] = np.float32(0.0)
        piece_pos[p] = 0
    for i in range(num_upcoming):
        pt = upcoming[i]
        if pt < PIECE_I or pt > PIECE_Z:
            continue
        w = np.float32(1.0) / np.float32(i + 1)
        if w > piece_weight[pt]:
            piece_weight[pt] = w
            piece_pos[pt] = i

    total_clearing = np.float32(0.0)
    total_lines = np.float32(0.0)
    best_pt = PIECE_N
    best_rot = 0
    best_r = 0
    best_c = 0
    best_lines = 0
    best_w = np.float32(0.0)

    for pt in range(PIECE_I, PIECE_Z + 1):
        if piece_weight[pt] <= np.float32(0.0):
            continue
        if pt == PIECE_O:
            continue
        w = piece_weight[pt]
        clearing_this_piece = 0
        lines_this_piece = 0

        for rot in range(ROTATIONS):
            min_row = PIECE_MIN_ROW[pt, rot]
            max_row = PIECE_MAX_ROW[pt, rot]
            c_lo = -PIECE_MIN_COL[pt, rot]
            c_hi = BOARD_COLS - PIECE_MAX_COL[pt, rot]

            for r in range(scan_start, scan_end):
                if r + min_row < 0:
                    continue
                if r + max_row >= board_height:
                    break

                for c in range(c_lo, c_hi):
                    fits = True
                    for i in range(4):
                        mask = PIECE_ROW_MASKS[pt, rot, i]
                        if mask == 0:
                            continue
                        if (board[r + i] & _piece_cells(mask, c)) != 0:
                            fits = False
                            break
                    if not fits:
                        continue

                    any_reachable = False
                    for i in range(4):
                        mask = PIECE_ROW_MASKS[pt, rot, i]
                        if mask == 0:
                            continue
                        if (reach[r + i] & _piece_cells(mask, c)) != 0:
                            any_reachable = True
                            break
                    if not any_reachable:
                        continue

                    if not collides(board, board_height, pt, rot, r + 1, c):
                        continue
                    if pt == PIECE_T:
                        spin = detect_t_spin(board, board_height, r, c, rot, 0)
                        if spin == SPIN_NONE:
                            continue
                    else:
                        if not check_immobility(board, board_height, pt, rot, r, c):
                            continue
                    if not kick_reachable(board, board_height, pt, rot, r, c):
                        continue

                    lines = 0
                    for i in range(4):
                        mask = PIECE_ROW_MASKS[pt, rot, i]
                        if mask == 0:
                            continue
                        combined = board[r + i] | _piece_cells(mask, c)
                        if (combined & FULL_ROW) == FULL_ROW:
                            lines += 1

                    if lines > 0:
                        clearing_this_piece += 1
                        lines_this_piece += lines
                        if lines > best_lines or (lines == best_lines and w > best_w):
                            best_pt = pt
                            best_rot = rot
                            best_r = r
                            best_c = c
                            best_lines = lines
                            best_w = w

        qc = piece_queue_count[pt]
        if qc <= 0:
            qc = 1
        capped_clearing = clearing_this_piece if clearing_this_piece < qc else qc
        lines_cap = qc * 4
        capped_lines = lines_this_piece if lines_this_piece < lines_cap else lines_cap
        total_clearing += np.float32(capped_clearing) / np.float32(piece_pos[pt] + 1)
        total_lines += np.float32(capped_lines) / np.float32(piece_pos[pt] + 1)

    out[0] = total_clearing
    out[1] = total_lines
    out[2] = np.float32(best_pt)
    out[3] = np.float32(best_rot)
    out[4] = np.float32(best_r)
    out[5] = np.float32(best_c)
    out[6] = np.float32(best_lines)
