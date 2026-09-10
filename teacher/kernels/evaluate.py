"""Leaf board stats and the heuristic evaluation, ported from b2b_search.c."""

# The teacher import runs the CUDA library bootstrap and must precede numba.cuda.
from teacher.constants import (
    BOARD_COLS,
    BOARD_ROWS,
    DEATH_HEIGHT_CAP,
    DEATH_SCORE,
    HOLE_HEIGHT_SCALE,
    MAX_SEARCH_DEPTH,
    PIECE_N,
    RISK_TABLE,
    W_ATTACK_H,
    W_AVG_HEIGHT,
    W_B2B_FLAT,
    W_B2B_LINEAR,
    W_BUMPINESS,
    W_CHAIN,
    W_EXEC_RAMP,
    W_GARBAGE_PREVENT,
    W_HOLE_CEILING,
    W_HOLES,
    W_IMMOBILE_CLEAR,
    W_IMMOBILE_LINES,
)
from teacher.kernels.rules import (
    clear_lines,
    compute_col_heights,
    lock_piece,
    spawn_envelope_blocked,
)
from teacher.kernels.stats import (
    compute_hole_ceiling_weight,
    compute_reachability,
    count_hole_sections,
    count_immobile_placements,
    top_filled_row,
)

from numba import cuda

import math
import numpy as np

# Chain payout by affordable consecutive spin clears; b2b_search.c:428.
_W_CHAIN = np.asarray(W_CHAIN, dtype=np.float32)
_RISK_TABLE = np.asarray(RISK_TABLE, dtype=np.float32)

# Length of the upcoming-piece list; b2b_search.c:1663.
UPCOMING_CAP = MAX_SEARCH_DEPTH + 2

# Fields of the ImmobilePlacementResult scratch; b2b_search.c:1289.
_IPR_CLEARING = 0
_IPR_LINES = 1
_IPR_BEST_PT = 2
_IPR_BEST_ROT = 3
_IPR_BEST_R = 4
_IPR_BEST_C = 5


# Upcoming pieces
@cuda.jit(device=True, inline=True)
def build_upcoming(hold_piece, queue, queue_len, next_queue_idx, upcoming_out):
    """Fill upcoming_out with hold then the live queue, returning its length.

    b2b_search.c:1663.
    """
    num_upcoming = 0
    if hold_piece != PIECE_N:
        upcoming_out[num_upcoming] = hold_piece
        num_upcoming += 1
    i = next_queue_idx
    while i < queue_len and num_upcoming < UPCOMING_CAP:
        upcoming_out[num_upcoming] = queue[i]
        num_upcoming += 1
        i += 1
    return num_upcoming


# Greedy spin chain
@cuda.jit(device=True, inline=True)
def _greedy_chain_len(board, board_height, reach, upcoming, num_upcoming):
    """Consecutive spin clears the carry pair affords, max 4; b2b_search.c:1519."""
    sb = cuda.local.array(BOARD_ROWS, np.uint16)
    sreach = cuda.local.array(BOARD_ROWS, np.uint16)
    pair = cuda.local.array(2, np.int32)
    pq = cuda.local.array(8, np.int32)
    ir = cuda.local.array(7, np.float32)

    for r in range(board_height):
        sb[r] = board[r]
        sreach[r] = reach[r]

    carry = upcoming[0]
    idx = 1
    chain_len = 0
    while chain_len < 4:
        pair[0] = carry
        if idx < num_upcoming:
            pair[1] = upcoming[idx]
        else:
            pair[1] = PIECE_N
        if pair[1] == PIECE_N:
            n_pair = 1
        else:
            n_pair = 2
        for k in range(8):
            pq[k] = 0
        for k in range(n_pair):
            pq[pair[k]] += 1

        count_immobile_placements(sb, board_height, sreach, pair, n_pair, pq, ir)
        best_pt = np.int32(ir[_IPR_BEST_PT])
        if best_pt == PIECE_N:
            break

        lock_piece(
            sb,
            board_height,
            best_pt,
            np.int32(ir[_IPR_BEST_ROT]),
            np.int32(ir[_IPR_BEST_R]),
            np.int32(ir[_IPR_BEST_C]),
        )
        clear_lines(sb, board_height)
        chain_len += 1
        if best_pt == carry:
            carry = pair[1]
        idx += 1
        if carry == PIECE_N:
            break
        compute_reachability(sb, board_height, top_filled_row(sb, board_height), sreach)
    return chain_len


# Board stats
@cuda.jit(device=True, inline=True)
def compute_board_stats(
    board,
    board_height,
    upcoming,
    num_upcoming,
    height_hint,
    has_hint,
    col_heights_out,
    stats_out,
):
    """Write the board-shape stats the eval reads; b2b_search.c:1452.

    stats_out is (max_height, avg_height, holes, hole_ceiling_weight,
    immobile_clearing_placements, immobile_clearable_lines, bumpiness_exempted,
    chain_len).
    """
    # Column heights, cached hint when available, else scan.
    if has_hint != 0:
        for c in range(BOARD_COLS):
            col_heights_out[c] = height_hint[c]
    else:
        compute_col_heights(board, board_height, col_heights_out)

    # Max + average height.
    max_height = 0
    total_h = np.float32(0.0)
    for c in range(BOARD_COLS):
        h = col_heights_out[c]
        if h > max_height:
            max_height = h
        total_h += np.float32(h)
    avg_height = total_h / np.float32(BOARD_COLS)

    # Raw bumpiness.
    bumpiness = np.float32(0.0)
    for c in range(BOARD_COLS - 1):
        d = col_heights_out[c] - col_heights_out[c + 1]
        if d < 0:
            d = -d
        bumpiness += np.float32(d)

    top_filled = top_filled_row(board, board_height)

    reach = cuda.local.array(BOARD_ROWS, np.uint16)
    compute_reachability(board, board_height, top_filled, reach)

    # Immobile spin placements, weighted by queue position and capped by queue count.
    piece_queue_count = cuda.local.array(8, np.int32)
    for p in range(8):
        piece_queue_count[p] = 0
    for i in range(num_upcoming):
        pt = upcoming[i]
        if pt >= 0 and pt < 8:
            piece_queue_count[pt] += 1
    ipr = cuda.local.array(7, np.float32)
    count_immobile_placements(
        board, board_height, reach, upcoming, num_upcoming, piece_queue_count, ipr
    )
    stats_out[4] = ipr[_IPR_CLEARING]
    stats_out[5] = ipr[_IPR_LINES]

    # Hole metrics.
    stats_out[2] = np.float32(count_hole_sections(board, board_height, reach))
    stats_out[3] = np.float32(0.0)
    if W_HOLE_CEILING != 0.0:
        stats_out[3] = compute_hole_ceiling_weight(
            board, board_height, top_filled, reach
        )

    # Greedy ordered chain, skipped below two upcoming pieces.
    chain_len = 0
    if num_upcoming >= 2:
        chain_len = _greedy_chain_len(
            board, board_height, reach, upcoming, num_upcoming
        )

    # Deepest well column.
    well_col = -1
    well_depth = 0
    for c in range(BOARD_COLS):
        if c > 0:
            left_h = col_heights_out[c - 1]
        else:
            left_h = board_height
        if c < BOARD_COLS - 1:
            right_h = col_heights_out[c + 1]
        else:
            right_h = board_height
        if left_h < right_h:
            min_neighbor = left_h
        else:
            min_neighbor = right_h
        depth = min_neighbor - col_heights_out[c]
        if depth >= 2 and depth > well_depth:
            well_depth = depth
            well_col = c

    # Bumpiness, exempting the two adjacencies around the deepest well.
    bumpiness_exempted = bumpiness
    if well_col >= 0:
        if well_col > 0:
            d = col_heights_out[well_col - 1] - col_heights_out[well_col]
            if d < 0:
                d = -d
            bumpiness_exempted -= np.float32(d)
        if well_col < BOARD_COLS - 1:
            d = col_heights_out[well_col] - col_heights_out[well_col + 1]
            if d < 0:
                d = -d
            bumpiness_exempted -= np.float32(d)
        if bumpiness_exempted < np.float32(0.0):
            bumpiness_exempted = np.float32(0.0)

    stats_out[0] = np.float32(max_height)
    stats_out[1] = avg_height
    stats_out[6] = bumpiness_exempted
    stats_out[7] = np.float32(chain_len)


# Leaf evaluation
@cuda.jit(device=True, inline=True)
def evaluate_state(
    board,
    board_height,
    b2b,
    combo,
    total_attack,
    pieces_placed,
    chain_ramp,
    garbage_remaining,
    garbage_prevented,
    unlicensed_cash,
    parent_avg_height,
    unlicensed_cash_A,
    hold_piece,
    next_queue_idx,
    queue,
    queue_len,
    col_heights,
    has_hint,
):
    """Heuristic score of one leaf state; b2b_search.c:1659.

    Carries no transposition table, so it always computes.
    """
    score = np.float32(0.0)

    upcoming = cuda.local.array(UPCOMING_CAP, np.int32)
    num_upcoming = build_upcoming(
        hold_piece, queue, queue_len, next_queue_idx, upcoming
    )

    heights = cuda.local.array(BOARD_COLS, np.int8)
    stats = cuda.local.array(8, np.float32)
    compute_board_stats(
        board,
        board_height,
        upcoming,
        num_upcoming,
        col_heights,
        has_hint,
        heights,
        stats,
    )

    effective_h = np.int32(stats[0]) + garbage_remaining

    # Instant death, spawn box blocked or a column at the height cap.
    if spawn_envelope_blocked(board) or effective_h >= DEATH_HEIGHT_CAP:
        return np.float32(DEATH_SCORE)

    # Height risk and volume
    score -= _RISK_TABLE[effective_h]
    if unlicensed_cash != 0:
        avg_h = np.float32(parent_avg_height)
    else:
        avg_h = stats[1]
    score -= np.float32(W_AVG_HEIGHT) * avg_h
    score -= np.float32(W_BUMPINESS) * stats[6]

    # Hole accounting
    holes = np.int32(stats[2])
    if holes > 0:
        hole_mult = np.float32(1.0) + np.float32(effective_h) / np.float32(
            HOLE_HEIGHT_SCALE
        )
        score -= np.float32(W_HOLES) * (np.float32(holes) * hole_mult)
    if stats[3] > np.float32(0.0):
        score -= np.float32(W_HOLE_CEILING) * stats[3]

    # B2B economy
    if b2b >= 0:
        score += np.float32(W_B2B_FLAT)
    if b2b > 0:
        score += np.float32(W_B2B_LINEAR) * np.float32(b2b)

    # Attack realization
    atk_real = np.float32(total_attack) - np.float32(unlicensed_cash_A)
    if atk_real < np.float32(0.0):
        atk_real = np.float32(0.0)
    if pieces_placed > 0:
        score += (np.float32(W_ATTACK_H) * atk_real) / np.float32(pieces_placed)
    if garbage_prevented > np.float32(0.0):
        score += np.float32(W_GARBAGE_PREVENT) * np.float32(garbage_prevented)

    # Spin-setup structure
    if stats[4] > np.float32(0.0):
        line_reward = np.float32(W_IMMOBILE_CLEAR) * np.float32(math.sqrt(stats[4]))
        line_reward += np.float32(W_IMMOBILE_LINES) * stats[5]
        score += line_reward
    score += _W_CHAIN[np.int32(stats[7])]
    if pieces_placed > 0:
        score += (np.float32(W_EXEC_RAMP) * np.float32(chain_ramp)) / np.float32(
            pieces_placed
        )

    return score


# Kernel
@cuda.jit
def evaluate_kernel(
    n,
    board,
    board_height,
    b2b,
    combo,
    total_attack,
    pieces_placed,
    chain_ramp,
    garbage_remaining,
    garbage_prevented,
    unlicensed_cash,
    parent_avg_height,
    unlicensed_cash_A,
    hold_piece,
    next_queue_idx,
    col_heights,
    dead,
    game,
    queues,
    queue_len,
    score,
):
    """Score every state in the bank, one thread per state."""
    i = cuda.grid(1)
    if i >= n:
        return
    if dead[i] != 0:
        score[i] = np.float32(DEATH_SCORE)
        return
    g = game[i]
    score[i] = evaluate_state(
        board[i],
        board_height,
        b2b[i],
        combo[i],
        total_attack[i],
        pieces_placed[i],
        chain_ramp[i],
        garbage_remaining[i],
        garbage_prevented[i],
        unlicensed_cash[i],
        parent_avg_height[i],
        unlicensed_cash_A[i],
        hold_piece[i],
        next_queue_idx[i],
        queues[g],
        queue_len[g],
        col_heights[i],
        1,
    )


def debug_evaluate(
    board_masks,
    queue,
    hold_piece=0,
    b2b=-1,
    combo=-1,
    total_attack=0.0,
    pieces_placed=1,
    chain_ramp=0,
    garbage_remaining=0,
    garbage_prevented=0.0,
    unlicensed_cash=0,
    parent_avg_height=0.0,
    unlicensed_cash_A=0.0,
    next_queue_idx=0,
    board_height=40,
):
    """Run evaluate_kernel on one host board and return its score as a float."""
    masks = np.asarray(board_masks, dtype=np.uint16).ravel()
    board = np.zeros((1, BOARD_ROWS), dtype=np.uint16)
    board[0, : masks.shape[0]] = masks

    heights = np.zeros((1, BOARD_COLS), dtype=np.int8)
    for c in range(BOARD_COLS):
        for r in range(board_height):
            if board[0, r] & (1 << c):
                heights[0, c] = board_height - r
                break

    q = np.asarray(queue, dtype=np.int32).ravel()
    queues = np.zeros((1, max(q.shape[0], 1)), dtype=np.int32)
    queues[0, : q.shape[0]] = q
    score = np.zeros(1, dtype=np.float32)

    evaluate_kernel[1, 32](
        1,
        board,
        board_height,
        np.full(1, b2b, dtype=np.int32),
        np.full(1, combo, dtype=np.int32),
        np.full(1, total_attack, dtype=np.float32),
        np.full(1, pieces_placed, dtype=np.int32),
        np.full(1, chain_ramp, dtype=np.int32),
        np.full(1, garbage_remaining, dtype=np.int32),
        np.full(1, garbage_prevented, dtype=np.float32),
        np.full(1, unlicensed_cash, dtype=np.uint8),
        np.full(1, parent_avg_height, dtype=np.float32),
        np.full(1, unlicensed_cash_A, dtype=np.float32),
        np.full(1, hold_piece, dtype=np.int32),
        np.full(1, next_queue_idx, dtype=np.int32),
        heights,
        np.zeros(1, dtype=np.uint8),
        np.zeros(1, dtype=np.int32),
        queues,
        np.full(1, q.shape[0], dtype=np.int32),
        score,
    )
    return float(score[0])


@cuda.jit
def _board_stats_kernel(
    n,
    board,
    board_height,
    hold_piece,
    next_queue_idx,
    queues,
    queue_len,
    col_heights_out,
    stats_out,
):
    """Write compute_board_stats output for each state into stats_out."""
    i = cuda.grid(1)
    if i >= n:
        return
    upcoming = cuda.local.array(18, np.int32)
    num_upcoming = build_upcoming(
        hold_piece[i], queues[i], queue_len[i], next_queue_idx[i], upcoming
    )
    hint = cuda.local.array(10, np.int8)
    compute_board_stats(
        board[i],
        board_height,
        upcoming,
        num_upcoming,
        hint,
        0,
        col_heights_out[i],
        stats_out[i],
    )


def debug_board_stats(
    board_masks, queue, hold_piece=0, next_queue_idx=0, board_height=40
):
    """Board stats of one host board, as a numpy float32 array of 8 values."""
    masks = np.asarray(board_masks, dtype=np.uint16).ravel()
    board = np.zeros((1, BOARD_ROWS), dtype=np.uint16)
    board[0, : masks.shape[0]] = masks
    q = np.asarray(queue, dtype=np.int32).ravel()
    queues = np.zeros((1, max(q.shape[0], 1)), dtype=np.int32)
    queues[0, : q.shape[0]] = q
    heights = np.zeros((1, BOARD_COLS), dtype=np.int8)
    stats = np.zeros((1, 8), dtype=np.float32)
    _board_stats_kernel[1, 32](
        1,
        board,
        board_height,
        np.full(1, hold_piece, dtype=np.int32),
        np.full(1, next_queue_idx, dtype=np.int32),
        queues,
        np.full(1, q.shape[0], dtype=np.int32),
        heights,
        stats,
    )
    return stats[0]
