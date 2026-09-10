"""Child construction kernel, a port of expand_and_insert; b2b_search.c:1792."""

# The teacher import runs the CUDA library bootstrap and must precede numba.cuda.
from teacher.buffers import StateBank
from teacher.constants import BOARD_COLS, BOARD_ROWS
from teacher.kernels import rules

from numba import cuda

import cupy as cp
import numpy as np

THREADS_PER_BLOCK = 128


@cuda.jit
def expand_kernel(
    n_children,
    child_item,
    child_slot,
    wi_parent,
    wi_piece,
    wi_new_hold,
    wi_new_qi,
    wi_new_bag,
    wi_is_hold,
    wi_game,
    wi_carry,
    pl_rot,
    pl_col,
    pl_row,
    pl_spin,
    p_board,
    p_col_heights,
    p_b2b,
    p_combo,
    p_total_attack,
    p_pieces_placed,
    p_rows_cleared,
    p_chain_ramp,
    p_hold_piece,
    p_next_queue_idx,
    p_depth0_idx,
    p_garbage_remaining,
    p_garbage_timer,
    p_garbage_prevented,
    p_bag_seen,
    p_unlicensed_cash,
    p_parent_avg_height,
    p_unlicensed_cash_A,
    p_sort_hash,
    p_game,
    p_dead,
    c_board,
    c_col_heights,
    c_b2b,
    c_combo,
    c_total_attack,
    c_pieces_placed,
    c_rows_cleared,
    c_chain_ramp,
    c_hold_piece,
    c_next_queue_idx,
    c_depth0_idx,
    c_garbage_remaining,
    c_garbage_timer,
    c_garbage_prevented,
    c_bag_seen,
    c_unlicensed_cash,
    c_parent_avg_height,
    c_unlicensed_cash_A,
    c_sort_hash,
    c_game,
    c_dead,
    queues,
    queue_len,
    board_height,
    depth,
    root_index,
):
    """Build one child per thread from a parent and one of its placements.

    Thread k reads work item child_item[k], placement child_slot[k] within it,
    and writes child bank row k. root_index is read only when depth is 0. A work
    item with wi_carry set takes exactly one child, a verbatim parent copy.
    """
    k = cuda.grid(1)
    if k >= n_children:
        return

    wi = child_item[k]
    pi = wi_parent[wi]

    # Carry: the parent enters the next beam unchanged; b2b_search.c:2288-2292.
    if wi_carry[wi] != 0:
        parent_board = p_board[pi]
        board = c_board[k]
        for r in range(board_height):
            board[r] = parent_board[r]
        parent_heights = p_col_heights[pi]
        heights = c_col_heights[k]
        for c in range(BOARD_COLS):
            heights[c] = parent_heights[c]
        c_b2b[k] = p_b2b[pi]
        c_combo[k] = p_combo[pi]
        c_total_attack[k] = p_total_attack[pi]
        c_pieces_placed[k] = p_pieces_placed[pi]
        c_rows_cleared[k] = p_rows_cleared[pi]
        c_chain_ramp[k] = p_chain_ramp[pi]
        c_hold_piece[k] = p_hold_piece[pi]
        c_next_queue_idx[k] = p_next_queue_idx[pi]
        c_depth0_idx[k] = p_depth0_idx[pi]
        c_garbage_remaining[k] = p_garbage_remaining[pi]
        c_garbage_timer[k] = p_garbage_timer[pi]
        c_garbage_prevented[k] = p_garbage_prevented[pi]
        c_bag_seen[k] = p_bag_seen[pi]
        c_unlicensed_cash[k] = p_unlicensed_cash[pi]
        c_parent_avg_height[k] = p_parent_avg_height[pi]
        c_unlicensed_cash_A[k] = p_unlicensed_cash_A[pi]
        c_sort_hash[k] = p_sort_hash[pi]
        c_game[k] = p_game[pi]
        c_dead[k] = p_dead[pi]
        return

    slot = child_slot[k]
    piece = np.int32(wi_piece[wi])
    rot = np.int32(pl_rot[wi, slot])
    col = np.int32(pl_col[wi, slot])
    land_r = np.int32(pl_row[wi, slot])
    spin = np.int32(pl_spin[wi, slot])
    parent_b2b = p_b2b[pi]
    parent_combo = p_combo[pi]

    # Board: parent copy, lock, clear, perfect-clear scan
    parent_board = p_board[pi]
    board = c_board[k]
    for r in range(board_height):
        board[r] = parent_board[r]
    rules.lock_piece(board, board_height, piece, rot, land_r, col)
    clears = rules.clear_lines(board, board_height)
    perfect_clear = rules.is_perfect_clear(board, board_height)

    # Attack
    ar = cuda.local.array(5, np.float32)
    rules.compute_attack(clears, spin, parent_b2b, parent_combo, perfect_clear, ar)
    attack = ar[0]
    new_b2b = np.int32(ar[1])
    new_combo = np.int32(ar[2])
    b2b_maintaining = ar[3] != np.float32(0.0)
    new_hold = wi_new_hold[wi]
    new_qi = wi_new_qi[wi]
    new_bag = wi_new_bag[wi]

    # Path accumulators; b2b_search.c:1816-1824
    ramp = 0
    if clears > 0 and b2b_maintaining and parent_combo >= 0:
        ramp = parent_combo + 1
    c_b2b[k] = new_b2b
    c_combo[k] = new_combo
    c_total_attack[k] = p_total_attack[pi] + attack
    c_pieces_placed[k] = p_pieces_placed[pi] + 1
    c_rows_cleared[k] = p_rows_cleared[pi] + clears
    c_chain_ramp[k] = p_chain_ramp[pi] + ramp
    c_hold_piece[k] = new_hold
    c_next_queue_idx[k] = new_qi
    c_bag_seen[k] = new_bag
    c_game[k] = wi_game[wi]
    if depth == 0:
        c_depth0_idx[k] = root_index[k]
    else:
        c_depth0_idx[k] = p_depth0_idx[pi]

    # Garbage cancel, push and prevention; b2b_search.c:1825-1855
    gr = p_garbage_remaining[pi]
    gt = p_garbage_timer[pi]
    gr_initial = gr
    was_imminent = gt <= 0 and gr_initial > 0
    cancelled = 0
    pushed_garbage = False
    if attack > np.float32(0.0) and gr > 0:
        whole = np.int32(attack)
        if whole > gr:
            cancelled = gr
        else:
            cancelled = whole
        gr -= cancelled
    if clears == 0 and gr > 0:
        if gt <= 0:
            rules.push_garbage(board, board_height, gr)
            gr = 0
            pushed_garbage = True
        else:
            gt -= 1
    prevented = 0
    if was_imminent:
        if pushed_garbage:
            prevented = cancelled
        else:
            prevented = gr_initial
    c_garbage_remaining[k] = gr
    c_garbage_timer[k] = gt
    c_garbage_prevented[k] = p_garbage_prevented[pi] + np.float32(prevented)

    # Heights: patch the parent skyline only when the board just gained the piece
    heights = c_col_heights[k]
    if clears == 0 and not pushed_garbage:
        rules.patch_col_heights(
            p_col_heights[pi], piece, rot, land_r, col, board_height, heights
        )
    else:
        rules.compute_col_heights(board, board_height, heights)

    # Unlicensed cash; b2b_search.c:172
    cash = rules.latch_unlicensed_cash(clears, spin, perfect_clear, parent_combo)
    c_unlicensed_cash[k] = np.uint8(cash)
    c_parent_avg_height[k] = rules.mean_col_heights(p_col_heights[pi])
    if cash != 0:
        c_unlicensed_cash_A[k] = attack
    else:
        c_unlicensed_cash_A[k] = np.float32(0.0)

    if rules.placement_is_dead(board, heights, gr):
        c_dead[k] = np.uint8(1)
    else:
        c_dead[k] = np.uint8(0)

    c_sort_hash[k] = rules.state_hash(
        board, board_height, new_b2b, new_combo, new_hold, new_qi, new_bag, gr, gt
    )


# Host helpers
_DEBUG_SCALARS = (
    "b2b",
    "combo",
    "total_attack",
    "pieces_placed",
    "rows_cleared",
    "chain_ramp",
    "hold_piece",
    "next_queue_idx",
    "depth0_idx",
    "garbage_remaining",
    "garbage_timer",
    "garbage_prevented",
    "bag_seen",
    "unlicensed_cash",
    "parent_avg_height",
    "unlicensed_cash_A",
    "sort_hash",
    "game",
    "dead",
)


def _host_col_heights(board: np.ndarray) -> np.ndarray:
    """Per-column heights of one host board; b2b_search.c:153."""
    bit = (np.uint16(1) << np.arange(BOARD_COLS, dtype=np.uint16)).astype(np.uint16)
    filled = (board[:, None] & bit) != 0
    top = filled.argmax(axis=0)
    return np.where(filled.any(axis=0), BOARD_ROWS - top, 0).astype(np.int8)


def debug_expand(
    board_masks,
    piece,
    rot,
    col,
    landing_row,
    spin,
    b2b,
    combo,
    total_garbage,
    garbage_timer=0,
    parent_col_heights=None,
    hold=0,
    next_queue_idx=0,
    carry=0,
):
    """Expand one placement of a fresh root and return the child state as numpy."""
    board = np.ascontiguousarray(board_masks, dtype=np.uint16).reshape(BOARD_ROWS)
    if parent_col_heights is None:
        parent_col_heights = _host_col_heights(board)
    heights = np.ascontiguousarray(parent_col_heights, dtype=np.int8)
    heights = heights.reshape(BOARD_COLS)

    child_item = cp.zeros(1, dtype=cp.int32)
    child_slot = cp.zeros(1, dtype=cp.int32)
    root_index = cp.zeros(1, dtype=cp.int32)

    wi_parent = cp.zeros(1, dtype=cp.int32)
    wi_piece = cp.full(1, int(piece), dtype=cp.int32)
    wi_new_hold = cp.full(1, int(hold), dtype=cp.int32)
    wi_new_qi = cp.full(1, int(next_queue_idx), dtype=cp.int32)
    wi_new_bag = cp.zeros(1, dtype=cp.uint8)
    wi_is_hold = cp.zeros(1, dtype=cp.int32)
    wi_game = cp.zeros(1, dtype=cp.int32)
    wi_carry = cp.full(1, int(carry), dtype=cp.uint8)

    pl_rot = cp.full((1, 1), int(rot), dtype=cp.int8)
    pl_col = cp.full((1, 1), int(col), dtype=cp.int8)
    pl_row = cp.full((1, 1), int(landing_row), dtype=cp.int8)
    pl_spin = cp.full((1, 1), int(spin), dtype=cp.int8)

    # Parent: the zero-path root the C seeds at depth 0; b2b_search.c:2003-2016
    parent = StateBank(1)
    parent.board[0] = cp.asarray(board)
    parent.col_heights[0] = cp.asarray(heights)
    parent.b2b[0] = int(b2b)
    parent.combo[0] = int(combo)
    parent.hold_piece[0] = int(hold)
    parent.next_queue_idx[0] = int(next_queue_idx)
    parent.depth0_idx[0] = -1
    parent.garbage_remaining[0] = int(total_garbage)
    parent.garbage_timer[0] = int(garbage_timer)

    child = StateBank(1)
    queues = cp.zeros((1, 1), dtype=cp.int32)
    queue_len = cp.zeros(1, dtype=cp.int32)

    expand_kernel[1, THREADS_PER_BLOCK](
        1,
        child_item,
        child_slot,
        wi_parent,
        wi_piece,
        wi_new_hold,
        wi_new_qi,
        wi_new_bag,
        wi_is_hold,
        wi_game,
        wi_carry,
        pl_rot,
        pl_col,
        pl_row,
        pl_spin,
        parent.board,
        parent.col_heights,
        parent.b2b,
        parent.combo,
        parent.total_attack,
        parent.pieces_placed,
        parent.rows_cleared,
        parent.chain_ramp,
        parent.hold_piece,
        parent.next_queue_idx,
        parent.depth0_idx,
        parent.garbage_remaining,
        parent.garbage_timer,
        parent.garbage_prevented,
        parent.bag_seen,
        parent.unlicensed_cash,
        parent.parent_avg_height,
        parent.unlicensed_cash_A,
        parent.sort_hash,
        parent.game,
        parent.dead,
        child.board,
        child.col_heights,
        child.b2b,
        child.combo,
        child.total_attack,
        child.pieces_placed,
        child.rows_cleared,
        child.chain_ramp,
        child.hold_piece,
        child.next_queue_idx,
        child.depth0_idx,
        child.garbage_remaining,
        child.garbage_timer,
        child.garbage_prevented,
        child.bag_seen,
        child.unlicensed_cash,
        child.parent_avg_height,
        child.unlicensed_cash_A,
        child.sort_hash,
        child.game,
        child.dead,
        queues,
        queue_len,
        BOARD_ROWS,
        0,
        root_index,
    )
    cuda.synchronize()

    out = {
        "board": cp.asnumpy(child.board[0]),
        "col_heights": cp.asnumpy(child.col_heights[0]),
    }
    for name in _DEBUG_SCALARS:
        out[name] = cp.asnumpy(getattr(child, name))[0]
    return out
