"""Host driver for the batched beam search; b2b_search.c:1986-2435."""

# The teacher import runs the CUDA library bootstrap and must come first.
from teacher.constants import (
    DEATH_SCORE,
    MAX_BEAM_WIDTH,
    MAX_SEARCH_DEPTH,
    ROOT_FLOOR,
    ROOT_SCORE_INIT,
)
from teacher.kernels.enumerate import enumerate_kernel
from teacher.kernels.evaluate import evaluate_kernel
from teacher.kernels.expand import expand_kernel
from teacher.kernels.finalize import (
    count_branches_kernel,
    emit_items_kernel,
    gather_next_beam,
    frontier_range_kernel,
    raise_root_norm_kernel,
    record_fallback_kernel,
    seed_roots_kernel,
    select_next_beam,
)

import time
from typing import NamedTuple

import cupy as cp
import numpy as np
from numba import cuda

THREADS = 128


class BatchResult(NamedTuple):
    """One batched search result, every array indexed by game."""

    action: np.ndarray
    best_score: np.ndarray
    root_index: np.ndarray
    root_action: np.ndarray
    root_score: np.ndarray
    root_row: np.ndarray
    root_count: np.ndarray
    alive: np.ndarray
    placement_overflow: bool
    pool_overflow: bool
    workitem_overflow: bool


def _blocks(n):
    """Blocks of THREADS threads covering n items."""
    return max((int(n) + THREADS - 1) // THREADS, 1)


# Per-depth stages
def _emit_work_items(buf, depth, n_parents):
    """Write this depth's work items, returning their count and parent scan."""
    p = buf.curr
    branches = cp.zeros(n_parents, dtype=cp.int32)
    count_branches_kernel[_blocks(n_parents), THREADS](
        n_parents,
        depth,
        p.next_queue_idx,
        p.hold_piece,
        p.bag_seen,
        p.game,
        buf.active_piece,
        buf.queue_len,
        branches,
    )
    scan = cp.cumsum(branches, dtype=cp.int32)
    offsets = scan - branches

    n_items = int(scan[-1])
    if n_items > buf.workitem_capacity:
        buf.workitem_overflow.fill(1)
        n_items = buf.workitem_capacity
    buf.wi_total.fill(n_items)
    if n_items == 0:
        return 0, offsets

    emit_items_kernel[_blocks(n_parents), THREADS](
        n_parents,
        depth,
        offsets,
        p.next_queue_idx,
        p.hold_piece,
        p.bag_seen,
        p.game,
        buf.active_piece,
        buf.queues,
        buf.queue_len,
        buf.wi_parent,
        buf.wi_piece,
        buf.wi_new_hold,
        buf.wi_new_qi,
        buf.wi_new_bag,
        buf.wi_is_hold,
        buf.wi_game,
        buf.wi_carry,
    )
    return n_items, offsets


def _enumerate(buf, n_items):
    """Fill the placement arrays and wi_count for every work item."""
    enumerate_kernel[_blocks(n_items), THREADS](
        n_items,
        buf.wi_parent,
        buf.wi_piece,
        buf.wi_carry,
        buf.curr.board,
        buf.board_height,
        buf.pl_rot,
        buf.pl_col,
        buf.pl_row,
        buf.pl_spin,
        buf.wi_count,
        buf.placement_overflow,
    )


def _child_index(buf, n_items):
    """Child count with its (work item, placement slot) pair per child.

    A carry item takes one child whatever its placement count holds.
    """
    per_item = cp.where(
        buf.wi_carry[:n_items] != 0, np.int32(1), buf.wi_count[:n_items]
    ).astype(cp.int32, copy=False)
    scan = cp.cumsum(per_item, dtype=cp.int32)
    item_base = scan - per_item

    n_children = int(scan[-1])
    if n_children > buf.state_capacity:
        buf.pool_overflow.fill(1)
        n_children = buf.state_capacity
    buf.child_total.fill(n_children)
    if n_children == 0:
        return 0, None, None, item_base

    flat = cp.arange(n_children, dtype=cp.int32)
    child_item = cp.searchsorted(scan, flat, side="right").astype(cp.int32)
    child_slot = flat - item_base[child_item]
    return n_children, child_item, child_slot, item_base


def _seed_roots(buf, n_items, offsets, item_base):
    """Write the depth-0 roots and fallback, returning each item's root base.

    Parent i is game i at depth 0, so offsets[g] is game g's first work item.
    """
    root_base = item_base - item_base[offsets][buf.wi_game[:n_items]]
    plmax = buf.max_placements
    seed_roots_kernel[_blocks(n_items * plmax), THREADS](
        n_items,
        buf.wi_game,
        buf.wi_piece,
        buf.wi_is_hold,
        buf.wi_count,
        root_base,
        buf.pl_rot,
        buf.pl_col,
        buf.pl_row,
        buf.pl_spin,
        buf.root_action,
        buf.root_row,
        buf.root_count,
    )
    record_fallback_kernel[_blocks(n_items), THREADS](
        n_items,
        buf.wi_game,
        buf.wi_piece,
        buf.wi_is_hold,
        buf.wi_count,
        root_base,
        buf.pl_rot,
        buf.pl_col,
        buf.pl_row,
        buf.pl_spin,
        buf.fallback_action,
        buf.fallback_row,
    )
    return root_base


def _expand(buf, depth, n_children, child_item, child_slot, root_index):
    """Build, score and root-credit every child of this depth."""
    p = buf.curr
    c = buf.next
    grid = _blocks(n_children)
    expand_kernel[grid, THREADS](
        n_children,
        child_item,
        child_slot,
        buf.wi_parent,
        buf.wi_piece,
        buf.wi_new_hold,
        buf.wi_new_qi,
        buf.wi_new_bag,
        buf.wi_is_hold,
        buf.wi_game,
        buf.wi_carry,
        buf.pl_rot,
        buf.pl_col,
        buf.pl_row,
        buf.pl_spin,
        p.board,
        p.col_heights,
        p.b2b,
        p.combo,
        p.total_attack,
        p.pieces_placed,
        p.rows_cleared,
        p.chain_ramp,
        p.hold_piece,
        p.next_queue_idx,
        p.depth0_idx,
        p.garbage_remaining,
        p.garbage_timer,
        p.garbage_prevented,
        p.bag_seen,
        p.unlicensed_cash,
        p.parent_avg_height,
        p.unlicensed_cash_A,
        p.sort_hash,
        p.game,
        p.dead,
        c.board,
        c.col_heights,
        c.b2b,
        c.combo,
        c.total_attack,
        c.pieces_placed,
        c.rows_cleared,
        c.chain_ramp,
        c.hold_piece,
        c.next_queue_idx,
        c.depth0_idx,
        c.garbage_remaining,
        c.garbage_timer,
        c.garbage_prevented,
        c.bag_seen,
        c.unlicensed_cash,
        c.parent_avg_height,
        c.unlicensed_cash_A,
        c.sort_hash,
        c.game,
        c.dead,
        buf.queues,
        buf.queue_len,
        buf.board_height,
        depth,
        root_index,
    )
    evaluate_kernel[grid, THREADS](
        n_children,
        c.board,
        buf.board_height,
        c.b2b,
        c.combo,
        c.total_attack,
        c.pieces_placed,
        c.chain_ramp,
        c.garbage_remaining,
        c.garbage_prevented,
        c.unlicensed_cash,
        c.parent_avg_height,
        c.unlicensed_cash_A,
        c.hold_piece,
        c.next_queue_idx,
        c.col_heights,
        c.dead,
        c.game,
        buf.queues,
        buf.queue_len,
        c.score,
    )
    # Every live child raises its root against this frontier, pruned or not;
    # b2b_search.c:2247, 2351.
    buf.frontier_hi.fill(np.float32(-np.inf))
    buf.frontier_lo.fill(np.float32(np.inf))
    frontier_range_kernel[grid, THREADS](
        n_children, c.game, c.dead, c.score, buf.frontier_hi, buf.frontier_lo
    )
    raise_root_norm_kernel[grid, THREADS](
        n_children,
        depth,
        c.game,
        c.dead,
        c.depth0_idx,
        c.score,
        buf.frontier_hi,
        buf.frontier_lo,
        buf.root_norm,
    )


def _select(buf, n_children, width):
    """Prune the children into the parent bank, returning size and per-game counts."""
    c = buf.next
    kept, counts = select_next_beam(
        c.game[:n_children],
        c.score[:n_children],
        c.sort_hash[:n_children],
        c.depth0_idx[:n_children],
        c.rows_cleared[:n_children],
        c.dead[:n_children],
        buf.batch,
        width,
    )
    # The survivors land in the bank the next depth reads as parents.
    return gather_next_beam(buf.next, buf.curr, kept), counts


# Result extraction
def _collect(buf, counts, n_parents):
    """Read the per-game action and root arrays back as numpy."""
    batch = buf.batch
    counts_h = cp.asnumpy(counts)
    root_action = cp.asnumpy(buf.root_action)
    root_score = cp.asnumpy(buf.root_norm)
    root_row = cp.asnumpy(buf.root_row)
    root_count = cp.asnumpy(buf.root_count)
    fallback = cp.asnumpy(buf.fallback_action)

    if n_parents > 0:
        depth0 = cp.asnumpy(buf.curr.depth0_idx[:n_parents])
        leaf_score = cp.asnumpy(buf.curr.score[:n_parents])
    else:
        depth0 = np.zeros(0, dtype=np.int32)
        leaf_score = np.zeros(0, dtype=np.float32)

    starts = np.zeros(batch, dtype=np.int64)
    if batch > 1:
        starts[1:] = np.cumsum(counts_h[:-1])

    alive = counts_h > 0
    action = np.full(batch, -1, dtype=np.int32)
    # Value target: the played line's score in attack lines; b2b_search.c:2392.
    best_score = np.full(batch, DEATH_SCORE, dtype=np.float32)
    # Index of the root the best leaf descends from. An action index alone does not
    # identify a root, since multi-landing placements share one.
    root_index = np.full(batch, -1, dtype=np.int32)
    for g in range(batch):
        if alive[g]:
            # The beam is grouped by game in the C total order; b2b_search.c:2387.
            ri = int(depth0[starts[g]])
            best_score[g] = leaf_score[starts[g]]
            if 0 <= ri < root_action.shape[1]:
                action[g] = root_action[g, ri]
                root_index[g] = ri
        else:
            # An emptied beam returns before the root block; b2b_search.c:2363-2382.
            action[g] = fallback[g]
            root_count[g] = 0

    # A root every child of which died was never raised; b2b_search.c:2422-2423.
    tail = np.arange(root_action.shape[1])[None, :] >= root_count[:, None]
    root_score[~tail & (root_score < -1e29)] = ROOT_FLOOR
    root_action[tail] = -1
    root_row[tail] = -1
    root_score[tail] = ROOT_SCORE_INIT

    return BatchResult(
        action,
        best_score,
        root_index,
        root_action,
        root_score,
        root_row,
        root_count,
        alive,
        buf.placement_overflowed,
        buf.pool_overflowed,
        buf.workitem_overflowed,
    )


def _stage(profile, name, start):
    """Record the wall time of a completed stage into profile, in milliseconds."""
    if profile is None:
        return None
    cuda.synchronize()
    now = time.perf_counter()
    profile[name] = profile.get(name, 0.0) + (now - start) * 1e3
    return now


def search_batch(buffers, depth, width, profile=None):
    """Run every depth of the batched beam search over a filled BeamBuffers.

    Passing a dict as profile accumulates per-stage milliseconds into it and forces a
    device synchronisation after every stage.
    """
    # Both are clamped to the C's caps and to what the buffers were sized for;
    # b2b_search.c:1917-1919.
    depth = min(max(int(depth), 1), MAX_SEARCH_DEPTH, buffers.depth)
    width = min(int(width), MAX_BEAM_WIDTH, buffers.width)

    n_parents = buffers.n_parents
    counts = cp.zeros(buffers.batch, dtype=cp.int32)

    # Depth 0 then depths 1..depth-1; b2b_search.c:2265.
    for d in range(depth):
        if n_parents == 0:
            break

        mark = time.perf_counter() if profile is not None else None
        n_items, offsets = _emit_work_items(buffers, d, n_parents)
        mark = _stage(profile, "work_items", mark)
        if n_items == 0:
            n_parents = 0
            counts = cp.zeros(buffers.batch, dtype=cp.int32)
            break

        _enumerate(buffers, n_items)
        mark = _stage(profile, "enumerate", mark)
        n_children, child_item, child_slot, item_base = _child_index(buffers, n_items)
        mark = _stage(profile, "child_index", mark)
        if n_children == 0:
            n_parents = 0
            counts = cp.zeros(buffers.batch, dtype=cp.int32)
            break

        if d == 0:
            root_base = _seed_roots(buffers, n_items, offsets, item_base)
            root_index = root_base[child_item] + child_slot
        else:
            # expand_kernel reads root_index only at depth 0.
            root_index = child_slot

        mark = _stage(profile, "seed_roots", mark)
        _expand(buffers, d, n_children, child_item, child_slot, root_index)
        mark = _stage(profile, "expand_and_evaluate", mark)
        n_parents, counts = _select(buffers, n_children, width)
        mark = _stage(profile, "select_and_gather", mark)
        buffers.n_parents = n_parents

    return _collect(buffers, counts, n_parents)
