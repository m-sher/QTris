"""Beam bookkeeping kernels and the finalize_beam selection from b2b_search.c."""

# The teacher import runs the CUDA library bootstrap and must precede numba.cuda.
from teacher.buffers import StateBank
from teacher.constants import BEAM_STRATA, PIECE_N, ROOT_DEPTH_STEP
from teacher.kernels.rules import (
    bag_consume_piece,
    bag_nth_remaining,
    bag_remaining_count,
)
from teacher.tables import PIECE_MIN_COL

from numba import cuda

import cupy as cp
import numpy as np

_THREADS = 128


def _blocks(n):
    """Blocks of _THREADS threads covering n items."""
    return max((int(n) + _THREADS - 1) // _THREADS, 1)


# Work item generation
@cuda.jit
def count_branches_kernel(
    n_parents,
    depth,
    next_queue_idx,
    hold_piece,
    bag_seen,
    game,
    active_piece,
    queue_len,
    out_counts,
):
    """Work items one parent opens at this depth; b2b_search.c:1990-2345."""
    i = cuda.grid(1)
    if i >= n_parents:
        return

    ql = queue_len[game[i]]
    held = hold_piece[i]

    # Depth 0: the active piece, then the hold swap; b2b_search.c:1990-2248.
    if depth == 0:
        if held != PIECE_N or ql > 0:
            out_counts[i] = 2
        else:
            out_counts[i] = 1
        return

    qi = next_queue_idx[i]

    if qi >= ql:
        remaining = bag_remaining_count(bag_seen[i])
        if remaining == 0:
            out_counts[i] = 1
        elif held != PIECE_N:
            out_counts[i] = 2 * remaining
        else:
            out_counts[i] = remaining
        return

    if held != PIECE_N or qi + 1 < ql:
        out_counts[i] = 2
    else:
        out_counts[i] = 1


@cuda.jit(device=True, inline=True)
def _write_item(
    slot,
    parent,
    g,
    piece,
    new_hold,
    new_qi,
    new_bag,
    is_hold,
    carry,
    wi_parent,
    wi_piece,
    wi_new_hold,
    wi_new_qi,
    wi_new_bag,
    wi_is_hold,
    wi_game,
    wi_carry,
):
    """Store one branch of one parent into the work item arrays."""
    wi_parent[slot] = parent
    wi_piece[slot] = piece
    wi_new_hold[slot] = new_hold
    wi_new_qi[slot] = new_qi
    wi_new_bag[slot] = np.uint8(new_bag)
    wi_is_hold[slot] = is_hold
    wi_game[slot] = g
    wi_carry[slot] = np.uint8(carry)


@cuda.jit
def emit_items_kernel(
    n_parents,
    depth,
    offsets,
    next_queue_idx,
    hold_piece,
    bag_seen,
    game,
    active_piece,
    queues,
    queue_len,
    wi_parent,
    wi_piece,
    wi_new_hold,
    wi_new_qi,
    wi_new_bag,
    wi_is_hold,
    wi_game,
    wi_carry,
):
    """Write a parent's branches at its scan offset; b2b_search.c:1990-2345.

    At depth 0 the active-piece branch precedes the hold branch, so ranking the
    children of a game by (item, placement) numbers them as the C's
    depth0_placements array does.
    """
    i = cuda.grid(1)
    if i >= n_parents:
        return

    g = game[i]
    qi = next_queue_idx[i]
    ql = queue_len[g]
    held = hold_piece[i]
    bag = bag_seen[i]
    base = offsets[i]
    cap = wi_parent.shape[0]

    # Depth 0: active piece keeps the hold, the swap takes it; b2b_search.c:1990-2248.
    if depth == 0:
        active = active_piece[g]
        if base < cap:
            _write_item(
                base,
                i,
                g,
                active,
                held,
                0,
                bag,
                0,
                0,
                wi_parent,
                wi_piece,
                wi_new_hold,
                wi_new_qi,
                wi_new_bag,
                wi_is_hold,
                wi_game,
                wi_carry,
            )
        if held != PIECE_N:
            if base + 1 < cap:
                _write_item(
                    base + 1,
                    i,
                    g,
                    held,
                    active,
                    0,
                    bag,
                    1,
                    0,
                    wi_parent,
                    wi_piece,
                    wi_new_hold,
                    wi_new_qi,
                    wi_new_bag,
                    wi_is_hold,
                    wi_game,
                    wi_carry,
                )
        elif ql > 0:
            if base + 1 < cap:
                _write_item(
                    base + 1,
                    i,
                    g,
                    queues[g, 0],
                    active,
                    1,
                    bag,
                    1,
                    0,
                    wi_parent,
                    wi_piece,
                    wi_new_hold,
                    wi_new_qi,
                    wi_new_bag,
                    wi_is_hold,
                    wi_game,
                    wi_carry,
                )
        return

    if qi >= ql:
        remaining = bag_remaining_count(bag)
        if remaining == 0:
            # Bag empty: the parent is copied forward; b2b_search.c:2288-2292.
            if base < cap:
                _write_item(
                    base,
                    i,
                    g,
                    PIECE_N,
                    held,
                    qi,
                    bag,
                    0,
                    1,
                    wi_parent,
                    wi_piece,
                    wi_new_hold,
                    wi_new_qi,
                    wi_new_bag,
                    wi_is_hold,
                    wi_game,
                    wi_carry,
                )
            return

        k = 0
        for p in range(remaining):
            spec = bag_nth_remaining(bag, p)
            new_bag = bag_consume_piece(bag, spec)
            if base + k < cap:
                _write_item(
                    base + k,
                    i,
                    g,
                    spec,
                    held,
                    qi + 1,
                    new_bag,
                    0,
                    0,
                    wi_parent,
                    wi_piece,
                    wi_new_hold,
                    wi_new_qi,
                    wi_new_bag,
                    wi_is_hold,
                    wi_game,
                    wi_carry,
                )
            k += 1
            if held != PIECE_N:
                if base + k < cap:
                    _write_item(
                        base + k,
                        i,
                        g,
                        held,
                        spec,
                        qi + 1,
                        new_bag,
                        1,
                        0,
                        wi_parent,
                        wi_piece,
                        wi_new_hold,
                        wi_new_qi,
                        wi_new_bag,
                        wi_is_hold,
                        wi_game,
                        wi_carry,
                    )
                k += 1
        return

    piece = queues[g, qi]
    if base < cap:
        _write_item(
            base,
            i,
            g,
            piece,
            held,
            qi + 1,
            bag,
            0,
            0,
            wi_parent,
            wi_piece,
            wi_new_hold,
            wi_new_qi,
            wi_new_bag,
            wi_is_hold,
            wi_game,
            wi_carry,
        )

    if held != PIECE_N:
        if base + 1 < cap:
            _write_item(
                base + 1,
                i,
                g,
                held,
                piece,
                qi + 1,
                bag,
                1,
                0,
                wi_parent,
                wi_piece,
                wi_new_hold,
                wi_new_qi,
                wi_new_bag,
                wi_is_hold,
                wi_game,
                wi_carry,
            )
    elif qi + 1 < ql:
        # Empty hold plays queue[qi + 1] and holds queue[qi]; b2b_search.c:2337-2344.
        if base + 1 < cap:
            _write_item(
                base + 1,
                i,
                g,
                queues[g, qi + 1],
                piece,
                qi + 2,
                bag,
                1,
                0,
                wi_parent,
                wi_piece,
                wi_new_hold,
                wi_new_qi,
                wi_new_bag,
                wi_is_hold,
                wi_game,
                wi_carry,
            )


# Root bookkeeping
@cuda.jit
def frontier_range_kernel(n, game, dead, score, hi, lo):
    """Per-game high and low score of one live frontier; b2b_search.c:1606-1611."""
    i = cuda.grid(1)
    if i >= n or dead[i] != 0:
        return
    g = game[i]
    cuda.atomic.max(hi, g, score[i])
    cuda.atomic.min(lo, g, score[i])


@cuda.jit
def raise_root_norm_kernel(n, depth, game, dead, depth0_idx, score, hi, lo, root_norm):
    """Raise each root against this frontier's range; b2b_search.c:1603-1619."""
    i = cuda.grid(1)
    if i >= n or dead[i] != 0:
        return
    ri = depth0_idx[i]
    if ri < 0 or ri >= root_norm.shape[1]:
        return
    g = game[i]
    span = hi[g] - lo[g]
    if span > np.float32(0.0):
        z = (score[i] - hi[g]) / span
    else:
        z = np.float32(0.0)
    v = np.float32(ROOT_DEPTH_STEP) * np.float32(depth) + z
    cuda.atomic.max(root_norm, (g, ri), v)


@cuda.jit(device=True, inline=True)
def _action_index(is_hold, piece, rot, col, spin):
    """Action id of a placement, piece being the played one; b2b_search.c:2403-2411."""
    norm_col = col + PIECE_MIN_COL[piece, rot]
    return is_hold * 160 + rot * 40 + norm_col * 4 + spin


@cuda.jit
def seed_roots_kernel(
    n_items,
    wi_game,
    wi_piece,
    wi_is_hold,
    pl_count,
    wi_root_base,
    pl_rot,
    pl_col,
    pl_row,
    pl_spin,
    root_action,
    root_row,
    root_count,
):
    """Write the depth-0 placements as per-game roots; b2b_search.c:2418-2436.

    wi_root_base[w] is the per-game exclusive scan of pl_count over that game's
    depth-0 items in emission order, so wi_root_base[w] + p is both the root
    index and the child's depth0_idx. wi_piece[w] is the played piece.
    """
    t = cuda.grid(1)
    plmax = pl_rot.shape[1]
    if t >= n_items * plmax:
        return

    w = t // plmax
    p = t - w * plmax
    if p >= pl_count[w]:
        return

    ri = wi_root_base[w] + p
    if ri >= root_action.shape[1]:
        return

    g = wi_game[w]
    rot = pl_rot[w, p]
    root_action[g, ri] = _action_index(
        wi_is_hold[w], wi_piece[w], rot, pl_col[w, p], pl_spin[w, p]
    )
    root_row[g, ri] = pl_row[w, p]
    cuda.atomic.max(root_count, g, ri + 1)


@cuda.jit
def record_fallback_kernel(
    n_items,
    wi_game,
    wi_piece,
    wi_is_hold,
    pl_count,
    wi_root_base,
    pl_rot,
    pl_col,
    pl_row,
    pl_spin,
    fallback_action,
    fallback_row,
):
    """Record each game's first depth-0 placement; b2b_search.c:1993-1997.

    The item holding root index 0 is the first branch with any placement, dead
    or not, which the C keeps for an emptied beam at :2369-2377.
    """
    i = cuda.grid(1)
    if i >= n_items:
        return
    if pl_count[i] == 0 or wi_root_base[i] != 0:
        return

    g = wi_game[i]
    rot = pl_rot[i, 0]
    fallback_action[g] = _action_index(
        wi_is_hold[i], wi_piece[i], rot, pl_col[i, 0], pl_spin[i, 0]
    )
    fallback_row[g] = pl_row[i, 0]


# Finalize
def select_next_beam(
    game, score, sort_hash, depth0_idx, rows_cleared, dead, batch, width
):
    """Kept child indices per game in the finalize_beam order; b2b_search.c:1626-1656.

    Returns the flat indices grouped by ascending game and the per-game count.
    """
    counts = cp.zeros(batch, dtype=cp.int32)
    live = cp.flatnonzero(dead == 0)
    if live.size == 0:
        return cp.empty(0, dtype=cp.int64), counts

    # Score descending, sort_hash ascending, depth0_idx ascending, per game;
    # the last lexsort key is the primary one. b2b_search.c:1576-1586.
    h = sort_hash[live]
    keys = cp.empty((5, live.size), dtype=cp.float64)
    keys[0] = depth0_idx[live]
    keys[1] = h & np.uint64(0xFFFFFFFF)
    keys[2] = h >> np.uint64(32)
    keys[3] = -score[live]
    keys[4] = game[live]
    order = live[cp.lexsort(keys)]

    # Keep the first entry of each hash run within a game; b2b_search.c:1631-1635.
    gs = game[order]
    hs = sort_hash[order]
    fresh = cp.ones(order.size, dtype=cp.bool_)
    fresh[1:] = (gs[1:] != gs[:-1]) | (hs[1:] != hs[:-1])
    kept = order[fresh]
    gk = gs[fresh]
    n = kept.size

    # Per-game, per-stratum quota; b2b_search.c:1636-1647.
    strat = cp.minimum(rows_cleared[kept], BEAM_STRATA - 1)
    key = gk.astype(cp.int64) * BEAM_STRATA + strat
    per_stratum = cp.bincount(key, minlength=batch * BEAM_STRATA)
    per_stratum = per_stratum.reshape(batch, BEAM_STRATA)
    nonempty = (per_stratum > 0).sum(axis=1)
    quota = width // cp.maximum(nonempty, 1)

    # Rank inside (game, stratum) in sorted order.
    by_key = cp.argsort(key, kind="stable")
    group_start = cp.zeros(batch * BEAM_STRATA, dtype=cp.int64)
    group_start[1:] = cp.cumsum(per_stratum.ravel())[:-1]
    rank = cp.empty(n, dtype=cp.int64)
    rank[by_key] = cp.arange(n, dtype=cp.int64) - group_start[key[by_key]]
    first = rank < quota[gk]

    # Fill to width from the entries the quota pass left; b2b_search.c:1648-1649.
    filled = cp.minimum(per_stratum, quota[:, None]).sum(axis=1)
    rest = ~first
    prefix = cp.zeros(n + 1, dtype=cp.int64)
    prefix[1:] = cp.cumsum(rest)
    per_game = cp.bincount(gk, minlength=batch)
    game_start = cp.zeros(batch, dtype=cp.int64)
    game_start[1:] = cp.cumsum(per_game)[:-1]
    rank_rest = prefix[:-1] - prefix[game_start[gk]]
    second = rest & (filled[gk] + rank_rest < width)

    take = first | second
    return kept[take], cp.bincount(gk[take], minlength=batch).astype(cp.int32)


def gather_next_beam(src, dst, kept):
    """Copy the kept children into the front of dst; the banks must differ."""
    n = int(kept.size)
    for name in StateBank.COLUMNS:
        getattr(dst, name)[:n] = getattr(src, name)[kept]
    return n


# Single-launch debug entry points
def debug_branches(
    next_queue_idx,
    hold_piece,
    bag_seen,
    game,
    queues,
    queue_len,
    depth=1,
    active_piece=0,
):
    """Count and emit work items for host parents, returning numpy arrays."""
    n = len(next_queue_idx)
    d_qi = cp.asarray(next_queue_idx, dtype=cp.int32)
    d_hold = cp.asarray(hold_piece, dtype=cp.int32)
    d_bag = cp.asarray(bag_seen, dtype=cp.uint8)
    d_game = cp.asarray(game, dtype=cp.int32)
    d_queues = cp.atleast_2d(cp.asarray(queues, dtype=cp.int32))
    d_qlen = cp.asarray(queue_len, dtype=cp.int32)
    active = np.asarray(active_piece, dtype=np.int32)
    d_active = cp.asarray(np.ascontiguousarray(np.broadcast_to(active, d_qlen.shape)))

    counts = cp.zeros(n, dtype=cp.int32)
    count_branches_kernel[_blocks(n), _THREADS](
        n, depth, d_qi, d_hold, d_bag, d_game, d_active, d_qlen, counts
    )

    offsets = cp.zeros(n, dtype=cp.int32)
    offsets[1:] = cp.cumsum(counts)[:-1]
    m = max(int(counts.sum()), 1)
    wi_parent = cp.full(m, -1, dtype=cp.int32)
    wi_piece = cp.full(m, -1, dtype=cp.int32)
    wi_new_hold = cp.full(m, -1, dtype=cp.int32)
    wi_new_qi = cp.full(m, -1, dtype=cp.int32)
    wi_new_bag = cp.zeros(m, dtype=cp.uint8)
    wi_is_hold = cp.zeros(m, dtype=cp.int32)
    wi_game = cp.full(m, -1, dtype=cp.int32)
    wi_carry = cp.zeros(m, dtype=cp.uint8)

    emit_items_kernel[_blocks(n), _THREADS](
        n,
        depth,
        offsets,
        d_qi,
        d_hold,
        d_bag,
        d_game,
        d_active,
        d_queues,
        d_qlen,
        wi_parent,
        wi_piece,
        wi_new_hold,
        wi_new_qi,
        wi_new_bag,
        wi_is_hold,
        wi_game,
        wi_carry,
    )
    return {
        "counts": counts.get(),
        "offsets": offsets.get(),
        "parent": wi_parent.get(),
        "piece": wi_piece.get(),
        "new_hold": wi_new_hold.get(),
        "new_qi": wi_new_qi.get(),
        "new_bag": wi_new_bag.get(),
        "is_hold": wi_is_hold.get(),
        "game": wi_game.get(),
        "carry": wi_carry.get(),
    }


def debug_roots(
    wi_game,
    wi_piece,
    wi_is_hold,
    pl_count,
    wi_root_base,
    pl_rot,
    pl_col,
    pl_row,
    pl_spin,
    batch,
    root_capacity,
):
    """Seed roots and the fallback from host placements, returning numpy."""
    n = len(wi_game)
    d_game = cp.asarray(wi_game, dtype=cp.int32)
    d_piece = cp.asarray(wi_piece, dtype=cp.int32)
    d_is_hold = cp.asarray(wi_is_hold, dtype=cp.int32)
    d_count = cp.asarray(pl_count, dtype=cp.int32)
    d_base = cp.asarray(wi_root_base, dtype=cp.int32)
    d_rot = cp.asarray(pl_rot, dtype=cp.int8)
    d_col = cp.asarray(pl_col, dtype=cp.int8)
    d_row = cp.asarray(pl_row, dtype=cp.int8)
    d_spin = cp.asarray(pl_spin, dtype=cp.int8)

    root_action = cp.full((batch, root_capacity), -1, dtype=cp.int32)
    root_row = cp.full((batch, root_capacity), -1, dtype=cp.int32)
    root_count = cp.zeros(batch, dtype=cp.int32)
    fallback_action = cp.full(batch, -1, dtype=cp.int32)
    fallback_row = cp.full(batch, -1, dtype=cp.int32)

    total = n * d_rot.shape[1]
    seed_roots_kernel[_blocks(total), _THREADS](
        n,
        d_game,
        d_piece,
        d_is_hold,
        d_count,
        d_base,
        d_rot,
        d_col,
        d_row,
        d_spin,
        root_action,
        root_row,
        root_count,
    )
    record_fallback_kernel[_blocks(n), _THREADS](
        n,
        d_game,
        d_piece,
        d_is_hold,
        d_count,
        d_base,
        d_rot,
        d_col,
        d_row,
        d_spin,
        fallback_action,
        fallback_row,
    )
    return {
        "root_action": root_action.get(),
        "root_row": root_row.get(),
        "root_count": root_count.get(),
        "fallback_action": fallback_action.get(),
        "fallback_row": fallback_row.get(),
    }


def debug_select(game, score, sort_hash, depth0_idx, rows_cleared, dead, batch, width):
    """Run select_next_beam on host columns, returning numpy arrays."""
    kept, counts = select_next_beam(
        cp.asarray(game, dtype=cp.int32),
        cp.asarray(score, dtype=cp.float32),
        cp.asarray(sort_hash, dtype=cp.uint64),
        cp.asarray(depth0_idx, dtype=cp.int32),
        cp.asarray(rows_cleared, dtype=cp.int32),
        cp.asarray(dead, dtype=cp.uint8),
        batch,
        width,
    )
    return {"kept": kept.get(), "counts": counts.get()}
