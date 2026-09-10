"""Structure-of-arrays device buffers for the batched GPU beam search."""

import cupy as cp
import numpy as np

from teacher.constants import (
    BOARD_COLS,
    BOARD_ROWS,
    MAX_BEAM_WIDTH,
    MAX_PLACEMENTS,
    MAX_SEARCH_DEPTH,
    ROOT_CAPACITY,
    ROOT_SCORE_INIT,
)

# Branches one parent opens at one depth
NORMAL_BRANCHES = 2  # played piece and hold swap; b2b_search.c:2320-2344
SPECULATIVE_BRANCHES = 14  # 7 bag pieces, each with a hold swap; b2b_search.c:2283-2315

DEFAULT_POOL_PER_PARENT = 96


def speculative_reachable(depth: int, queue_len: int) -> bool:
    """True when an expanded depth can reach next_queue_idx >= queue_len."""
    # A parent entering loop depth d carries next_queue_idx <= d and the deepest
    # expanded depth is depth - 1; b2b_search.c:2269-2282.
    return depth - 1 >= queue_len


def _per_game(values, batch: int, dtype) -> np.ndarray:
    """Broadcast a scalar or per-game sequence to a contiguous (batch,) array."""
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        arr = np.full(batch, arr, dtype=dtype)
    if arr.size != batch:
        raise ValueError(f"expected {batch} per-game values, got {arr.size}")
    return np.ascontiguousarray(arr.reshape(batch), dtype=dtype)


def _root_col_heights(boards: np.ndarray) -> np.ndarray:
    """Per-column heights of each root board; b2b_search.c:153-164."""
    bit = (np.uint16(1) << np.arange(BOARD_COLS, dtype=np.uint16)).astype(np.uint16)
    filled = (boards[:, :, None] & bit) != 0
    top = filled.argmax(axis=1)
    return np.where(filled.any(axis=1), BOARD_ROWS - top, 0).astype(np.int8)


class StateBank:
    """One beam bank holding the SearchState SoA; b2b_search.c:124-150."""

    COLUMNS = (
        "board",
        "col_heights",
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
        "score",
        "sort_hash",
        "tt_hash",
        "game",
        "dead",
    )

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.board = cp.zeros((capacity, BOARD_ROWS), dtype=cp.uint16)
        self.col_heights = cp.zeros((capacity, BOARD_COLS), dtype=cp.int8)
        self.b2b = cp.zeros(capacity, dtype=cp.int32)
        self.combo = cp.zeros(capacity, dtype=cp.int32)
        self.total_attack = cp.zeros(capacity, dtype=cp.float32)
        self.pieces_placed = cp.zeros(capacity, dtype=cp.int32)
        self.rows_cleared = cp.zeros(capacity, dtype=cp.int32)
        self.chain_ramp = cp.zeros(capacity, dtype=cp.int32)
        self.hold_piece = cp.zeros(capacity, dtype=cp.int32)
        self.next_queue_idx = cp.zeros(capacity, dtype=cp.int32)
        self.depth0_idx = cp.full(capacity, -1, dtype=cp.int32)
        self.garbage_remaining = cp.zeros(capacity, dtype=cp.int32)
        self.garbage_timer = cp.zeros(capacity, dtype=cp.int32)
        self.garbage_prevented = cp.zeros(capacity, dtype=cp.float32)
        self.bag_seen = cp.zeros(capacity, dtype=cp.uint8)
        self.unlicensed_cash = cp.zeros(capacity, dtype=cp.uint8)
        self.parent_avg_height = cp.zeros(capacity, dtype=cp.float32)
        self.unlicensed_cash_A = cp.zeros(capacity, dtype=cp.float32)
        self.score = cp.zeros(capacity, dtype=cp.float32)
        self.sort_hash = cp.zeros(capacity, dtype=cp.uint64)
        self.tt_hash = cp.zeros(capacity, dtype=cp.uint64)
        self.game = cp.zeros(capacity, dtype=cp.int32)
        self.dead = cp.zeros(capacity, dtype=cp.uint8)

    @property
    def nbytes(self) -> int:
        """Device bytes held by this bank."""
        return sum(getattr(self, name).nbytes for name in self.COLUMNS)


class BeamBuffers:
    """Every device buffer one batched search shape needs, allocated once."""

    SIDE_ARRAYS = (
        "queues",
        "queue_len",
        "active_piece",
        "wi_parent",
        "wi_piece",
        "wi_new_hold",
        "wi_new_qi",
        "wi_new_bag",
        "wi_is_hold",
        "wi_game",
        "wi_count",
        "wi_carry",
        "pl_rot",
        "pl_col",
        "pl_row",
        "pl_spin",
        "root_norm",
        "root_action",
        "root_row",
        "root_count",
        "frontier_hi",
        "frontier_lo",
        "fallback_action",
        "fallback_row",
        "wi_total",
        "child_total",
        "placement_overflow",
        "pool_overflow",
        "workitem_overflow",
    )

    def __init__(
        self,
        batch: int,
        width: int,
        depth: int,
        queue_len: int,
        max_placements: int = MAX_PLACEMENTS,
        root_capacity: int = ROOT_CAPACITY,
        pool_per_parent: int = DEFAULT_POOL_PER_PARENT,
    ) -> None:
        self.batch = int(batch)
        self.width = min(int(width), MAX_BEAM_WIDTH)  # b2b_search.c:1923
        self.depth = min(max(int(depth), 1), MAX_SEARCH_DEPTH)  # b2b_search.c:1922-1924
        self.queue_capacity = max(int(queue_len), 1)
        self.max_placements = int(max_placements)
        self.root_capacity = int(root_capacity)
        self.pool_per_parent = int(pool_per_parent)

        self.branches_per_parent = (
            SPECULATIVE_BRANCHES
            if speculative_reachable(self.depth, self.queue_capacity)
            else NORMAL_BRANCHES
        )
        # spec_mult doubles the child bound while speculative depths run;
        # b2b_search.c:1931
        self.spec_mult = 2 if self.branches_per_parent > NORMAL_BRANCHES else 1
        # M = one work item per branch of every beam slot of every game
        self.workitem_capacity = self.batch * self.width * self.branches_per_parent
        # N = pool_per_parent children per work item, clamped to the C's max_next
        # of beam_width * MAX_PLACEMENTS * spec_mult per game (b2b_search.c:1932)
        # and floored at one full beam or one full root expansion per game
        self.state_capacity = max(
            min(
                self.workitem_capacity * self.pool_per_parent,
                self.batch * self.width * self.max_placements * self.spec_mult,
            ),
            self.batch * max(self.width, self.root_capacity),
        )

        self.curr = StateBank(self.state_capacity)
        self.next = StateBank(self.state_capacity)

        # Per-game inputs
        self.board_height = BOARD_ROWS
        self.queues = cp.zeros((self.batch, self.queue_capacity), dtype=cp.int32)
        self.queue_len = cp.zeros(self.batch, dtype=cp.int32)
        self.active_piece = cp.zeros(self.batch, dtype=cp.int32)

        # Work items, capacity M
        m = self.workitem_capacity
        self.wi_parent = cp.zeros(m, dtype=cp.int32)
        self.wi_piece = cp.zeros(m, dtype=cp.int32)
        self.wi_new_hold = cp.zeros(m, dtype=cp.int32)
        self.wi_new_qi = cp.zeros(m, dtype=cp.int32)
        self.wi_new_bag = cp.zeros(m, dtype=cp.uint8)
        self.wi_is_hold = cp.zeros(m, dtype=cp.int32)
        self.wi_game = cp.zeros(m, dtype=cp.int32)
        self.wi_count = cp.zeros(m, dtype=cp.int32)
        self.wi_carry = cp.zeros(m, dtype=cp.uint8)

        # Placements, (M, PLMAX)
        shape = (m, self.max_placements)
        self.pl_rot = cp.zeros(shape, dtype=cp.int8)
        self.pl_col = cp.zeros(shape, dtype=cp.int8)
        self.pl_row = cp.zeros(shape, dtype=cp.int8)
        self.pl_spin = cp.zeros(shape, dtype=cp.int8)

        # Roots, per game
        roots = (self.batch, self.root_capacity)
        self.root_norm = cp.full(roots, ROOT_SCORE_INIT, dtype=cp.float32)
        self.root_action = cp.full(roots, -1, dtype=cp.int32)
        self.root_row = cp.full(roots, -1, dtype=cp.int32)
        self.root_count = cp.zeros(self.batch, dtype=cp.int32)
        self.frontier_hi = cp.zeros(self.batch, dtype=cp.float32)
        self.frontier_lo = cp.zeros(self.batch, dtype=cp.float32)
        self.fallback_action = cp.full(self.batch, -1, dtype=cp.int32)
        self.fallback_row = cp.full(self.batch, -1, dtype=cp.int32)

        # Device counters and overflow flags, set by atomics inside kernels
        self.wi_total = cp.zeros(1, dtype=cp.int32)
        self.child_total = cp.zeros(1, dtype=cp.int32)
        self.placement_overflow = cp.zeros(1, dtype=cp.int32)
        self.pool_overflow = cp.zeros(1, dtype=cp.int32)
        self.workitem_overflow = cp.zeros(1, dtype=cp.int32)

        self.n_parents = 0

    @classmethod
    def allocate(
        cls,
        batch: int,
        width: int,
        depth: int,
        queue_len: int,
        max_placements: int = MAX_PLACEMENTS,
        root_capacity: int = ROOT_CAPACITY,
        pool_per_parent: int = DEFAULT_POOL_PER_PARENT,
    ) -> "BeamBuffers":
        """Allocate every buffer for one (batch, width, depth, queue_len) shape."""
        return cls(
            batch,
            width,
            depth,
            queue_len,
            max_placements,
            root_capacity,
            pool_per_parent,
        )

    # Counters
    @property
    def workitem_count(self) -> int:
        """Work items appended by the last branch pass."""
        return int(self.wi_total[0])

    @property
    def child_count(self) -> int:
        """Children appended into the next bank by the last expansion pass."""
        return int(self.child_total[0])

    # Overflow flags
    @property
    def placement_overflowed(self) -> bool:
        """True when a work item found more than max_placements placements."""
        return bool(self.placement_overflow[0])

    @property
    def pool_overflowed(self) -> bool:
        """True when children outran the state capacity."""
        return bool(self.pool_overflow[0])

    @property
    def workitem_overflowed(self) -> bool:
        """True when branches outran the work item capacity."""
        return bool(self.workitem_overflow[0])

    @property
    def any_overflowed(self) -> bool:
        """True when any capacity was exceeded."""
        return (
            self.placement_overflowed
            or self.pool_overflowed
            or self.workitem_overflowed
        )

    @property
    def nbytes(self) -> int:
        """Total device bytes across both banks and every side buffer."""
        banks = self.curr.nbytes + self.next.nbytes
        return banks + sum(getattr(self, n).nbytes for n in self.SIDE_ARRAYS)

    def reset(self) -> None:
        """Clear the counters, roots and overflow flags without reallocating."""
        self.root_norm.fill(ROOT_SCORE_INIT)
        self.root_action.fill(-1)
        self.root_row.fill(-1)
        self.root_count.fill(0)
        self.fallback_action.fill(-1)
        self.fallback_row.fill(-1)
        self.wi_count.fill(0)
        self.wi_total.fill(0)
        self.child_total.fill(0)
        self.placement_overflow.fill(0)
        self.pool_overflow.fill(0)
        self.workitem_overflow.fill(0)
        self.n_parents = 0

    def swap(self) -> None:
        """Exchange the parent and child banks and zero the child counter."""
        # b2b_search.c:2260-2266
        self.curr, self.next = self.next, self.curr
        self.child_total.fill(0)

    def set_roots(
        self,
        boards,
        active,
        hold,
        queues,
        queue_len,
        b2b,
        combo,
        total_garbage,
        bag_seen=0,
    ) -> None:
        """Reset, then fill the depth-0 parent bank and per-game inputs from numpy."""
        b = self.batch
        boards = np.ascontiguousarray(boards, dtype=np.uint16)
        if boards.shape != (b, BOARD_ROWS):
            raise ValueError(f"boards must be ({b}, {BOARD_ROWS}), got {boards.shape}")
        queues = np.ascontiguousarray(queues, dtype=np.int32).reshape(b, -1)
        if queues.shape[1] > self.queue_capacity:
            raise ValueError(
                f"queue of {queues.shape[1]} exceeds capacity {self.queue_capacity}"
            )

        qlen = _per_game(queue_len, b, np.int32)
        if qlen.min() < 1 or qlen.max() > queues.shape[1]:
            raise ValueError(
                f"queue_len {qlen.min()}..{qlen.max()} outside the "
                f"{queues.shape[1]} queue entries supplied"
            )
        if speculative_reachable(self.depth, int(qlen.min())) and (
            self.branches_per_parent < SPECULATIVE_BRANCHES
        ):
            raise ValueError(
                f"depth {self.depth} with queue_len {int(qlen.min())} reaches the "
                f"speculative branch; allocate with queue_len <= {int(qlen.min())}"
            )

        self.reset()
        self.queues.fill(0)
        self.queues[:, : queues.shape[1]] = cp.asarray(queues)
        self.queue_len[:] = cp.asarray(qlen)
        self.active_piece[:] = cp.asarray(_per_game(active, b, np.int32))

        # Depth-0 parent, one per game; b2b_search.c:1915-1975
        p = self.curr
        p.board[:b] = cp.asarray(boards)
        p.col_heights[:b] = cp.asarray(_root_col_heights(boards))
        p.b2b[:b] = cp.asarray(_per_game(b2b, b, np.int32))
        p.combo[:b] = cp.asarray(_per_game(combo, b, np.int32))
        p.total_attack[:b] = 0.0
        p.pieces_placed[:b] = 0
        p.rows_cleared[:b] = 0
        p.chain_ramp[:b] = 0
        p.hold_piece[:b] = cp.asarray(_per_game(hold, b, np.int32))
        p.next_queue_idx[:b] = 0
        p.depth0_idx[:b] = -1
        p.garbage_remaining[:b] = cp.asarray(_per_game(total_garbage, b, np.int32))
        p.garbage_timer[:b] = 0  # garbage_push_delay ignored; b2b_search.c:1969-1970
        p.garbage_prevented[:b] = 0.0
        p.bag_seen[:b] = cp.asarray(_per_game(bag_seen, b, np.uint8))
        p.unlicensed_cash[:b] = 0
        p.parent_avg_height[:b] = 0.0
        p.unlicensed_cash_A[:b] = 0.0
        p.score[:b] = 0.0
        p.sort_hash[:b] = 0
        p.tt_hash[:b] = 0
        p.game[:b] = cp.arange(b, dtype=cp.int32)
        p.dead[:b] = 0
        self.n_parents = b
