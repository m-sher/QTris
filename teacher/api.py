"""Public GPU teacher API over the batched beam search."""

# The teacher import runs the CUDA library bootstrap and must come first.
from teacher import beam
from teacher.buffers import DEFAULT_POOL_PER_PARENT, BeamBuffers
from teacher.constants import (
    ACTION_COL_STRIDE,
    ACTION_HOLD_STRIDE,
    ACTION_ROT_STRIDE,
    BOARD_COLS,
    BOARD_ROWS,
    MAX_PLACEMENTS,
    ROOT_CAPACITY,
    decode_action,
)

from typing import NamedTuple

import numpy as np

_COL_BITS = (np.uint16(1) << np.arange(BOARD_COLS, dtype=np.uint16)).astype(np.uint16)


class Placement(NamedTuple):
    """A placement in descriptor form; b2b_search.c:2411."""

    is_hold: int
    rot: int
    norm_col: int
    landing_row: int
    spin: int


# Board and action conversion
def board_bitmasks(boards):
    """Row bitmasks of one or many BOARD_ROWS by BOARD_COLS occupancy grids."""
    grid = np.asarray(boards)
    if grid.shape[-2:] != (BOARD_ROWS, BOARD_COLS):
        raise ValueError(
            f"board must end in ({BOARD_ROWS}, {BOARD_COLS}), got {grid.shape}"
        )
    return ((grid != 0).astype(np.uint16) * _COL_BITS).sum(axis=-1, dtype=np.uint16)


def _batch_masks(boards):
    """Batched row bitmasks from (B, 40, 10) grids or (B, 40) masks."""
    arr = np.asarray(boards)
    if arr.ndim == 3:
        return board_bitmasks(arr)
    if arr.ndim == 2 and arr.shape[1] == BOARD_ROWS:
        return np.ascontiguousarray(arr, dtype=np.uint16)
    raise ValueError(
        f"boards must be (B, {BOARD_ROWS}, {BOARD_COLS}) or (B, {BOARD_ROWS}), "
        f"got {arr.shape}"
    )


def placement_of(action, landing_row):
    """Descriptor of one action index, all -1 when there is no action."""
    if int(action) < 0:
        return Placement(-1, -1, -1, -1, -1)
    is_hold, rot, norm_col, spin = decode_action(int(action))
    return Placement(is_hold, rot, norm_col, int(landing_row), spin)


def placement_array(actions, landing_rows):
    """Descriptors of many action indices, as an (n, 5) int32 array."""
    actions = np.asarray(actions, dtype=np.int32)
    is_hold, rest = np.divmod(actions, ACTION_HOLD_STRIDE)
    rot, rest = np.divmod(rest, ACTION_ROT_STRIDE)
    norm_col, spin = np.divmod(rest, ACTION_COL_STRIDE)
    rows = np.asarray(landing_rows, dtype=np.int32)
    return np.stack([is_hold, rot, norm_col, rows, spin], axis=1).astype(np.int32)


class GpuTeacher:
    """Batched beam search on the GPU over one reused BeamBuffers allocation."""

    def __init__(
        self,
        max_batch=1,
        width=128,
        depth=10,
        queue_len=10,
        max_placements=MAX_PLACEMENTS,
        root_capacity=ROOT_CAPACITY,
        pool_per_parent=DEFAULT_POOL_PER_PARENT,
    ):
        self.max_batch = int(max_batch)
        self.width = int(width)
        self.depth = int(depth)
        self.queue_len = int(queue_len)
        self.max_placements = int(max_placements)
        self.root_capacity = int(root_capacity)
        self.pool_per_parent = int(pool_per_parent)
        self.buffers = None
        self._shape = None
        self._ensure(self.max_batch, self.width, self.depth, self.queue_len)

    def _ensure(self, batch, width, depth, queue_len):
        """Buffers for one shape, reallocated only when that shape changes."""
        shape = (int(batch), int(width), int(depth), int(queue_len))
        if self._shape != shape:
            self.buffers = BeamBuffers.allocate(
                *shape,
                max_placements=self.max_placements,
                root_capacity=self.root_capacity,
                pool_per_parent=self.pool_per_parent,
            )
            self._shape = shape
        return self.buffers

    @property
    def nbytes(self):
        """Device bytes the current allocation holds."""
        return self.buffers.nbytes

    def search_batch(
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
    ):
        """Search a batch of positions, returning one BatchResult of numpy arrays."""
        masks = _batch_masks(boards)
        batch = masks.shape[0]
        if batch > self.max_batch:
            raise ValueError(f"batch {batch} over max_batch {self.max_batch}")
        queues = np.ascontiguousarray(queues, dtype=np.int32).reshape(batch, -1)

        buf = self._ensure(batch, self.width, self.depth, queues.shape[1])
        buf.set_roots(
            masks,
            active,
            hold,
            queues,
            queue_len,
            b2b,
            combo,
            total_garbage,
            bag_seen,
        )
        return beam.search_batch(buf, self.depth, self.width)

    def search_with_scores(
        self,
        board,
        active_piece,
        hold_piece,
        queue,
        b2b,
        combo,
        total_garbage,
        garbage_push_delay=1,  # unread, as b2b_search.c:1969
        bag_seen=0,
        search_depth=10,
        beam_width=128,
        max_roots=512,
    ):
        """Search one position for its best move and every root candidate.

        Returns (action, placement, root_actions, root_scores, root_placements,
        root_landing_rows, best_score), the descriptor form of
        CB2BSearch.search_with_scores. Root scores rank the roots on the per-depth
        normalised scale; best_score is the played line's score in attack lines.
        """
        queue_arr = np.ascontiguousarray(queue, dtype=np.int32).reshape(1, -1)
        buf = self._ensure(1, int(beam_width), int(search_depth), queue_arr.shape[1])
        buf.set_roots(
            board_bitmasks(board).reshape(1, BOARD_ROWS),
            active_piece,
            hold_piece,
            queue_arr,
            queue_arr.shape[1],
            b2b,
            combo,
            total_garbage,
            bag_seen,
        )
        result = beam.search_batch(buf, int(search_depth), int(beam_width))

        action = int(result.action[0])
        if bool(result.alive[0]):
            ri = int(result.root_index[0])
            row = int(result.root_row[0, ri]) if ri >= 0 else -1
        else:
            # An emptied beam plays the depth-0 fallback; b2b_search.c:2368-2378.
            row = int(buf.fallback_row[0])

        n = min(int(result.root_count[0]), int(max_roots))
        root_actions = result.root_action[0, :n]
        root_rows = result.root_row[0, :n]
        return (
            action,
            placement_of(action, row),
            root_actions,
            result.root_score[0, :n],
            placement_array(root_actions, root_rows),
            root_rows,
            float(result.best_score[0]),
        )
