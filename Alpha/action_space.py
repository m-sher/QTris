"""Canonical fixed-size action space (320 slots).

Layout matches PyTetrisEnv's pathfinding policy interface
(see KeySequencesBitboard._placement_index + main loop):

    out_idx = is_hold * 160 + base_idx * 2 + tier
    base_idx = rot * 20 + norm_col * 2 + is_spin

with:
    is_hold ∈ {0, 1}
    rot     ∈ {0, 1, 2, 3}
    norm_col ∈ {0..9}     (board column = bbox-col + min_col_offset[piece, rot])
    is_spin ∈ {0, 1}      (collapsed: T_MINI / T_FULL / ALL_MINI all → 1)
    tier    ∈ {0, 1}      (0 = lowest landing row, 1 = highest)

When a base slot has K landing rows:
    K == 1 → only tier 0 used
    K == 2 → both tiers used
    K >= 3 → tiers selected via _select_tiers (lowest + highest); middles dropped

Dropped placements are never reachable in the canonical space — uncommon in
practice but a real (small) loss of expressiveness.  Trade-off: a fixed
action-space matches the existing PyTetrisEnv interface and gives the NN a
static output shape.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np

from TetrisEnv.CTetrisCore import TetrisCore, get_piece_min_col


NUM_HOLDS = 2
NUM_ROTATIONS = 4
NUM_COLS = 10
NUM_SPIN_FLAGS = 2
NUM_TIERS = 2
BASE_POSITIONS = NUM_ROTATIONS * NUM_COLS * NUM_SPIN_FLAGS  # 80
NUM_ACTIONS = NUM_HOLDS * BASE_POSITIONS * NUM_TIERS         # 320


# Cached (piece_type, rotation) → bounding-box-to-board-col offset.  Built on
# first import so we don't hit the C accessor in hot paths.
_MIN_COL_OFFSETS = np.zeros((8, 4), dtype=np.int32)
for _p in range(8):
    for _r in range(4):
        _MIN_COL_OFFSETS[_p, _r] = get_piece_min_col(_p, _r)


def _resolve_piece_types(
    placements: np.ndarray, active: int, hold: int, queue_first: int
) -> np.ndarray:
    """For each placement, identify which piece type is being placed.

    is_hold == 0 → active piece
    is_hold == 1, hold piece set     → hold piece
    is_hold == 1, hold piece empty   → queue[0] (the swap-and-play case)
    """
    is_hold = placements[:, 0]
    if hold == 0:
        held_piece = queue_first
    else:
        held_piece = hold
    return np.where(is_hold == 0, active, held_piece).astype(np.int32)


def _select_tiers(K: int, N: int = NUM_TIERS) -> List[int]:
    """Mirror of KeySequencesBitboard._select_tiers."""
    if K <= N:
        return list(range(K))
    if N == 1:
        return [0]
    return [round(t * (K - 1) / (N - 1)) for t in range(N)]


def canonical_indices(
    placements: np.ndarray,
    state: TetrisCore,
    num_row_tiers: int = NUM_TIERS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a state's placements onto canonical-action-space slots.

    Args:
        placements: (N, 5) int32 from tet_enumerate_placements:
            [is_hold, rot, col, landing_row, spin_type]
        state: the TetrisCore the placements were enumerated from
        num_row_tiers: tiers per (is_hold, rot, norm_col, is_spin) slot

    Returns:
        action_idx: (N,) int32      — canonical slot for each placement, or -1 if dropped
        kept_mask:  (N,) bool       — True where placement is reachable in canonical space
        valid_mask: (NUM_ACTIONS,)  — True where some placement maps to that slot
    """
    N = len(placements)
    action_idx = np.full(N, -1, dtype=np.int32)
    kept_mask = np.zeros(N, dtype=np.bool_)
    valid_mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)

    if N == 0:
        return action_idx, kept_mask, valid_mask

    queue = state.queue
    queue_first = queue[0] if queue else 0
    piece_types = _resolve_piece_types(
        placements, state.active_piece, state.hold_piece, queue_first
    )

    is_hold = placements[:, 0].astype(np.int32)
    rot = placements[:, 1].astype(np.int32)
    bbox_col = placements[:, 2].astype(np.int32)
    landing_row = placements[:, 3].astype(np.int32)
    spin_type = placements[:, 4].astype(np.int32)
    is_spin = (spin_type != 0).astype(np.int32)

    # Normalize col into board space.
    min_col = _MIN_COL_OFFSETS[piece_types, rot]
    norm_col = bbox_col + min_col

    # Drop placements that fall outside the 0..9 column range (defensive — find_placements
    # shouldn't emit these, but cheap to guard).
    in_range = (norm_col >= 0) & (norm_col < NUM_COLS)

    # Group by (is_hold, rot, norm_col, is_spin) → list of (landing_row, plac_idx).
    base_idx = rot * (NUM_COLS * NUM_SPIN_FLAGS) + norm_col * NUM_SPIN_FLAGS + is_spin
    slot_key = is_hold * BASE_POSITIONS + base_idx  # 0..159

    # Build per-slot landing-row lists, sorted by landing_row.
    # 160 buckets × at most a handful of landings each.
    buckets: List[List[Tuple[int, int]]] = [[] for _ in range(NUM_HOLDS * BASE_POSITIONS)]
    for i in range(N):
        if not in_range[i]:
            continue
        buckets[slot_key[i]].append((int(landing_row[i]), i))

    for sk, entries in enumerate(buckets):
        if not entries:
            continue
        entries.sort(key=lambda t: t[0])
        K = len(entries)
        chosen_tiers = _select_tiers(K, num_row_tiers)
        for tier, ci in enumerate(chosen_tiers):
            _, plac_i = entries[ci]
            slot = sk * num_row_tiers + tier  # canonical 320-slot index
            action_idx[plac_i] = slot
            kept_mask[plac_i] = True
            valid_mask[slot] = True

    return action_idx, kept_mask, valid_mask
