"""Placement feature contract for the candidate-ranking (placement) model.

Each candidate move is an 18-dim placement descriptor (fusion-style: the piece
and where it locks, no resulting board / lines / attack). Candidates are packed
into a fixed 128-slot tensor: slots 0:64 are no-hold placements, 64:128 are hold
placements; within each branch the top-64 by search score are kept and the rest
are SENTINEL-scored (masked at train time). This module is the single source of
truth shared by datagen and any runtime/demo candidate builder.
"""

import numpy as np

PLACEMENT_FEATURE_DIM = 18
BRANCH_CAPACITY = 64
CANDIDATE_CAPACITY = 2 * BRANCH_CAPACITY  # 128
SENTINEL = np.float32(-1e30)  # score for empty/masked slots

# 18-dim layout: piece onehot[0:7] | rot onehot[7:11] | col[11] | row[12]
#                | spin onehot[13:17] | hold[17]


def encode_placement_features(
    piece_value, rot, norm_col, landing_row, spin_type, is_hold, row_norm
):
    """One placement descriptor. `piece_value` is the C enum (1..7, I,J,L,O,S,T,Z)."""
    f = np.zeros(PLACEMENT_FEATURE_DIM, dtype=np.float32)
    if 1 <= piece_value <= 7:
        f[piece_value - 1] = 1.0
    f[7 + rot] = 1.0
    f[11] = norm_col / 9.0
    f[12] = min(landing_row / row_norm, 1.0) if row_norm > 0 else 0.0
    f[13 + spin_type] = 1.0
    f[17] = float(is_hold)
    return f


def build_placement_target(
    cand_actions, cand_scores, cand_rows, active_piece, hold_piece, queue0, row_norm
):
    """Pack per-root search candidates into the 128-slot placement target.

    Returns (placements[128,18] f32, scores[128] f32). `cand_actions` are the
    dense action indices (is_hold*160 + rot*40 + norm_col*4 + spin)."""
    placements = np.zeros((CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM), dtype=np.float32)
    scores = np.full(CANDIDATE_CAPACITY, SENTINEL, dtype=np.float32)

    actions = np.asarray(cand_actions, dtype=np.int64)
    cand_scores = np.asarray(cand_scores, dtype=np.float32)
    is_hold = actions // 160
    rem = actions % 160
    rot, norm_col, spin = rem // 40, (rem % 40) // 4, rem % 4

    for branch in (0, 1):
        sel = np.flatnonzero(is_hold == branch)
        if sel.size == 0:
            continue
        order = sel[np.argsort(cand_scores[sel])[::-1]][:BRANCH_CAPACITY]
        piece = (
            active_piece if branch == 0 else (hold_piece if hold_piece != 0 else queue0)
        )
        base = branch * BRANCH_CAPACITY
        for rank, ci in enumerate(order):
            placements[base + rank] = encode_placement_features(
                piece,
                int(rot[ci]),
                int(norm_col[ci]),
                int(cand_rows[ci]),
                int(spin[ci]),
                branch,
                row_norm,
            )
            scores[base + rank] = cand_scores[ci]
    return placements, scores
