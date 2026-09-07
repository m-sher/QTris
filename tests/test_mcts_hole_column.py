"""Pending garbage lands inside the tree at a hole column the tree draws itself: the
column the env stored is invisible, the landing still happens, siblings draw different
columns, and four-wide draws inside columns 3-6."""

import numpy as np
from qtris.search.cmcts import CMCTS, CANDIDATE_CAPACITY
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.Pieces import PieceType

from test_mcts_rounds import _played_env

SIMS = 64
LPR = 8
LEAF = 0.3


def _engine(four_wide=False):
    return CMCTS(
        1,
        board_height=40,
        queue_size=5,
        max_holes=50,
        garbage_push_delay=1,
        auto_push_garbage=1,
        auto_fill_queue=1,
        c_puct=1.5,
        gamma=1.0,
        w_death=1.0,
        return_scale=1.0,
        max_len=15,
        num_simulations=SIMS,
        leaves_per_round=LPR,
        vloss=1.0,
        q_norm=True,
        w_attack=0.0,
        w_b2b=0.0,
        w_height=0.0,
        w_bumpiness=0.0,
        four_wide=four_wide,
    )


def _raise_stack(env, height, gap=3):
    rows = env._board.shape[0]
    env._board[rows - height :, :] = 1
    env._board[rows - height :, gap] = 0
    env._vis_board[rows - height :, :] = env._board[rows - height :, :]


def _search(env, four_wide=False):
    """Flat priors, no noise, constant leaf values; returns the root counts, the root
    readout and the root children rows (every visited non-terminal child)."""
    engine = _engine(four_wide)
    try:
        engine.set_root(0, env)
        nv, _ = engine.collect_roots()
        assert nv == 1
        zeros = np.zeros(nv * CANDIDATE_CAPACITY, np.float32)
        engine.apply_roots(zeros, np.zeros(nv, np.float32), zeros.copy(), 0.0)
        for _ in range((SIMS + LPR - 1) // LPR):
            nv2, _ = engine.collect_leaves()
            if nv2 == 0:
                break
            engine.apply_leaves(
                np.zeros(nv2 * CANDIDATE_CAPACITY, np.float32),
                np.full(nv2, LEAF, np.float32),
                np.full(nv2, LEAF, np.float32),
            )
        _pi, counts, _desc, dead, root_value = engine.result()
        assert not dead[0]
        _ns, rows = engine.collect_root_children(CANDIDATE_CAPACITY, 1.0)
        return np.array(counts[0], np.float64), float(root_value[0]), rows
    finally:
        engine.destroy()


def _landed_columns(rows, garbage_rows):
    """Hole column of the bottom row of each child board carrying the landed entry."""
    cols = []
    for b in rows[0]:
        bottom = b[-garbage_rows:, :, 0]
        if bottom.sum() == garbage_rows * 9:
            cols.append(int(np.flatnonzero(bottom[-1] == 0)[0]))
    return cols


def test_the_stored_column_is_invisible_to_the_tree():
    env_a, env_b = _played_env(7, 30), _played_env(7, 30)
    _raise_stack(env_a, 14)
    _raise_stack(env_b, 14)
    env_a._garbage_queue = [(3, 1, 0)]
    env_b._garbage_queue = [(3, 8, 0)]
    counts_a, rv_a, rows_a = _search(env_a)
    counts_b, rv_b, rows_b = _search(env_b)
    assert np.array_equal(counts_a, counts_b)
    assert rv_a == rv_b
    assert np.array_equal(rows_a[0], rows_b[0])


def test_the_landing_still_happens_and_moves_the_search():
    with_garbage = _played_env(7, 30)
    _raise_stack(with_garbage, 14)
    with_garbage._garbage_queue = [(3, 1, 0)]
    clean = _played_env(7, 30)
    _raise_stack(clean, 14)
    counts_g, rv_g, rows_g = _search(with_garbage)
    counts_c, _rv_c, _rows_c = _search(clean)
    assert not np.array_equal(counts_g, counts_c)
    assert rv_g < LEAF
    assert len(_landed_columns(rows_g, 3)) >= 3


def test_siblings_land_the_same_entry_at_different_columns():
    env = _played_env(7, 30)
    _raise_stack(env, 12)
    env._garbage_queue = [(3, 1, 0)]
    _counts, _rv, rows = _search(env)
    cols = _landed_columns(rows, 3)
    assert len(cols) >= 5
    assert len(set(cols)) >= 2
    assert set(cols) <= set(range(10))


def test_four_wide_draws_inside_the_playable_columns():
    env = PyTetrisEnv(
        queue_size=5,
        max_holes=50,
        max_steps=None,
        max_len=15,
        pathfinding=False,
        garbage_chance=0.0,
        auto_push_garbage=False,
        auto_fill_queue=True,
        seed=3,
        idx=0,
        four_wide=True,
    )
    env.reset()
    env._active_piece = env._spawn_piece(PieceType.J)
    env._hold_piece = PieceType.J
    env._queue = [PieceType.J] * 5
    env._garbage_queue = [(2, 5, 0)]
    _counts, _rv, rows = _search(env, four_wide=True)
    cols = []
    for b in rows[0]:
        bottom = b[-2:, 3:7, 0]
        if bottom.sum() == 2 * 3:
            cols.append(3 + int(np.flatnonzero(bottom[-1] == 0)[0]))
    assert len(cols) >= 3
    assert set(cols) <= {3, 4, 5, 6}
