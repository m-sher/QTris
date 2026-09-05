"""Root children as value rows: their shaping-free readouts, visit filter and cap, and
the search driver's `siblings` result."""

import numpy as np
import pytest
import tensorflow as tf
from qtris.search.cmcts import CMCTS, CANDIDATE_CAPACITY
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS

from test_mcts_rounds import _played_env

SIMS = 64
LPR = 8
LEAF = 0.3


def _engine():
    return CMCTS(
        1,
        board_height=40,
        queue_size=5,
        max_holes=50,
        garbage_push_delay=1,
        auto_push_garbage=0,
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
    )


def _raise_stack(env, height, gap=3):
    """Fill the bottom `height` rows except one column, so placements near the spawn box
    can die inside the tree."""
    rows = env._board.shape[0]
    env._board[rows - height :, :] = 1
    env._board[rows - height :, gap] = 0
    env._vis_board[rows - height :, :] = env._board[rows - height :, :]


def _search(env, max_k, min_n):
    """Flat priors, no noise, constant leaf value LEAF on both channels; returns the
    root counts, the row count and the collected rows."""
    engine = _engine()
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
        _pi, counts, _desc, dead, _rv = engine.result()
        assert not dead[0]
        ns, rows = engine.collect_root_children(max_k, min_n)
        return np.array(counts[0], np.float64), ns, rows
    finally:
        engine.destroy()


def test_children_carry_the_leaf_value_when_nothing_dies():
    for seed in (7, 11):
        counts, ns, rows = _search(_played_env(seed, 30), 64, 1.0)
        assert ns > 1
        assert rows[5] == pytest.approx(np.full(ns, LEAF), abs=1e-4)
        assert ns == int((counts >= 1.0).sum())
        assert np.all(rows[6] >= 1.0)
        assert np.all(np.diff(rows[6]) <= 0.0)
        assert np.all(rows[7] == 0)
        assert rows[0].shape == (ns, 24, 10, 1)
        assert rows[4].dtype == bool


def test_visit_floor_and_cap_bound_the_rows():
    env = _played_env(7, 30)
    counts, ns_all, _ = _search(env, 64, 1.0)
    _, ns_min, rows = _search(_played_env(7, 30), 64, 3.0)
    assert ns_min == int((counts >= 3.0).sum()) < ns_all
    assert np.all(rows[6] >= 3.0)
    _, ns_cap, rows = _search(_played_env(7, 30), 2, 1.0)
    assert ns_cap == 2
    assert rows[6][0] == counts.max()


def test_a_child_whose_subtree_dies_reads_below_the_leaf_value():
    env = _played_env(7, 30)
    _raise_stack(env, 19)
    _counts, ns, rows = _search(env, 64, 1.0)
    assert ns > 0
    assert rows[5].min() < LEAF - 0.1
    assert rows[5].max() <= LEAF + 1e-4


class _FlatNet:
    def policy_value(self, inputs):
        n = int(inputs[0].shape[0])
        return (
            tf.zeros((n, CANDIDATE_CAPACITY), tf.float32),
            tf.fill((n, 1), LEAF),
            tf.zeros((n, 1), tf.float32),
        )


def _cfg(sibling_max):
    return MCTSConfig(
        num_simulations=SIMS,
        dirichlet_eps=0.0,
        leaves_per_round=LPR,
        gamma=1.0,
        w_death=1.0,
        attack_window=0,
        sibling_max=sibling_max,
        sibling_min_visits=2.0,
    )


def test_search_returns_siblings_only_when_asked():
    off = PlacementMCTS(_FlatNet(), _cfg(0)).search([_played_env(7, 30)], 1.0, 0.0)[0]
    assert off["siblings"] == []
    on = PlacementMCTS(_FlatNet(), _cfg(4)).search([_played_env(7, 30)], 1.0, 0.0)[0]
    assert 0 < len(on["siblings"]) <= 4
    for s in on["siblings"]:
        assert s["board"].shape == (24, 10, 1)
        assert s["pieces"].shape == (7,)
        assert s["bcg"].shape == (3,)
        assert s["cand_placements"].shape == (CANDIDATE_CAPACITY, 18)
        assert s["cand_mask"].dtype == bool and s["cand_mask"].any()
        assert s["n"] >= 2.0
        assert s["v_out"] == pytest.approx(LEAF, abs=1e-4)
    assert np.array_equal(off["counts"], on["counts"])
