"""The shaping-free root value readout: leaf values in, shaping weights kept out."""

import numpy as np
import pytest
from qtris.search.cmcts import CMCTS, CANDIDATE_CAPACITY

from test_mcts_rounds import _played_env

SIMS = 64
LPR = 8
LEAF = 0.3


def _run(seed, **weights):
    """Flat priors, no noise, every leaf valued LEAF. At this budget no descent reaches a
    terminal or exhausts the arena, so every backup carries LEAF and the root readout must
    equal it exactly."""
    env = _played_env(seed, 30)
    engine = CMCTS(
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
        **weights,
    )
    try:
        engine.set_root(0, env)
        nv, _ = engine.collect_roots()
        assert nv == 1
        zeros = np.zeros(nv * CANDIDATE_CAPACITY, np.float32)
        engine.apply_roots(zeros, np.zeros(nv, np.float32), zeros.copy(), 0.0)
        rounds = (SIMS + LPR - 1) // LPR
        for _ in range(rounds):
            nv2, _ = engine.collect_leaves()
            if nv2 == 0:
                break
            engine.apply_leaves(
                np.zeros(nv2 * CANDIDATE_CAPACITY, np.float32),
                np.full(nv2, LEAF, np.float32),
            )
        _pi, counts, _desc, dead, root_value = engine.result()
        assert not dead[0]
        return np.array(counts[0], np.float64), float(root_value[0])
    finally:
        engine.destroy()


def test_root_value_is_the_backed_up_leaf_value():
    for seed in (7, 11):
        _counts, rv = _run(seed, w_attack=0.0, w_b2b=0.0, w_height=0.0, w_bumpiness=0.0)
        assert rv == pytest.approx(LEAF, abs=1e-4)


def test_root_value_excludes_the_shaping_weights():
    base_counts, base_rv = _run(
        7, w_attack=0.0, w_b2b=0.0, w_height=0.0, w_bumpiness=0.0
    )
    counts, rv = _run(7, w_attack=5.0, w_b2b=5.0, w_height=5.0, w_bumpiness=5.0)
    assert not np.array_equal(base_counts, counts)
    assert rv == pytest.approx(base_rv, abs=1e-4)
