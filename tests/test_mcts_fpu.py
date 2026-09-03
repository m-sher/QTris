"""First-play urgency reaches the search and narrows a flat root; fpu < 0 scores 0."""

import inspect

import numpy as np
from qtris.search.cmcts import CMCTS, CANDIDATE_CAPACITY
from qtris.search.placement_mcts import MCTSConfig

from test_mcts_rounds import _played_env

SIMS = 64
LPR = 8


def _root_counts(fpu, seed):
    """Flat priors and zero leaf values: without first-play urgency every child ties and the
    exploration term visits each once; with it a visited child outranks an unvisited one."""
    env = _played_env(seed)
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
        w_attack=0.0,
        w_death=1.0,
        return_scale=1.0,
        max_len=15,
        num_simulations=SIMS,
        leaves_per_round=LPR,
        vloss=1.0,
        w_b2b=0.0,
        q_norm=True,
        fpu=fpu,
    )
    try:
        engine.set_root(0, env)
        nv, req = engine.collect_roots()
        assert nv == 1
        n_legal = int(req[4][0].sum())
        zeros = np.zeros(nv * CANDIDATE_CAPACITY, np.float32)
        engine.apply_roots(zeros, np.zeros(nv, np.float32), zeros.copy(), 0.0)
        for _ in range((SIMS + LPR - 1) // LPR):
            nv2, _ = engine.collect_leaves()
            engine.apply_leaves(
                np.zeros(nv2 * CANDIDATE_CAPACITY, np.float32),
                np.zeros(nv2, np.float32),
            )
        _pi, counts, _desc, dead, _rv = engine.result()
        assert not dead[0]
        return n_legal, np.array(counts[0], dtype=np.float64)
    finally:
        engine.destroy()


def test_fpu_narrows_a_flat_root():
    for seed in (7, 11):
        n_legal, off = _root_counts(-1.0, seed)
        _n, on = _root_counts(0.2, seed)
        assert n_legal <= SIMS
        assert off.sum() == SIMS and on.sum() == SIMS
        assert int((off > 0).sum()) == n_legal, (n_legal, int((off > 0).sum()))
        assert int((on > 0).sum()) < n_legal, (n_legal, int((on > 0).sum()))


def test_fpu_defaults():
    """MCTSConfig searches with first-play urgency; the raw engine scores unvisited
    children 0."""
    assert MCTSConfig().fpu == 0.4
    assert inspect.signature(CMCTS.__init__).parameters["fpu"].default < 0
