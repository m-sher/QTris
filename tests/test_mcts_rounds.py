"""Every simulation backs up exactly once, and the Q-normalisation flag reaches the search."""

import numpy as np
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from qtris.search.cmcts import CMCTS, CANDIDATE_CAPACITY

SIMS = 64
LPR = 8


def _engine(q_norm):
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
        w_attack=0.0,
        w_death=1.0,
        return_scale=1.0,
        max_len=15,
        num_simulations=SIMS,
        leaves_per_round=LPR,
        vloss=1.0,
        w_b2b=0.0,
        q_norm=q_norm,
    )


def _played_env(seed, moves=30):
    env = PyTetrisEnv(
        queue_size=5,
        max_holes=50,
        max_steps=None,
        max_len=15,
        pathfinding=False,
        garbage_chance=0.0,
        auto_push_garbage=False,
        auto_fill_queue=True,
        seed=seed,
        idx=0,
    )
    env.reset()
    searcher = CB2BSearch()
    for _ in range(moves):
        idx, seq, *_ = searcher.search_with_scores(
            board=env._board,
            active_piece=env._active_piece.piece_type.value,
            hold_piece=env._hold_piece.value,
            queue=np.array([p.value for p in env._queue], dtype=np.int32),
            b2b=env._scorer._b2b,
            combo=env._scorer._combo,
            total_garbage=env._get_total_garbage(),
            garbage_push_delay=env._garbage_push_delay,
            search_depth=4,
            beam_width=32,
            max_len=15,
        )
        if idx < 0:
            break
        env._step(seq)
    return env


def _root_counts(q_norm, seed, leaf_values):
    """Drive one search with flat priors and constant leaf values, so the descent is fully
    deterministic and every round's descents would otherwise collide on one leaf."""
    env = _played_env(seed)
    engine = _engine(q_norm)
    try:
        engine.set_root(0, env)
        nv, _ = engine.collect_roots()
        assert nv == 1
        zeros = np.zeros(nv * CANDIDATE_CAPACITY, np.float32)
        engine.apply_roots(zeros, np.zeros(nv, np.float32), zeros.copy(), 0.0)
        rounds = (SIMS + LPR - 1) // LPR
        for _ in range(rounds):
            nv2, _ = engine.collect_leaves()
            engine.apply_leaves(
                np.zeros(nv2 * CANDIDATE_CAPACITY, np.float32),
                np.full(nv2, leaf_values, np.float32),
                np.full(nv2, leaf_values, np.float32),
            )
        _pi, counts, _desc, dead, _rv = engine.result()
        assert not dead[0]
        return np.array(counts[0], dtype=np.float64)
    finally:
        engine.destroy()


def test_num_simulations_is_a_count():
    """Root visits sum to num_simulations exactly, collisions included."""
    for seed in (7, 11):
        counts = _root_counts(False, seed, leaf_values=0.0)
        assert counts.sum() == SIMS, counts.sum()


def test_q_norm_reaches_the_search():
    """With distinct leaf values in play the normalised ranking visits differently."""
    off = _root_counts(False, 7, leaf_values=0.3)
    on = _root_counts(True, 7, leaf_values=0.3)
    assert off.sum() == SIMS and on.sum() == SIMS
    assert not np.array_equal(off, on)


def test_q_norm_is_the_default_everywhere():
    """Every pipeline that builds a search without naming q_norm gets it."""
    import inspect
    from qtris.search.placement_mcts import MCTSConfig

    assert MCTSConfig().q_norm is True
    assert inspect.signature(CMCTS.__init__).parameters["q_norm"].default is True
