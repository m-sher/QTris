"""The per-edge board-quality penalty reaches the search with the documented normalisers."""

import numpy as np
import pytest
from TetrisEnv.Pieces import PieceType
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from qtris.search.cmcts import CMCTS, CANDIDATE_CAPACITY
from qtris.search.placement_mcts import MCTSConfig

HEIGHT_MAX = 24.0
BUMPINESS_MAX = 48.0


def _env(seed, moves=10):
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


def _run(env, sims, w_height, w_bumpiness):
    """One search with flat priors and zero leaf values, so a root edge's Q after its single
    visit is exactly that edge's reward. Returns the root visit counts and descriptors."""
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
        w_death=10.0,
        return_scale=1.0,
        max_len=15,
        num_simulations=sims,
        leaves_per_round=1,
        vloss=1.0,
        w_b2b=0.0,
        q_norm=False,
        w_height=w_height,
        w_bumpiness=w_bumpiness,
    )
    try:
        engine.set_root(0, env)
        nv, _ = engine.collect_roots()
        assert nv == 1
        zeros = np.zeros(nv * CANDIDATE_CAPACITY, np.float32)
        engine.apply_roots(zeros, np.zeros(nv, np.float32), zeros.copy(), 0.0)
        for _ in range(sims):
            nv2, _ = engine.collect_leaves()
            if nv2 == 0:
                break
            engine.apply_leaves(
                np.zeros(nv2 * CANDIDATE_CAPACITY, np.float32),
                np.zeros(nv2, np.float32),
            )
        _pi, counts, desc, dead = engine.result()
        assert not dead[0]
        return np.array(counts[0], np.float64), np.array(desc[0], np.int64)
    finally:
        engine.destroy()


def _child_stats(env, desc, slots):
    """Max height and bumpiness of each candidate's resulting board, locked through the same
    C core the search steps with, so spins and clears land identically."""
    searcher = CB2BSearch()
    out = {}
    for slot in slots:
        is_hold, rot, norm_col, landing_row, spin = (int(x) for x in desc[slot])
        if is_hold:
            placed = (
                env._queue[0] if env._hold_piece == PieceType.N else env._hold_piece
            )
        else:
            placed = env._active_piece.piece_type
        board, *_ = searcher.lock_score(
            env._board,
            placed.value,
            rot,
            norm_col,
            landing_row,
            spin,
            env._scorer._b2b,
            env._scorer._combo,
        )
        occ = board != 0
        heights = np.where(occ.any(axis=0), occ.shape[0] - occ.argmax(axis=0), 0)
        out[slot] = (int(heights.max()), int(np.abs(np.diff(heights)).sum()))
    return out


def _extra_visit_slot(counts, slots):
    """The one root edge visited twice: with uniform priors and equal visit counts, the last
    simulation goes to the highest-Q edge."""
    top = [s for s in slots if counts[s] > 1]
    assert len(top) == 1, counts[slots]
    return top[0]


# The last case is the one whose cheapest child depends on the ratio of the two normalisers.
@pytest.mark.parametrize(
    "seed,moves,w_height,w_bumpiness",
    [(7, 10, 1.0, 0.0), (7, 10, 0.0, 1.0), (5, 40, 2.0, 1.0)],
)
def test_penalty_picks_the_cheapest_child(seed, moves, w_height, w_bumpiness):
    env = _env(seed, moves)
    counts, desc = _run(env, sims=1, w_height=0.0, w_bumpiness=0.0)
    slots = [s for s in range(CANDIDATE_CAPACITY) if desc[s, 0] >= 0]
    counts, desc = _run(
        env, sims=len(slots) + 1, w_height=w_height, w_bumpiness=w_bumpiness
    )
    stats = _child_stats(env, desc, slots)
    cost = {
        s: w_height * min(1.0, h / HEIGHT_MAX)
        + w_bumpiness * min(1.0, b / BUMPINESS_MAX)
        for s, (h, b) in stats.items()
    }
    cheapest = min(cost.values())
    picked = _extra_visit_slot(counts, slots)
    assert cost[picked] == pytest.approx(cheapest, abs=1e-6), (
        picked,
        cost[picked],
        cost,
    )


def test_penalty_changes_the_visit_distribution():
    env = _env(11)
    off, _ = _run(env, sims=64, w_height=0.0, w_bumpiness=0.0)
    on, _ = _run(env, sims=64, w_height=1.0, w_bumpiness=1.0)
    assert off.sum() == 64 and on.sum() == 64
    assert not np.array_equal(off, on)


def test_shaping_weights_are_on_by_default():
    """Every pipeline builds its search from these defaults and overrides none of them."""
    cfg = MCTSConfig()
    assert (cfg.w_attack, cfg.w_b2b, cfg.w_height, cfg.w_bumpiness) == (
        0.05,
        0.05,
        0.05,
        0.05,
    )
