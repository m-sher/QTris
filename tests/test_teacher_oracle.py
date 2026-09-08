"""The demo's GPU oracle adapter against the C oracle on a live environment."""

import numpy as np
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.PyTetrisEnv import PyTetrisEnv

from qtris.data.placement_features import PAD
from qtris.search.gpu_oracle import GpuOracle
from qtris.search.placement_search import clone_sim_env

DEPTH, WIDTH, QUEUE, MAX_LEN = 2, 32, 5, 15
MOVES = 12
DUP_SEED = 0
SCORE_TOL = 0.02


def _env(seed=11):
    env = PyTetrisEnv(
        queue_size=QUEUE,
        max_holes=50,
        max_steps=None,
        max_len=MAX_LEN,
        pathfinding=True,
        garbage_chance=0.15,
        garbage_min=1,
        garbage_max=4,
        seed=seed,
        idx=0,
    )
    env.reset()
    return env


def _call(searcher, env):
    queue = np.array([p.value for p in env._queue], dtype=np.int32)
    return searcher.search_with_scores(
        board=env._board,
        active_piece=env._active_piece.piece_type.value,
        hold_piece=env._hold_piece.value,
        queue=queue,
        b2b=int(env._scorer._b2b),
        combo=int(env._scorer._combo),
        total_garbage=int(env._get_total_garbage()),
        garbage_push_delay=env._garbage_push_delay,
        search_depth=DEPTH,
        beam_width=WIDTH,
        max_len=MAX_LEN,
    )


def test_gpu_oracle_matches_the_c_oracle_on_a_played_game():
    env = _env()
    c_search = CB2BSearch()
    gpu = GpuOracle(
        env, search_depth=DEPTH, beam_width=WIDTH, queue_size=QUEUE, max_len=MAX_LEN
    )
    moves = 0
    for _ in range(MOVES):
        c_best, c_seq, c_acts, c_scores, _c_seqs, c_rows, c_val = _call(c_search, env)
        g_best, _g_seq, g_acts, g_scores, _g_seqs, g_rows, g_val = _call(gpu, env)
        if c_best < 0:
            break
        assert np.array_equal(g_acts, c_acts)
        assert np.array_equal(g_rows, c_rows)
        assert np.allclose(g_scores, c_scores, atol=SCORE_TOL)
        assert int(g_best) == int(c_best)
        assert abs(g_val - c_val) <= SCORE_TOL
        env._step(c_seq)
        moves += 1
        if env._is_top_out(env._board):
            break
    assert moves >= 8


def test_every_candidate_carries_a_real_key_sequence():
    env = _env()
    c_search = CB2BSearch()
    gpu = GpuOracle(
        env, search_depth=DEPTH, beam_width=WIDTH, queue_size=QUEUE, max_len=MAX_LEN
    )
    checked = 0
    for _ in range(MOVES):
        best, _seq, acts, _scores, seqs, _rows, _val = _call(gpu, env)
        if best < 0:
            break
        assert len(seqs) == len(acts)
        for row in seqs:
            assert not np.all(row == PAD), "candidate with no key sequence"
            checked += 1
        c_best, c_seq, *_ = _call(c_search, env)
        env._step(c_seq)
        if env._is_top_out(env._board):
            break
    assert checked >= 100


def test_a_duplicated_action_index_still_plays_the_chosen_landing_row():
    """Multi-landing placements share an action index, so the row picks the move."""
    env = _env(seed=DUP_SEED)
    c_search = CB2BSearch()
    gpu = GpuOracle(
        env, search_depth=DEPTH, beam_width=WIDTH, queue_size=QUEUE, max_len=MAX_LEN
    )
    duplicated = 0
    for _ in range(MOVES):
        c_best, c_seq, c_acts, _cs, _cq, c_rows, _cv = _call(c_search, env)
        g_best, g_seq, *_ = _call(gpu, env)
        if c_best < 0:
            break
        assert int(g_best) == int(c_best)
        hits = np.flatnonzero(np.asarray(c_acts, np.int32) == int(c_best))
        rows = {int(r) for r in np.asarray(c_rows)[hits]}
        duplicated += int(hits.size > 1 and len(rows) > 1)
        on_gpu, on_c = clone_sim_env(env), clone_sim_env(env)
        on_gpu._step(g_seq)
        on_c._step(c_seq)
        assert np.array_equal(on_gpu._board, on_c._board)
        env._step(c_seq)
        if env._is_top_out(env._board):
            break
    assert duplicated >= 1, "no position shared one action index across landing rows"


def test_the_chosen_sequence_reaches_the_chosen_placement():
    env = _env()
    c_search = CB2BSearch()
    gpu = GpuOracle(
        env, search_depth=DEPTH, beam_width=WIDTH, queue_size=QUEUE, max_len=MAX_LEN
    )
    compared = 0
    for _ in range(MOVES):
        c_best, c_seq, *_ = _call(c_search, env)
        g_best, g_seq, *_ = _call(gpu, env)
        if c_best < 0:
            break
        assert int(g_best) == int(c_best)
        on_gpu = clone_sim_env(env)
        on_gpu._step(g_seq)
        on_c = clone_sim_env(env)
        on_c._step(c_seq)
        assert np.array_equal(on_gpu._board, on_c._board)
        assert on_gpu._scorer._b2b == on_c._scorer._b2b
        assert on_gpu._scorer._combo == on_c._scorer._combo
        env._step(c_seq)
        compared += 1
        if env._is_top_out(env._board):
            break
    assert compared >= 8
