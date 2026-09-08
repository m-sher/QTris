"""The oracle demo plays the same game on the C engine and on the GPU teacher."""

import types

import numpy as np
from qtris.demo import oracle as od

TURNS = 20


def _args(kind):
    return types.SimpleNamespace(
        oracle=kind,
        seed=17,
        queue_size=5,
        max_len=15,
        garbage_chance=0.15,
        search_depth=3,
        beam_width=32,
        num_steps=TURNS,
        fps=240,
        dist_temp=1.0,
    )


def _play(kind):
    """Per-turn trace of a game driven entirely by the named oracle."""
    args = _args(kind)
    env = od.make_env(args, args.seed)
    env.reset()
    engine = od.build_oracle(
        kind, env, args.search_depth, args.beam_width, args.queue_size, args.max_len
    )
    stats = od.Stats()
    trace = []
    for _ in range(TURNS):
        action, sequence, cand_actions, cand_scores, _seqs, _rows, _v = od.search(
            engine, env, args
        )
        if action < 0:
            break
        attack, clears = od.commit(env, sequence, stats)
        trace.append(
            (
                int(action),
                len(cand_actions),
                attack,
                clears,
                int(env._scorer._b2b),
                int(env._scorer._combo),
                float(np.asarray(cand_scores, np.float32).max()),
            )
        )
        if env._is_top_out(env._board):
            break
    return trace, stats


def test_the_gpu_teacher_plays_the_same_game_as_the_c():
    c_trace, c_stats = _play("c")
    g_trace, g_stats = _play("gpu")
    assert len(c_trace) >= 15
    assert len(g_trace) == len(c_trace)
    for turn, (c, g) in enumerate(zip(c_trace, g_trace, strict=True)):
        assert c[:6] == g[:6], f"turn {turn}: C {c[:6]} GPU {g[:6]}"
        assert abs(c[6] - g[6]) <= 0.02, f"turn {turn}: best score {c[6]} vs {g[6]}"
    assert c_stats.app == g_stats.app
    assert c_stats.max_b2b == g_stats.max_b2b


def test_the_oracle_demo_needs_no_checkpoint_and_makes_progress():
    trace, stats = _play("gpu")
    assert stats.pieces == len(trace)
    assert stats.clears > 0, "the oracle cleared no lines in 20 moves"
    assert all(count > 0 for _a, count, *_rest in trace)


def test_the_candidate_distribution_is_a_distribution():
    args = _args("gpu")
    env = od.make_env(args, args.seed)
    env.reset()
    engine = od.build_oracle("gpu", env, 3, 32, 5, 15)
    _a, _s, _ca, cand_scores, _sq, _r, _v = od.search(engine, env, args)
    probs = od.candidate_distribution(cand_scores, args.dist_temp)
    assert probs.shape == (len(cand_scores),)
    assert np.isclose(probs.sum(), 1.0)
    assert (probs >= 0).all()
    assert od.candidate_distribution(np.zeros(0, np.float32), 30.0).size == 0
