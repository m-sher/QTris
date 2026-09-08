"""Batching a set of positions gives the same result as searching each alone."""

import numpy as np
from teacher_common import load
from test_teacher_beam import grid_of, run

from teacher.api import GpuTeacher

DEPTH, WIDTH, QLEN = 2, 32, 10
STRIDE = 7


def _positions(d):
    return list(range(0, len(d["boards"]), STRIDE))


def _batched(d, sel):
    teacher = GpuTeacher(max_batch=len(sel), width=WIDTH, depth=DEPTH, queue_len=QLEN)
    return teacher.search_batch(
        grid_of(d["boards"][sel]),
        d["active"][sel],
        d["hold"][sel],
        d["queues"][sel, :QLEN],
        QLEN,
        d["b2b"][sel],
        d["combo"][sel],
        d["garbage"][sel],
    )


def test_batched_search_equals_single_position_search():
    d = load()
    sel = _positions(d)
    result = _batched(d, sel)
    single = GpuTeacher(max_batch=1, width=WIDTH, depth=DEPTH, queue_len=QLEN)
    for g, i in enumerate(sel):
        action, _pl, ra, rs, _rp, rr, _v = run(single, d, i, DEPTH, WIDTH, QLEN)
        n = int(result.root_count[g])
        assert n == len(ra), f"position {i}: batched {n} roots, single {len(ra)}"
        assert int(result.action[g]) == int(action), f"position {i}"
        assert np.array_equal(result.root_action[g, :n], np.asarray(ra, np.int32))
        assert np.array_equal(result.root_row[g, :n], np.asarray(rr, np.int32))
        assert np.allclose(result.root_score[g, :n], np.asarray(rs, np.float32))


def test_batched_search_matches_the_c_oracle():
    d = load()
    sel = _positions(d)
    result = _batched(d, sel)
    for g, i in enumerate(sel):
        n = int(d["n_d2w32"][i])
        assert int(result.root_count[g]) == n, f"position {i}"
        assert np.array_equal(result.root_action[g, :n], d["acts_d2w32"][i, :n])
        assert np.allclose(
            result.root_score[g, :n], d["scores_d2w32"][i, :n], atol=0.02
        )
        assert int(result.action[g]) == int(d["best_d2w32"][i]), f"position {i}"


def test_a_batch_mixing_live_dead_and_blocked_games_is_correct():
    d = load()
    live = [i for i in range(len(d["boards"])) if int(d["n_d2w32"][i]) > 20][:4]
    dead = [i for i in range(len(d["boards"])) if int(d["n_d2w32"][i]) == 0][:4]
    blocked = [i for i in range(len(d["boards"])) if int(d["best_d2w32"][i]) == -1][:4]
    sel = live + dead + blocked
    assert len(live) and len(dead) and len(blocked)
    result = _batched(d, sel)
    for g, i in enumerate(sel):
        assert int(result.root_count[g]) == int(d["n_d2w32"][i]), f"position {i}"
        assert int(result.action[g]) == int(d["best_d2w32"][i]), f"position {i}"
    assert not result.placement_overflow
    assert not result.pool_overflow
    assert not result.workitem_overflow


def test_batch_order_does_not_matter():
    d = load()
    sel = _positions(d)[:12]
    forward = _batched(d, sel)
    reverse = _batched(d, sel[::-1])
    for g in range(len(sel)):
        h = len(sel) - 1 - g
        assert int(forward.action[g]) == int(reverse.action[h])
        n = int(forward.root_count[g])
        assert n == int(reverse.root_count[h])
        assert np.array_equal(forward.root_action[g, :n], reverse.root_action[h, :n])
        assert np.allclose(forward.root_score[g, :n], reverse.root_score[h, :n])
