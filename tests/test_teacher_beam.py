"""The batched GPU beam against the C oracle, per search config."""

import numpy as np
import pytest
from teacher_common import load

from teacher.api import GpuTeacher

_COL_BITS = (np.uint16(1) << np.arange(10, dtype=np.uint16)).astype(np.uint16)

# tag, depth, width, queue_len, position stride
EXACT_CONFIGS = [
    ("d1w32", 1, 32, 10, 1),
    ("d2w128", 2, 128, 10, 1),
    ("d2w32", 2, 32, 10, 1),
    ("d4w32", 4, 32, 10, 2),
]
SPECULATIVE_CONFIGS = [
    ("d4w32q2", 4, 32, 2, 2),
    ("d16w64q5", 16, 64, 5, 4),
]
SCORE_TOL = 0.02


def grid_of(masks):
    """Float32 occupancy grid of row bitmasks."""
    rows = np.asarray(masks, np.uint16)[..., None]
    return ((rows & _COL_BITS) > 0).astype(np.float32)


def run(teacher, d, i, depth, width, qlen):
    """One GPU search on fixture position i."""
    return teacher.search_with_scores(
        grid_of(d["boards"][i]),
        int(d["active"][i]),
        int(d["hold"][i]),
        d["queues"][i][:qlen],
        int(d["b2b"][i]),
        int(d["combo"][i]),
        int(d["garbage"][i]),
        search_depth=depth,
        beam_width=width,
        max_roots=1024,
    )


def compare(tag, depth, width, qlen, stride):
    """Per-position agreement counts against the C fixture."""
    d = load()
    teacher = GpuTeacher(max_batch=1, width=width, depth=depth, queue_len=qlen)
    n_pos = same_roots = same_scores = same_best = 0
    worst = 0.0
    for i in range(0, len(d["boards"]), stride):
        _a, _p, ra, rs, _rp, rr, _v = run(teacher, d, i, depth, width, qlen)
        n = int(d[f"n_{tag}"][i])
        ca = d[f"acts_{tag}"][i, :n]
        cs = d[f"scores_{tag}"][i, :n]
        crow = d[f"rows_{tag}"][i, :n]
        n_pos += 1
        roots_match = len(ra) == n
        roots_match = roots_match and np.array_equal(np.asarray(ra, np.int32), ca)
        roots_match = roots_match and np.array_equal(np.asarray(rr, np.int32), crow)
        same_roots += roots_match
        if roots_match and n:
            err = np.abs(np.asarray(rs, np.float32) - cs)
            worst = max(worst, float(err.max()))
            same_scores += int((err <= SCORE_TOL).all())
        elif roots_match:
            same_scores += 1
        same_best += int(_a) == int(d[f"best_{tag}"][i])
    return n_pos, same_roots, same_scores, same_best, worst


@pytest.mark.parametrize(("tag", "depth", "width", "qlen", "stride"), EXACT_CONFIGS)
def test_beam_matches_the_c_exactly(tag, depth, width, qlen, stride):
    n, roots, scores, best, worst = compare(tag, depth, width, qlen, stride)
    assert n > 100
    assert roots == n, f"{tag}: {n - roots} positions differ on roots or rows"
    assert scores == n, f"{tag}: root scores differ on {n - scores} positions"
    assert best == n, f"{tag}: chosen move differs on {n - best} positions"
    assert worst <= SCORE_TOL


@pytest.mark.parametrize(
    ("tag", "depth", "width", "qlen", "stride"), SPECULATIVE_CONFIGS
)
def test_speculative_beam_matches_up_to_ties(tag, depth, width, qlen, stride):
    n, roots, scores, best, _worst = compare(tag, depth, width, qlen, stride)
    assert n > 50
    assert roots == n, f"{tag}: {n - roots} positions differ on roots or rows"
    assert scores >= n - max(2, n // 50), f"{tag}: {n - scores} positions differ"
    assert best >= n - max(2, n // 50), f"{tag}: {n - best} chosen moves differ"


def test_no_buffer_overflow_at_any_config():
    d = load()
    for tag, depth, width, qlen, _stride in EXACT_CONFIGS + SPECULATIVE_CONFIGS:
        sel = range(0, len(d["boards"]), 40)
        teacher = GpuTeacher(
            max_batch=len(sel), width=width, depth=depth, queue_len=qlen
        )
        result = teacher.search_batch(
            grid_of(d["boards"][list(sel)]),
            d["active"][list(sel)],
            d["hold"][list(sel)],
            d["queues"][list(sel), :qlen],
            qlen,
            d["b2b"][list(sel)],
            d["combo"][list(sel)],
            d["garbage"][list(sel)],
        )
        assert not result.placement_overflow, tag
        assert not result.pool_overflow, tag
        assert not result.workitem_overflow, tag


def test_a_fully_blocked_spawn_reports_no_move():
    d = load()
    teacher = GpuTeacher(max_batch=1, width=32, depth=1, queue_len=10)
    blocked = [i for i in range(len(d["boards"])) if int(d["best_d1w32"][i]) == -1]
    assert blocked, "fixture has no spawn-blocked position"
    for i in blocked:
        action = run(teacher, d, i, 1, 32, 10)[0]
        assert int(action) == -1
