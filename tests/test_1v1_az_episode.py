import numpy as np
import pytest

from qtris.training._1v1_placement_az import (
    _episode,
    _grounding,
    _td_lambda,
)


def _pend(n1, n2, v=0.5, atk=0.0, b2b=0):
    return {
        "p1": [{"v_root": v, "atk": atk, "b2b": b2b} for _ in range(n1)],
        "p2": [{"v_root": v, "atk": atk, "b2b": b2b} for _ in range(n2)],
    }


def test_episode_emits_both_players_rows():
    rows, glen, p1_won, draw = _episode(_pend(6, 6), True, False, 0.9)
    assert len(rows) == 12
    assert glen == 6
    assert (p1_won, draw) == (False, False)
    assert [r[2] for r in rows] == [1.0] * 6 + [0.0] * 6


def test_episode_z_is_per_player_and_opposite():
    rows, *_ = _episode(_pend(4, 4), False, True, 0.9)
    assert {r[3] for r in rows if r[2] == 1.0} == {1.0}
    assert {r[3] for r in rows if r[2] == 0.0} == {-1.0}


def test_episode_steps_to_end_counts_down_per_trajectory():
    rows, *_ = _episode(_pend(4, 3), False, True, 0.9)
    assert [r[4] for r in rows if r[2] == 1.0] == [3, 2, 1, 0]
    assert [r[4] for r in rows if r[2] == 0.0] == [2, 1, 0]


def test_episode_targets_match_td_lambda():
    """The targets are the pure TD(lambda) targets, for both players."""
    pend = _pend(5, 5, v=0.25, atk=2.0, b2b=7)
    rows, *_ = _episode(pend, True, False, 0.7)
    exp1 = _td_lambda([p["v_root"] for p in pend["p1"]], -1.0, 0.7)
    exp2 = _td_lambda([p["v_root"] for p in pend["p2"]], 1.0, 0.7)
    assert [r[1] for r in rows if r[2] == 1.0] == pytest.approx(exp1)
    assert [r[1] for r in rows if r[2] == 0.0] == pytest.approx(exp2)


def test_episode_draw_when_both_die():
    rows, _glen, p1_won, draw = _episode(_pend(3, 3), True, True, 0.9)
    assert (p1_won, draw) == (False, True)
    assert {r[3] for r in rows} == {0.0}


def test_episode_none_when_empty():
    assert _episode(_pend(0, 0), True, False, 0.9) is None


def test_episode_uneven_trajectories():
    rows, glen, *_ = _episode(_pend(5, 0), True, False, 0.9)
    assert glen == 5
    assert len(rows) == 5


def test_grounding_buckets_by_steps_to_end():
    """Rows split by steps_to_end; empty and constant buckets yield None."""
    n = np.array([0, 1, 2, 70, 80, 90])
    z = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    v_root = np.array([0.9, -0.9, 0.9, 0.1, 0.1, 0.1])
    g = _grounding(v_root, z, n)
    assert g["corr_n0_10"] == pytest.approx(1.0)
    assert g["corr_n60plus"] is None  # v_root constant -> undefined
    assert g["brier_n0_10"] < g["brier_n60plus"]
    assert g["corr_n10_30"] is None and g["brier_n10_30"] is None


def test_grounding_scores_against_outcome_not_target():
    n = np.array([0, 1])
    g = _grounding(np.array([1.0, -1.0]), np.array([-1.0, 1.0]), n)
    assert g["brier_n0_10"] == pytest.approx(1.0)
    assert g["corr_n0_10"] == pytest.approx(-1.0)


def test_grounding_draws_map_to_one_half():
    n = np.array([0, 1])
    g = _grounding(np.array([0.0, 0.0]), np.array([0.0, 0.0]), n)
    assert g["brier_n0_10"] == pytest.approx(0.0)
