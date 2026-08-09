import numpy as np
import pytest

from qtris.training._1v1_placement_az import _episode, _grounding, _td_lambda


def _pend(n_p1, n_p2, v=0.5):
    return {
        "p1": [{"v_root": v} for _ in range(n_p1)],
        "p2": [{"v_root": v} for _ in range(n_p2)],
    }


def test_episode_emits_only_learner_rows():
    rows, glen, p1_won, draw = _episode(_pend(6, 6), True, False, 0.9)
    assert len(rows) == 6
    assert glen == 6
    assert (p1_won, draw) == (False, False)


def test_episode_carries_z_and_steps_to_end():
    """steps_to_end counts down to 0 at the terminal position, and z is the game outcome."""
    rows, _glen, _won, _draw = _episode(_pend(4, 4), False, True, 0.9)
    assert [n for _p, _t, _z, n in rows] == [3, 2, 1, 0]
    assert {z for _p, _t, z, _n in rows} == {1.0}


def test_episode_targets_match_td_lambda():
    """The stamped targets are exactly _td_lambda over the learner's root values."""
    pend = _pend(5, 5, v=0.25)
    rows, _glen, _won, _draw = _episode(pend, True, False, 0.7)
    expected = _td_lambda([p["v_root"] for p in pend["p1"]], -1.0, 0.7)
    assert [t for _p, t, _z, _n in rows] == pytest.approx(expected)


def test_episode_draw_when_both_die():
    rows, _glen, p1_won, draw = _episode(_pend(3, 3), True, True, 0.9)
    assert (p1_won, draw) == (False, True)
    assert {z for _p, _t, z, _n in rows} == {0.0}


def test_episode_none_when_empty():
    assert _episode(_pend(0, 0), True, False, 0.9) is None


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
