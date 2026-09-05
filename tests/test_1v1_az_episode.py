import numpy as np
import pytest

from qtris.training._1v1_placement_az import (
    _attack_window,
    _episode,
    _grounding,
    _n_step,
)

NORM = 28.0


def _pend(n1, n2, v=0.5):
    return {
        "p1": [{"v_search": v + 0.01 * i, "credit": float(i % 3)} for i in range(n1)],
        "p2": [{"v_search": v + 0.01 * i, "credit": float(i % 3)} for i in range(n2)],
    }


def test_episode_emits_both_players_rows():
    rows, glen, p1_won, draw = _episode(_pend(6, 6), True, False, 3, NORM)
    assert len(rows) == 12
    assert glen == 6
    assert (p1_won, draw) == (False, False)
    assert [r[2] for r in rows] == [1.0] * 6 + [0.0] * 6


def test_episode_z_is_per_player_and_opposite():
    rows, *_ = _episode(_pend(4, 4), False, True, 3, NORM)
    assert {r[3] for r in rows if r[2] == 1.0} == {1.0}
    assert {r[3] for r in rows if r[2] == 0.0} == {-1.0}


def test_episode_steps_to_end_counts_down_per_trajectory():
    rows, *_ = _episode(_pend(4, 3), False, True, 3, NORM)
    assert [r[4] for r in rows if r[2] == 1.0] == [3, 2, 1, 0]
    assert [r[4] for r in rows if r[2] == 0.0] == [2, 1, 0]


def test_episode_targets_match_n_step():
    """The targets are the pure n-step targets, for both players."""
    pend = _pend(5, 5, v=0.25)
    rows, *_ = _episode(pend, True, False, 2, NORM)
    exp1 = _n_step([p["v_search"] for p in pend["p1"]], -1.0, 2, False)
    exp2 = _n_step([p["v_search"] for p in pend["p2"]], 1.0, 2, False)
    assert [r[1] for r in rows if r[2] == 1.0] == pytest.approx(exp1)
    assert [r[1] for r in rows if r[2] == 0.0] == pytest.approx(exp2)


def test_n_step_bootstraps_n_ahead_and_grounds_the_tail():
    """A row bootstraps on the search value exactly n positions later; every row within
    n of the end gets raw z, the terminal row included, unless the game was truncated,
    when those rows take the final position's value instead."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    targets = _n_step(values, -1.0, 2, False)
    assert targets == pytest.approx([0.3, 0.4, 0.5, 0.6, -1.0, -1.0])
    assert _n_step(values, -1.0, 10, False) == pytest.approx([-1.0] * 6)
    assert _n_step([0.7], 1.0, 1, False) == pytest.approx([1.0])
    assert _n_step(values, 0.0, 2, True) == pytest.approx(
        [0.3, 0.4, 0.5, 0.6, 0.6, 0.6]
    )
    assert _n_step(values, 0.0, 10, True) == pytest.approx([0.6] * 6)


def test_capped_game_is_a_draw_for_rating_and_a_truncation_for_the_target():
    """A game the move cap ends with neither player dead scores as a draw (z=0, is_draw)
    while its tail rows bootstrap on the final position's search value."""
    pend = _pend(4, 4)
    rows, _glen, p1_won, draw = _episode(pend, False, False, 2, NORM)
    assert (p1_won, draw) == (False, True)
    assert {r[3] for r in rows} == {0.0}
    learner = [r for r in rows if r[2] == 1.0]
    last = pend["p1"][3]["v_search"]
    assert [r[1] for r in learner[-2:]] == pytest.approx([last, last])
    assert learner[0][1] == pytest.approx(pend["p1"][2]["v_search"])


def test_episode_draw_when_both_die():
    """A double-KO is a real draw: z=0, and the tail targets are that outcome."""
    rows, _glen, p1_won, draw = _episode(_pend(3, 3), True, True, 3, NORM)
    assert (p1_won, draw) == (False, True)
    assert {r[3] for r in rows} == {0.0}
    assert [r[1] for r in rows] == pytest.approx([0.0] * 6)


def test_episode_none_when_empty():
    assert _episode(_pend(0, 0), True, False, 3, NORM) is None


def test_episode_uneven_trajectories():
    rows, glen, *_ = _episode(_pend(5, 0), True, False, 3, NORM)
    assert glen == 5
    assert len(rows) == 5


def test_attack_window_targets_and_mask():
    """The target is the next n credits over norm, capped at 1. A death-ended trajectory
    has every window complete; a truncated one masks the windows that run past the end."""
    credits = [1.0, 0.0, 3.0, 2.0, 0.0]
    targets, masks = _attack_window(credits, 2, 4.0, False)
    assert targets == pytest.approx([0.25, 0.75, 1.0, 0.5, 0.0])
    assert masks == [1.0] * 5
    targets_t, masks_t = _attack_window(credits, 2, 4.0, True)
    assert targets_t == pytest.approx(targets)
    assert masks_t == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_episode_attack_rows_follow_the_window():
    pend = _pend(5, 5)
    rows, *_ = _episode(pend, True, False, 2, 4.0)
    exp_t, exp_m = _attack_window([p["credit"] for p in pend["p1"]], 2, 4.0, False)
    learner = [r for r in rows if r[2] == 1.0]
    assert [r[5] for r in learner] == pytest.approx(exp_t)
    assert [r[6] for r in learner] == exp_m
    assert all(0.0 <= r[5] <= 1.0 for r in rows)


def test_episode_truncation_masks_the_attack_tail_only():
    rows, *_ = _episode(_pend(6, 6), False, False, 3, NORM)
    learner = [r for r in rows if r[2] == 1.0]
    assert [r[6] for r in learner] == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


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
