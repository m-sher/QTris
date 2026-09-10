import numpy as np
import pytest

from qtris.training._1v1_placement_az import (
    _episode,
    _grounding,
    _n_step,
)


def _pend(n1, n2, v=0.5, attack=0.0):
    return {
        "p1": [
            {"v_search": v + 0.01 * i, "v_root": -v, "attack": attack}
            for i in range(n1)
        ],
        "p2": [
            {"v_search": v + 0.01 * i, "v_root": -v, "attack": attack}
            for i in range(n2)
        ],
    }


def _zeros(n):
    return [0.0] * n


def test_episode_emits_both_players_rows():
    rows, glen, p1_won, draw = _episode(_pend(6, 6), True, False, 3)
    assert len(rows) == 12
    assert glen == 6
    assert (p1_won, draw) == (False, False)
    assert [r[2] for r in rows] == [1.0] * 6 + [0.0] * 6


def test_episode_z_is_per_player_and_opposite():
    rows, *_ = _episode(_pend(4, 4), False, True, 3)
    assert {r[3] for r in rows if r[2] == 1.0} == {1.0}
    assert {r[3] for r in rows if r[2] == 0.0} == {-1.0}


def test_episode_steps_to_end_counts_down_per_trajectory():
    rows, *_ = _episode(_pend(4, 3), False, True, 3)
    assert [r[4] for r in rows if r[2] == 1.0] == [3, 2, 1, 0]
    assert [r[4] for r in rows if r[2] == 0.0] == [2, 1, 0]


def test_episode_targets_match_n_step():
    """The targets are the pure n-step targets, for both players."""
    pend = _pend(5, 5, v=0.25)
    rows, *_ = _episode(pend, True, False, 2)
    exp1 = _n_step([p["v_search"] for p in pend["p1"]], _zeros(5), -1.0, 2, 1.0, False)
    exp2 = _n_step([p["v_search"] for p in pend["p2"]], _zeros(5), 1.0, 2, 1.0, False)
    assert [r[1] for r in rows if r[2] == 1.0] == pytest.approx(exp1)
    assert [r[1] for r in rows if r[2] == 0.0] == pytest.approx(exp2)


def test_n_step_bootstraps_n_ahead_and_grounds_the_tail():
    """A row bootstraps on the search value exactly n positions later; every row within
    n of the end gets raw z, the terminal row included, unless the game was truncated,
    when those rows take the final position's value instead."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    z6 = _zeros(6)
    targets = _n_step(values, z6, -1.0, 2, 1.0, False)
    assert targets == pytest.approx([0.3, 0.4, 0.5, 0.6, -1.0, -1.0])
    assert _n_step(values, z6, -1.0, 10, 1.0, False) == pytest.approx([-1.0] * 6)
    assert _n_step([0.7], [0.0], 1.0, 1, 1.0, False) == pytest.approx([1.0])
    assert _n_step(values, z6, 0.0, 2, 1.0, True) == pytest.approx(
        [0.3, 0.4, 0.5, 0.6, 0.6, 0.6]
    )
    assert _n_step(values, z6, 0.0, 10, 1.0, True) == pytest.approx([0.6] * 6)


def test_n_step_sums_discounted_rewards_before_the_bootstrap():
    """G_t = sum_{i<n} gamma^i r_{t+i} + gamma^n v_{t+n}; a row within n of the end sums
    its rewards to the last step, where the outcome z joins the last reward, and a
    truncated tail discounts the final value by the steps to it."""
    values = [0.1, 0.2, 0.3, 0.4]
    rewards = [1.0, 2.0, 3.0, 4.0]
    g = 0.5
    got = _n_step(values, rewards, -1.0, 2, g, False)
    assert got == pytest.approx(
        [
            1.0 + g * 2.0 + g * g * 0.3,
            2.0 + g * 3.0 + g * g * 0.4,
            3.0 + g * (4.0 - 1.0),
            4.0 - 1.0,
        ]
    )
    got = _n_step(values, rewards, 0.0, 2, g, True)
    assert got == pytest.approx(
        [
            1.0 + g * 2.0 + g * g * 0.3,
            2.0 + g * 3.0 + g * g * 0.4,
            3.0 + g * 0.4,
            0.4,
        ]
    )
    # A whole-game horizon at gamma 1 is the Monte Carlo return.
    assert _n_step(values, rewards, 1.0, 10, 1.0, False) == pytest.approx(
        [11.0, 10.0, 8.0, 5.0]
    )


def test_episode_scales_attack_by_w_value_attack_and_can_bootstrap_on_the_root():
    pend = _pend(4, 4, v=0.5, attack=2.0)
    rows, *_ = _episode(pend, True, False, 2, gamma=1.0, w_value_attack=0.25)
    p1 = [r[1] for r in rows if r[2] == 1.0]
    # 0.25 * 2 lines = 0.5 per step: two steps then the search value, or to the end + z.
    assert p1 == pytest.approx([1.0 + 0.52, 1.0 + 0.53, 1.0 - 1.0, 0.5 - 1.0])
    rows, *_ = _episode(pend, True, False, 2, bootstrap="v_root")
    p1 = [r[1] for r in rows if r[2] == 1.0]
    assert p1 == pytest.approx([-0.5, -0.5, -1.0, -1.0])


def test_capped_game_is_a_draw_for_rating_and_a_truncation_for_the_target():
    """A game the move cap ends with neither player dead scores as a draw (z=0, is_draw)
    while its tail rows bootstrap on the final position's search value."""
    pend = _pend(4, 4)
    rows, _glen, p1_won, draw = _episode(pend, False, False, 2)
    assert (p1_won, draw) == (False, True)
    assert {r[3] for r in rows} == {0.0}
    learner = [r for r in rows if r[2] == 1.0]
    last = pend["p1"][3]["v_search"]
    assert [r[1] for r in learner[-2:]] == pytest.approx([last, last])
    assert learner[0][1] == pytest.approx(pend["p1"][2]["v_search"])


def test_episode_draw_when_both_die():
    """A double-KO is a real draw: z=0, and the tail targets are that outcome."""
    rows, _glen, p1_won, draw = _episode(_pend(3, 3), True, True, 3)
    assert (p1_won, draw) == (False, True)
    assert {r[3] for r in rows} == {0.0}
    assert [r[1] for r in rows] == pytest.approx([0.0] * 6)


def test_episode_none_when_empty():
    assert _episode(_pend(0, 0), True, False, 3) is None


def test_episode_uneven_trajectories():
    rows, glen, *_ = _episode(_pend(5, 0), True, False, 3)
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
