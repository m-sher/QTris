import numpy as np
import pytest

from qtris.training._1v1_placement_az import _episode, _finalize_episodes
from qtris.training.attack_risk import death_targets, n_step_attack, risk_calibration


def _pend(n1, n2):
    return {
        key: [{"reward": 0.006 * (i + 1), "bootstrap_value": 0.1 * i} for i in range(n)]
        for key, n in (("p1", n1), ("p2", n2))
    }


def test_discounted_setup_return_and_post_action_tail():
    assert n_step_attack([0, 0, 0.018], [10, 20, 30], 2, 0.5, 4) == pytest.approx(
        [7.5, 0.009 + 1, 0.018 + 2]
    )
    assert n_step_attack([0.006], [99], 14, 0.97, 0) == pytest.approx([0.006])
    assert n_step_attack([0.006], [99], 14, 0.97, 2) == pytest.approx([1.946])


def test_own_death_zero_bootstrap_winner_censored():
    rows, glen, won, draw = _episode(_pend(3, 3), True, False, 14, tails=(9, 2))
    assert glen == 3 and not won and not draw
    loser, winner = rows[2], rows[-1]
    assert loser[1] == pytest.approx(0.018)
    assert winner[1] == pytest.approx(0.018 + 0.97 * 2)
    assert loser[0]["hazard_target"][0] == 1
    assert winner[0]["hazard_target"].sum() == 0
    assert winner[0]["hazard_mask"].sum() == 1
    assert winner[0]["death_observed"].sum() == 1
    assert [r[2] for r in rows] == [1] * 3 + [0] * 3


def test_simultaneous_death_is_positive_for_both_players():
    rows, _, won, draw = _episode(_pend(2, 2), True, True, 14, tails=(4, 5))
    assert draw and not won
    for row in (rows[1], rows[-1]):
        assert row[1] == pytest.approx(0.012)
        assert row[0]["hazard_target"][0] == 1
        assert row[0]["death_target"].sum() == 24
    assert {r[3] for r in rows} == {0}


def test_timeout_retains_final_reward_and_bootstraps_both_post_states():
    rows, _, _, draw = _episode(_pend(2, 2), False, False, 14, tails=(4, 5))
    assert draw
    assert rows[1][1] == pytest.approx(0.012 + 0.97 * 4)
    assert rows[-1][1] == pytest.approx(0.012 + 0.97 * 5)
    assert _episode(_pend(0, 0), True, True, 14) is None


def test_hazard_death_at_horizon_and_censoring():
    targets, mask, cumulative, observed = death_targets(25, True)
    assert targets[0].sum() == 0 and mask[0].sum() == 24
    assert targets[1, 23] == 1 and cumulative[1, 23] == 1
    assert targets[-1, 0] == 1 and mask[-1].sum() == 1
    assert observed[-1].sum() == 24
    targets, mask, cumulative, observed = death_targets(3, False)
    assert targets.sum() == cumulative.sum() == 0
    assert mask.sum() == observed.sum() == 6


def test_calibration_excludes_unobserved_future():
    _, _, y, seen = death_targets(3, False)
    p = np.full_like(y, 0.2)
    result = risk_calibration(p, y, seen, horizons=(1, 24))
    assert result["brier_h1"] == pytest.approx(0.04)
    assert result["brier_h24"] is None
    assert result["censored_fraction_h24"] == 1


def test_both_players_bootstrap_from_same_current_learner():
    class Learner:
        def state_value(self, board, pieces, bcg):
            return bcg[:, :1]

    def observation(value):
        return (
            np.zeros((24, 10, 1), np.float32),
            np.zeros(7, np.int64),
            np.array([value, 0, 0], np.float32),
        )

    pend = _pend(3, 3)
    for player, offset in (("p1", 10), ("p2", 20)):
        for i, pos in enumerate(pend[player]):
            pos["board"], pos["pieces"], pos["bcg"] = observation(offset + i)
            pos["v_search"] = -999
    episodes = [(pend, False, False, (observation(13), observation(23)))]
    rows, *_ = _finalize_episodes(episodes, Learner(), 4, 1, 0.5)[0]
    assert rows[0][1] == pytest.approx(0.006 + 0.5 * 11)
    assert rows[3][1] == pytest.approx(0.006 + 0.5 * 21)
    assert rows[2][1] == pytest.approx(0.018 + 0.5 * 13)
    assert rows[5][1] == pytest.approx(0.018 + 0.5 * 23)
