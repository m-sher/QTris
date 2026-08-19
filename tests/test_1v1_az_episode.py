import numpy as np
import pytest

from qtris.training._1v1_placement_az import (
    ATK_HALF_SAT,
    B2B_HALF_SAT,
    BLEND_ATK_WEIGHT,
    BLEND_B2B_WEIGHT,
    _ahat,
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
    rows, glen, p1_won, draw = _episode(_pend(6, 6), True, False, 0.9, 0.0, 16)
    assert len(rows) == 12
    assert glen == 6
    assert (p1_won, draw) == (False, False)
    assert [r[2] for r in rows] == [1.0] * 6 + [0.0] * 6


def test_episode_z_is_per_player_and_opposite():
    rows, *_ = _episode(_pend(4, 4), False, True, 0.9, 0.0, 16)
    assert {r[3] for r in rows if r[2] == 1.0} == {1.0}
    assert {r[3] for r in rows if r[2] == 0.0} == {-1.0}


def test_episode_steps_to_end_counts_down_per_trajectory():
    rows, *_ = _episode(_pend(4, 3), False, True, 0.9, 0.0, 16)
    assert [r[4] for r in rows if r[2] == 1.0] == [3, 2, 1, 0]
    assert [r[4] for r in rows if r[2] == 0.0] == [2, 1, 0]


def test_episode_targets_match_td_lambda_at_blend_zero():
    """blend=0 reproduces the pure TD(lambda) targets exactly, for both players."""
    pend = _pend(5, 5, v=0.25, atk=2.0, b2b=7)
    rows, *_ = _episode(pend, True, False, 0.7, 0.0, 16)
    exp1 = _td_lambda([p["v_root"] for p in pend["p1"]], -1.0, 0.7)
    exp2 = _td_lambda([p["v_root"] for p in pend["p2"]], 1.0, 0.7)
    assert [r[1] for r in rows if r[2] == 1.0] == pytest.approx(exp1)
    assert [r[1] for r in rows if r[2] == 0.0] == pytest.approx(exp2)
    assert all(r[5] == 0.0 and r[6] == 0.0 for r in rows)


def test_episode_draw_when_both_die():
    rows, _glen, p1_won, draw = _episode(_pend(3, 3), True, True, 0.9, 0.0, 16)
    assert (p1_won, draw) == (False, True)
    assert {r[3] for r in rows} == {0.0}


def test_episode_none_when_empty():
    assert _episode(_pend(0, 0), True, False, 0.9, 0.0, 16) is None


def test_episode_uneven_trajectories():
    rows, glen, *_ = _episode(_pend(5, 0), True, False, 0.9, 0.0, 16)
    assert glen == 5
    assert len(rows) == 5


def test_blend_terminal_rows_stay_z():
    """Each trajectory's terminal row keeps that player's raw outcome at any blend."""
    pend = _pend(5, 5, v=0.2, atk=1.0, b2b=9)
    rows, *_ = _episode(pend, True, False, 0.9, 0.15, 16)
    p1 = [r for r in rows if r[2] == 1.0]
    p2 = [r for r in rows if r[2] == 0.0]
    assert p1[-1][1] == -1.0
    assert p2[-1][1] == 1.0
    pure = _td_lambda([p["v_root"] for p in pend["p1"]], -1.0, 0.9)
    assert p1[0][1] != pytest.approx(pure[0])


def test_blend_bounds_and_death_strictly_worst():
    """Blended rows land in [-(1-a), 1); the losing terminal alone sits at -1."""
    a = 0.15
    pend = _pend(8, 8, v=-0.9, atk=0.0, b2b=0)
    rows, *_ = _episode(pend, True, False, 0.9, a, 4)
    p1 = [r[1] for r in rows if r[2] == 1.0]
    assert p1[-1] == -1.0
    assert all(t >= -(1 - a) - 1e-9 for t in p1[:-1])
    assert all(-1.0 < t < 1.0 for t in p1[:-1])


def test_ahat_uncapped_monotonicity_and_holding_paid():
    """A held chain scores its ratio, and deeper always outscores shallower."""
    _, b2b_12, _ = _ahat([0.0] * 8, [12] * 8, 4)
    _, b2b_20, _ = _ahat([0.0] * 8, [20] * 8, 4)
    _, b2b_8, _ = _ahat([0.0] * 8, [8] * 8, 4)
    assert b2b_8[0] == pytest.approx(8 / (8 + B2B_HALF_SAT))
    assert (b2b_20 > b2b_12).all()
    assert (b2b_12 > b2b_8).all()


def test_ahat_attack_is_window_rate():
    atks = [20.0] + [0.0] * 15
    _a, _b, atk_ch = _ahat(atks, [0] * 16, 16)
    app = 20.0 / 16.0
    assert atk_ch[0] == pytest.approx(app / (app + ATK_HALF_SAT))


def test_ahat_window_truncates_at_episode_end():
    _, b2b_ch, atk_ch = _ahat([3.0, 0.0, 0.0], [4, 8, 12], 16)
    assert b2b_ch[-1] == pytest.approx(12 / (12 + B2B_HALF_SAT))
    assert atk_ch[-1] == pytest.approx(0.0)


def test_ahat_bounds():
    rng = np.random.default_rng(0)
    atks = rng.uniform(0, 25, 64)
    b2bs = rng.integers(-1, 25, 64)
    ahat, b2b_ch, atk_ch = _ahat(atks, b2bs, 16)
    for arr in (ahat, b2b_ch, atk_ch):
        assert (arr >= 0.0).all() and (arr < 1.0).all()


def test_ahat_favors_hoard_over_pure_attack():
    hoard, *_ = _ahat([0.0] * 16, [8] * 16, 16)
    attack, *_ = _ahat([1.0] * 16, [-1] * 16, 16)
    assert hoard[0] == pytest.approx(BLEND_B2B_WEIGHT * 0.5)
    assert attack[0] == pytest.approx(BLEND_ATK_WEIGHT * (1.0 / (1.0 + ATK_HALF_SAT)))
    assert hoard[0] > attack[0]


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
