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


def _pend(n, v=0.5, atk=0.0, b2b=0):
    return {"p1": [{"v_root": v, "atk": atk, "b2b": b2b} for _ in range(n)]}


def test_episode_emits_only_learner_rows():
    rows, glen, p1_won, draw = _episode(_pend(6), True, False, 0.9, 0.0, 16)
    assert len(rows) == 6
    assert glen == 6
    assert (p1_won, draw) == (False, False)


def test_episode_carries_z_and_steps_to_end():
    """steps_to_end counts down to 0 at the terminal position, and z is the game outcome."""
    rows, _glen, _won, _draw = _episode(_pend(4), False, True, 0.9, 0.0, 16)
    assert [r[3] for r in rows] == [3, 2, 1, 0]
    assert {r[2] for r in rows} == {1.0}


def test_episode_targets_match_td_lambda_at_blend_zero():
    """blend=0 reproduces the pure TD(lambda) targets exactly."""
    pend = _pend(5, v=0.25, atk=2.0, b2b=7)
    rows, _glen, _won, _draw = _episode(pend, True, False, 0.7, 0.0, 16)
    expected = _td_lambda([p["v_root"] for p in pend["p1"]], -1.0, 0.7)
    assert [r[1] for r in rows] == pytest.approx(expected)
    assert all(r[4] == 0.0 and r[5] == 0.0 for r in rows)


def test_episode_draw_when_both_die():
    rows, _glen, p1_won, draw = _episode(_pend(3), True, True, 0.9, 0.0, 16)
    assert (p1_won, draw) == (False, True)
    assert {r[2] for r in rows} == {0.0}


def test_episode_none_when_empty():
    assert _episode(_pend(0), True, False, 0.9, 0.0, 16) is None


def test_blend_terminal_rows_stay_z():
    """Terminal rows keep the raw outcome at any blend; earlier rows move."""
    for died, z in ((True, -1.0), (False, 1.0)):
        pend = _pend(5, v=0.2, atk=1.0, b2b=9)
        rows, *_ = _episode(pend, died, not died, 0.9, 0.15, 16)
        pure = _td_lambda([p["v_root"] for p in pend["p1"]], z, 0.9)
        assert rows[-1][1] == z
        assert rows[0][1] != pytest.approx(pure[0])


def test_blend_bounds_and_death_strictly_worst():
    """Every surviving row lands in [-(1-a), 1); the death terminal alone sits at -1."""
    a = 0.15
    pend = {
        "p1": [
            {"v_root": v, "atk": atk, "b2b": b2b}
            for v, atk, b2b in zip(
                np.linspace(-0.99, 0.99, 12),
                [0, 4, 0, 0, 9, 0, 0, 0, 2, 0, 0, 1],
                [0, 1, 2, 3, 4, 5, 6, -1, 0, 1, 2, 3],
            )
        ]
    }
    rows, *_ = _episode(pend, True, False, 0.9, a, 4)
    targets = [r[1] for r in rows]
    assert targets[-1] == -1.0
    assert all(t >= -(1 - a) - 1e-9 for t in targets[:-1])
    assert all(t > -1.0 for t in targets[:-1])
    assert all(t < 1.0 for t in targets[:-1])


def test_ahat_uncapped_monotonicity_and_holding_paid():
    """A held chain scores its ratio, and deeper always outscores shallower."""
    _, b2b_12, _ = _ahat([0.0] * 8, [12] * 8, 4)
    _, b2b_20, _ = _ahat([0.0] * 8, [20] * 8, 4)
    _, b2b_8, _ = _ahat([0.0] * 8, [8] * 8, 4)
    assert b2b_8[0] == pytest.approx(8 / (8 + B2B_HALF_SAT))
    assert (b2b_20 > b2b_12).all()
    assert (b2b_12 > b2b_8).all()


def test_ahat_attack_is_window_rate():
    """A single surge scores by the window's attack rate."""
    atks = [20.0] + [0.0] * 15
    ahat, _b, atk_ch = _ahat(atks, [0] * 16, 16)
    app = 20.0 / 16.0
    assert atk_ch[0] == pytest.approx(app / (app + ATK_HALF_SAT))


def test_ahat_window_truncates_at_episode_end():
    """The last position's window is its own step alone."""
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
    """A held b2b-8 hoard with zero attack outscores flat 1-attack-per-placement play."""
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
