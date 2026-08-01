"""Value-target composition in the 1v1 trainer's episode flush.

Targets must be the TD chain over each player's positions: realized reward plus the
discounted search return of the next position, death penalty on a dying player's final
position, own search return on a surviving player's final position.
"""

import pytest

from qtris.training._1v1_placement_az import VALUE_GAMMA, W_DEATH, _episode


def _p(value_search, r_real=None):
    p = {"value_search": value_search, "pi": None, "v_root": 0.0}
    if r_real is not None:
        p["r_real"] = r_real
    return p


def test_td_chain_and_death_penalty():
    pend = {
        "p1": [_p(0.5, r_real=0.10), _p(0.7, r_real=0.05), _p(0.4, r_real=0.0)],
        "p2": [_p(0.6, r_real=0.20), _p(0.3, r_real=0.0)],
    }
    rows, glen, p1_won, is_draw = _episode(pend, p1_died=True, p2_died=False)
    assert glen == 3 and not p1_won and not is_draw
    tgts = {(id(p)): t for p, t, _m in rows}

    # p1 (died): chain then death penalty on the final position
    assert tgts[id(pend["p1"][0])] == pytest.approx(0.10 + VALUE_GAMMA * 0.7)
    assert tgts[id(pend["p1"][1])] == pytest.approx(0.05 + VALUE_GAMMA * 0.4)
    assert tgts[id(pend["p1"][2])] == pytest.approx(0.0 - W_DEATH)

    # p2 (survived): chain then own search return on the final position
    assert tgts[id(pend["p2"][0])] == pytest.approx(0.20 + VALUE_GAMMA * 0.3)
    assert tgts[id(pend["p2"][1])] == pytest.approx(0.3)


def test_masks_and_outcome_bookkeeping():
    pend = {"p1": [_p(0.2, r_real=0.0)], "p2": [_p(0.1, r_real=0.0)]}
    rows, _glen, p1_won, is_draw = _episode(pend, p1_died=False, p2_died=True)
    assert p1_won and not is_draw
    masks = [m for _p_, _t, m in rows]
    assert masks == [1.0, 0.0]

    rows, _glen, p1_won, is_draw = _episode(pend, p1_died=True, p2_died=True)
    assert not p1_won and is_draw
