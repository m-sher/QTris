"""The GPU leaf evaluation against the C oracle's per-root scores."""

import numpy as np
from teacher_common import load, no_hold_roots

from teacher.constants import DEATH_SCORE, ROOT_FLOOR
from teacher.kernels.evaluate import debug_evaluate
from teacher.kernels.expand import debug_expand
from teacher.tables import PIECE_MIN_COL

TOL = 1e-2
STRIDE = 11


def _score(d, i, k):
    """GPU score of the child reached by root k of position i."""
    a = int(d["acts_d1w32"][i, k])
    rot, ncol, spin = (a % 160) // 40, (a % 40) // 4, a % 4
    piece = int(d["active"][i])
    child = debug_expand(
        d["boards"][i],
        piece,
        rot,
        ncol - int(PIECE_MIN_COL[piece, rot]),
        int(d["rows_d1w32"][i, k]),
        spin,
        int(d["b2b"][i]),
        int(d["combo"][i]),
        int(d["garbage"][i]),
        hold=int(d["hold"][i]),
    )
    return child, debug_evaluate(
        child["board"],
        d["queues"][i],
        hold_piece=int(child["hold_piece"]),
        b2b=int(child["b2b"]),
        combo=int(child["combo"]),
        total_attack=float(child["total_attack"]),
        pieces_placed=int(child["pieces_placed"]),
        chain_ramp=int(child["chain_ramp"]),
        garbage_remaining=int(child["garbage_remaining"]),
        garbage_prevented=float(child["garbage_prevented"]),
        unlicensed_cash=int(child["unlicensed_cash"]),
        parent_avg_height=float(child["parent_avg_height"]),
        unlicensed_cash_A=float(child["unlicensed_cash_A"]),
        next_queue_idx=int(child["next_queue_idx"]),
    )


def _best_no_hold_roots(d):
    """(position, root) of every position whose chosen root places the active piece."""
    out = []
    for i in range(len(d["n_d1w32"])):
        best = int(d["best_d1w32"][i])
        if best < 0 or best >= 160:
            continue
        hit = np.flatnonzero(d["acts_d1w32"][i, : int(d["n_d1w32"][i])] == best)
        # One action index can cover several landing rows, and only an unambiguous
        # match identifies the root the C actually played.
        if hit.size == 1:
            out.append((i, int(hit[0])))
    return out


def test_the_chosen_root_scores_what_the_c_oracle_valued_it():
    """The C reports the played line's score in attack lines, so it gates the leaf eval.

    Root scores themselves are on the per-depth normalised scale and carry no unit.
    """
    d = load()
    worst = 0.0
    checked = 0
    for i, k in _best_no_hold_roots(d):
        _child, got = _score(d, i, k)
        worst = max(worst, abs(got - float(d["value_d1w32"][i])))
        checked += 1
    assert checked >= 100
    assert worst <= TOL, f"worst score error {worst}"


def test_dead_roots_score_exactly_the_death_value():
    d = load()
    dead = 0
    for i, k in no_hold_roots(d)[::STRIDE]:
        child, got = _score(d, i, k)
        # A root every child of which died is never raised and reports the floor.
        if float(d["scores_d1w32"][i, k]) == ROOT_FLOOR:
            assert got == DEATH_SCORE
            dead += 1
    assert dead > 0, "no dead root in the sampled set"


def test_evaluation_is_deterministic():
    d = load()
    for i, k in no_hold_roots(d)[::157]:
        first = _score(d, i, k)[1]
        assert _score(d, i, k)[1] == first


def test_an_empty_board_beats_a_near_death_board():
    d = load()
    empty = np.zeros(40, np.uint16)
    tall = np.zeros(40, np.uint16)
    tall[19:] = np.uint16(0x1FF)
    q = d["queues"][0]
    assert debug_evaluate(empty, q) > debug_evaluate(tall, q)
