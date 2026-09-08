"""The GPU lock, line clear and attack against the C oracle."""

import numpy as np
from teacher_common import load

from teacher.kernels.expand import debug_expand
from teacher.tables import PIECE_MIN_COL

STRIDE = 17


def _records():
    d = load()
    n = len(d["lock_pos"])
    idx = range(0, n, STRIDE)
    return d, idx


def test_lock_and_score_match_the_c_oracle():
    d, idx = _records()
    boards = d["boards"]
    checked = 0
    for j in idx:
        p = int(d["lock_pos"][j])
        piece = int(d["lock_piece"][j])
        rot = int(d["lock_rot"][j])
        raw_col = int(d["lock_col"][j]) - int(PIECE_MIN_COL[piece, rot])
        child = debug_expand(
            boards[p],
            piece,
            rot,
            raw_col,
            int(d["lock_row"][j]),
            int(d["lock_spin"][j]),
            int(d["b2b"][p]),
            int(d["combo"][p]),
            0,
            hold=int(d["hold"][p]),
        )
        assert int(child["rows_cleared"]) == int(d["lock_clears"][j])
        assert float(child["total_attack"]) == float(d["lock_attack"][j])
        assert int(child["b2b"]) == int(d["lock_new_b2b"][j])
        assert int(child["combo"]) == int(d["lock_new_combo"][j])
        got = np.asarray(child["board"], np.uint16) & np.uint16(0x3FF)
        assert np.array_equal(got, d["lock_board"][j] & np.uint16(0x3FF))
        checked += 1
    assert checked >= 500


def test_column_heights_match_a_full_scan():
    d, idx = _records()
    boards = d["boards"]
    for j in list(idx)[:200]:
        p = int(d["lock_pos"][j])
        piece = int(d["lock_piece"][j])
        rot = int(d["lock_rot"][j])
        child = debug_expand(
            boards[p],
            piece,
            rot,
            int(d["lock_col"][j]) - int(PIECE_MIN_COL[piece, rot]),
            int(d["lock_row"][j]),
            int(d["lock_spin"][j]),
            int(d["b2b"][p]),
            int(d["combo"][p]),
            0,
            hold=int(d["hold"][p]),
        )
        board = np.asarray(child["board"], np.uint16)
        for c in range(10):
            filled = np.flatnonzero(board & np.uint16(1 << c))
            expected = 40 - int(filled[0]) if filled.size else 0
            assert int(child["col_heights"][c]) == expected
