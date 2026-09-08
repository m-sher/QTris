"""The GPU placement BFS against the C oracle's root placement set."""

import numpy as np
from teacher_common import load

from teacher.constants import (
    SPIN_ALL_MINI,
    SPIN_NONE,
    SPIN_T_FULL,
    SPIN_T_MINI,
)
from teacher.kernels.enumerate import debug_enumerate
from teacher.tables import PIECE_MIN_COL


def _expected(d, i):
    """C root placements of the active piece as (rot, norm_col, spin, landing_row)."""
    n = int(d["n_d1w32"][i])
    out = []
    for k in range(n):
        a = int(d["acts_d1w32"][i, k])
        if a < 160:
            out.append((a // 40, (a % 40) // 4, a % 4, int(d["rows_d1w32"][i, k])))
    return out


def _got(d, i):
    piece = int(d["active"][i])
    rot, col, row, spin = debug_enumerate(d["boards"][i], piece)
    return [
        (
            int(rot[k]),
            int(col[k]) + int(PIECE_MIN_COL[piece, int(rot[k])]),
            int(spin[k]),
            int(row[k]),
        )
        for k in range(len(rot))
    ]


def test_placement_set_and_emission_order_match_the_c():
    d = load()
    tested = 0
    for i in range(len(d["boards"])):
        if int(d["n_d1w32"][i]) == 0:
            continue
        assert _got(d, i) == _expected(d, i), f"position {i} {d['names'][i]}"
        tested += 1
    assert tested >= 200


def test_every_spin_type_is_exercised():
    d = load()
    seen = {SPIN_NONE: 0, SPIN_T_MINI: 0, SPIN_T_FULL: 0, SPIN_ALL_MINI: 0}
    for i in range(len(d["boards"])):
        if int(d["n_d1w32"][i]) == 0:
            continue
        for _rot, _col, spin, _row in _got(d, i):
            seen[spin] += 1
    assert all(count > 0 for count in seen.values()), seen


def test_landing_rows_are_at_or_below_the_spawn_row():
    d = load()
    for i in range(0, len(d["boards"]), 5):
        if int(d["n_d1w32"][i]) == 0:
            continue
        for _rot, _col, _spin, row in _got(d, i):
            assert 17 <= row < 40


def test_placements_are_unique():
    d = load()
    for i in range(0, len(d["boards"]), 5):
        got = _got(d, i)
        assert len(got) == len(set(got)), f"position {i} emitted a duplicate"


def test_empty_board_enumeration_is_stable_across_pieces():
    empty = np.zeros(40, np.uint16)
    for piece in range(1, 8):
        rot, col, row, spin = debug_enumerate(empty, piece)
        assert len(rot) > 0
        assert set(int(s) for s in spin) == {SPIN_NONE}
