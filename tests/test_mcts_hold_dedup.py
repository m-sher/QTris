"""A hold piece of the active piece's type adds no candidates; any other hold piece does."""

import numpy as np
from TetrisEnv.Pieces import PieceType
from qtris.search.cmcts import CMCTS, CANDIDATE_CAPACITY

from test_mcts_rounds import _played_env

HOLD_BASE = CANDIDATE_CAPACITY // 2


def _legal_mask(env):
    engine = CMCTS(1, board_height=40, queue_size=5, max_holes=50, max_len=15)
    try:
        engine.set_root(0, env)
        nv, req = engine.collect_roots()
        assert nv == 1
        return np.array(req[4][0], dtype=bool)
    finally:
        engine.destroy()


def test_same_type_hold_enumerates_only_the_no_hold_branch():
    for seed in (7, 11):
        env = _played_env(seed)
        active = env._active_piece.piece_type
        env._hold_piece = active
        same = _legal_mask(env)
        env._hold_piece = PieceType.I if active != PieceType.I else PieceType.O
        other = _legal_mask(env)
        assert same[:HOLD_BASE].sum() == other[:HOLD_BASE].sum() > 0
        assert same[HOLD_BASE:].sum() == 0
        assert other[HOLD_BASE:].sum() > 0
