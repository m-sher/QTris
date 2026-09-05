"""A chain-breaking clear earns no attack credit at any pending-garbage depth."""

import numpy as np
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Pieces import PieceType
from qtris.search.cmcts import CMCTS, CANDIDATE_CAPACITY

from test_mcts_rounds import _played_env

BANK = 6
SIMS = 16


def _candidates(env, desc, slots):
    """(clears, attack, new_b2b) per candidate, locked through the same C core the search
    steps with."""
    searcher = CB2BSearch()
    out = {}
    for slot in slots:
        is_hold, rot, norm_col, landing_row, spin = (int(x) for x in desc[slot])
        if is_hold:
            placed = (
                env._queue[0] if env._hold_piece == PieceType.N else env._hold_piece
            )
        else:
            placed = env._active_piece.piece_type
        _board, clears, attack, new_b2b, _combo = searcher.lock_score(
            env._board, placed.value, rot, norm_col, landing_row, spin, BANK, -1
        )
        out[slot] = (int(clears), float(attack), int(new_b2b))
    return out


def _search(env, pending, focus, w_attack=1.0, w_plain=0.0):
    """Priors concentrated on `focus`, zero leaf values, raw Q: the more-visited of the
    two focused children is the one with the larger edge reward. Returns the root visit
    counts and descriptors."""
    env._scorer._b2b = BANK
    env._scorer._combo = -1
    env._garbage_queue = [(pending, 3, 1)] if pending else []
    engine = CMCTS(
        1,
        board_height=40,
        queue_size=5,
        max_holes=50,
        garbage_push_delay=1,
        auto_push_garbage=0,
        auto_fill_queue=1,
        c_puct=1.5,
        gamma=1.0,
        w_attack=w_attack,
        w_death=1.0,
        return_scale=1.0,
        max_len=15,
        num_simulations=SIMS,
        leaves_per_round=1,
        vloss=1.0,
        w_b2b=0.0,
        q_norm=False,
        w_plain=w_plain,
    )
    try:
        engine.set_root(0, env)
        nv, req = engine.collect_roots()
        assert nv == 1
        logits = np.full(nv * CANDIDATE_CAPACITY, -30.0, np.float32)
        for slot in focus:
            logits[slot] = 0.0
        engine.apply_roots(logits, np.zeros(nv, np.float32), np.zeros_like(logits), 0.0)
        for _ in range(SIMS):
            nv2, _ = engine.collect_leaves()
            if nv2 == 0:
                break
            engine.apply_leaves(
                np.zeros(nv2 * CANDIDATE_CAPACITY, np.float32),
                np.zeros(nv2, np.float32),
                np.zeros(nv2, np.float32),
            )
        _pi, counts, desc, dead, _rv = engine.result()
        assert not dead[0]
        return np.array(counts[0], np.float64), np.array(desc[0], np.int64)
    finally:
        engine.destroy()


def _legal_desc(env):
    engine = CMCTS(1, board_height=40, queue_size=5, max_holes=50, max_len=15)
    try:
        env._scorer._b2b = BANK
        engine.set_root(0, env)
        nv, req = engine.collect_roots()
        assert nv == 1
        _pi, _counts, desc, _dead, _rv = engine.result()
        mask = np.array(req[4][0], dtype=bool)
        return np.array(desc[0], np.int64), [
            s for s in range(CANDIDATE_CAPACITY) if mask[s]
        ]
    finally:
        engine.destroy()


def _position_with_break_and_maintain():
    for seed in (7, 11, 13, 17, 19, 23):
        env = _played_env(seed)
        desc, slots = _legal_desc(env)
        cands = _candidates(env, desc, slots)
        breaks = [s for s, (c, a, b) in cands.items() if c > 0 and b == -1 and a > 0]
        keeps = [s for s, (c, a, b) in cands.items() if c > 0 and b == BANK + 1]
        if breaks and keeps:
            return env, breaks[0], keeps[0], cands
    raise AssertionError(
        "no seed offered both a chain-breaking and a maintaining clear"
    )


def test_break_earns_no_credit_at_any_queue_depth():
    env, brk, keep, cands = _position_with_break_and_maintain()
    assert cands[brk][1] >= BANK + 1  # surge released: base + bonus + bank
    pending = int(cands[keep][1]) + 3
    for queued in (0, pending):
        counts, _ = _search(env, queued, (brk, keep))
        assert counts[keep] > counts[brk], (queued, counts[keep], counts[brk])


def test_plain_cost_applies_only_with_nothing_queued():
    env, brk, keep, cands = _position_with_break_and_maintain()
    pending = int(cands[keep][1]) + 3
    # w_attack 0 isolates the plain cost: the break is charged at queue depth 0 and free
    # with garbage queued, so it takes more of the fixed visit budget when queued.
    empty, _ = _search(env, 0, (brk, keep), w_attack=0.0, w_plain=1.0)
    queued, _ = _search(env, pending, (brk, keep), w_attack=0.0, w_plain=1.0)
    assert empty[keep] > empty[brk], (empty[keep], empty[brk])
    assert queued[brk] > empty[brk], (queued[brk], empty[brk])
