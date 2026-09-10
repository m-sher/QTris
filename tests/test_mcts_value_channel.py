"""The shaping-free value channel carries w_value_attack per raw attack line."""

import sys

import numpy as np

sys.path.insert(0, "tests")
from test_mcts_break_credit import _candidates, _legal_desc, _played_env

from qtris.search.cmcts import CANDIDATE_CAPACITY, CMCTS

SIMS = 48


def _root_value(env, w_value_attack, focus, sims):
    """Shaping-free root value after `sims` single-leaf rounds with a zero net, the root
    prior concentrated on `focus`, and every hand potential off."""
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
        w_attack=0.006,
        w_death=1.0,
        return_scale=1.0,
        max_len=15,
        num_simulations=sims,
        leaves_per_round=1,
        vloss=1.0,
        w_b2b=0.0,
        q_norm=True,
        w_height=0.0,
        w_bumpiness=0.0,
        fpu=0.4,
        w_holes=0.0,
        w_plain=0.0,
        w_oracle=0.0,
        w_value_attack=w_value_attack,
    )
    try:
        engine.set_root(0, env)
        nv, _req = engine.collect_roots()
        assert nv == 1
        logits = np.full(nv * CANDIDATE_CAPACITY, -30.0, np.float32)
        for slot in focus:
            logits[slot] = 0.0
        engine.apply_roots(logits, np.zeros(nv, np.float32), np.zeros_like(logits), 0.0)
        for _ in range(sims):
            nv2, _ = engine.collect_leaves()
            if nv2 == 0:
                break
            engine.apply_leaves(
                np.zeros(nv2 * CANDIDATE_CAPACITY, np.float32),
                np.zeros(nv2, np.float32),
            )
        _pi, counts, _desc, dead, rv = engine.result()
        assert not dead[0]
        return float(rv[0]), np.array(counts[0], np.float64)
    finally:
        engine.destroy()


def _attacking_position():
    for seed in (9, 20, 30, 7, 11, 13):
        env = _played_env(seed)
        desc, slots = _legal_desc(env)
        cands = _candidates(env, desc, slots)
        attacking = [s for s, (_c, a, _b) in cands.items() if a > 0]
        if attacking:
            return env, attacking, cands
    raise AssertionError("no seed offered an attacking placement")


def test_one_ply_of_the_channel_is_exactly_the_attack_at_w_value_attack():
    """One simulation into one attacking child, zero net, no potentials: the root value is
    that child's raw attack times the weight, and nothing at weight zero."""
    env, attacking, cands = _attacking_position()
    slot = max(attacking, key=lambda s: cands[s][1])
    off, _ = _root_value(env, 0.0, [slot], sims=1)
    on, _ = _root_value(env, 1.0, [slot], sims=1)
    half, _ = _root_value(env, 0.5, [slot], sims=1)
    assert off == 0.0
    assert on == cands[slot][1]
    assert half == 0.5 * cands[slot][1]


def test_the_channel_accumulates_along_the_path_and_leaves_selection_alone():
    env, attacking, _cands = _attacking_position()
    off, counts_off = _root_value(env, 0.0, attacking[:2], sims=SIMS)
    on, counts_on = _root_value(env, 1.0, attacking[:2], sims=SIMS)
    assert np.array_equal(counts_off, counts_on)
    assert on > off
