"""Issue #23: the env pathfinder must not emit a redundant trailing soft-drop.

The BFS records a placement only at a resting state, so its reconstructed path ends in a
soft-drop that settles the piece; the appended hard-drop then re-descends to the same row,
making that soft-drop redundant. `write_sequence` now strips it.

Two guarantees, both exercised against the REAL env over boards drawn from random play:
  1. no emitted sequence contains SOFT_DROP immediately before HARD_DROP (the fix), and
  2. re-inserting that trailing soft-drop changes nothing the env scores (so stripping it is
     behavior-preserving: same locked board, clears, attack, spin).
"""

import numpy as np

from TetrisEnv.Moves import Keys
from TetrisEnv.PyTetrisEnv import PyTetrisEnv


def _pathfinding_env(seed: int) -> PyTetrisEnv:
    return PyTetrisEnv(
        queue_size=5,
        max_holes=50,
        max_steps=None,
        max_len=15,
        pathfinding=True,
        seed=seed,
        idx=0,
    )


def _valid_rows(obs):
    """obs['sequences'] rows that encode a real placement (contain a HARD_DROP)."""
    return [row for row in obs["sequences"] if Keys.HARD_DROP in row.tolist()]


def _through_harddrop(row):
    """Keys from START up to and including the first HARD_DROP (drops the PAD tail)."""
    keys = [int(k) for k in row.tolist()]
    return keys[: keys.index(Keys.HARD_DROP) + 1]


def _replay(env, keys):
    """Replay a key list on the env's current state; restore scorer so probes don't drift."""
    b2b, combo = env._scorer._b2b, env._scorer._combo
    top_out, clear, attack, is_spin, next_board, _, _, _, _ = env._execute_action(
        env._board,
        env._vis_board,
        env._active_piece,
        env._hold_piece,
        env._queue,
        np.array(keys, dtype=np.int64),
    )
    env._scorer._b2b, env._scorer._combo = b2b, combo
    return (
        bool(top_out),
        int(clear),
        float(attack),
        bool(is_spin),
        next_board.tobytes(),
    )


def _boards_from_play(seed: int, steps: int):
    """Yield observations from a random self-play rollout (varied stacks, tucks, overhangs)."""
    env = _pathfinding_env(seed)
    rng = np.random.default_rng(seed)
    ts = env.reset()
    for _ in range(steps):
        obs = ts.observation
        rows = _valid_rows(obs)
        if not rows:
            break
        yield env, obs
        ts = env.step(rows[rng.integers(len(rows))])
        if ts.is_last():
            ts = env.reset()


def test_no_redundant_softdrop_before_harddrop():
    """No emitted placement sequence settles with SOFT_DROP -> HARD_DROP (regression guard)."""
    total = 0
    for _, obs in _boards_from_play(seed=7, steps=40):
        for row in _valid_rows(obs):
            keys = _through_harddrop(row)
            total += 1
            for a, b in zip(keys, keys[1:]):
                assert not (a == Keys.SOFT_DROP and b == Keys.HARD_DROP), (
                    f"redundant trailing soft-drop in {keys}"
                )
    assert total > 100  # the rollout actually exercised the pathfinder


def test_trailing_softdrop_is_behavior_preserving():
    """Re-inserting the stripped soft-drop yields an identical locked outcome for every
    enumerated placement, proving the strip never changes the game state."""
    probes = 0
    for env, obs in _boards_from_play(seed=11, steps=40):
        for row in _valid_rows(obs):
            keys = _through_harddrop(row)
            with_drop = keys[:-1] + [Keys.SOFT_DROP, Keys.HARD_DROP]
            assert _replay(env, keys) == _replay(env, with_drop), (
                f"outcome changed when re-inserting trailing soft-drop: {keys}"
            )
            probes += 1
    assert probes > 100
