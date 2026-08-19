"""Descriptor path vs key-sequence path parity gate.

`placement_step` commits a move by descriptor, locking and scoring through the C core and
feeding the enumerator's own spin label straight into `lock_score` without executing keys.
`_step` executes the keys and lets the env's `Scorer` classify the spin itself. A label the
key path cannot reproduce would mint attack the game cannot produce.

Both sides are the real implementations on cloned envs.
"""

import os

import numpy as np
import pytest

from TetrisEnv.Pieces import PieceType
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch
from qtris.search.placement_search import (
    clone_sim_env,
    descriptor_key_sequence,
    placement_step,
)

MAX_LEN = 15
# Boards yielding at least one spin-labelled placement, one per (piece, spin) pair;
# the other fixture boards produce none, so these carry the spin path.
SPIN_BOARDS = os.path.join(os.path.dirname(__file__), "fixtures", "spin_boards.npz")


def _env(seed=0):
    return PyTetrisEnv(
        queue_size=5,
        max_holes=50,
        max_steps=None,
        max_len=MAX_LEN,
        pathfinding=True,
        garbage_chance=0.0,
        auto_push_garbage=False,
        auto_fill_queue=True,
        seed=seed,
        idx=0,
        num_row_tiers=2,
    )


def _overhang_board(env):
    """Multi-landing board reused from test_multi_landing_mcts: overhangs are where the
    descriptor and key paths are most likely to disagree."""
    b = env._board.copy()
    b[-2:, :] = 1.0
    b[-12:-10, 0:6] = 1.0
    b[-20:-18, 4:10] = 1.0
    return b


def _stack_board(env, heights):
    b = env._board.copy()
    for col, h in enumerate(heights):
        if h > 0:
            b[-h:, col] = 1.0
    return b


def _boards(env):
    """Fixture boards: empty, overhang, a ragged stack, a near-flat well, and the
    spin-producing boards. The first four produce zero spin-labelled placements between
    them, so the spin boards carry the whole spin path."""
    out = [
        ("empty", env._board.copy()),
        ("overhang", _overhang_board(env)),
        ("ragged", _stack_board(env, [4, 7, 2, 9, 5, 3, 8, 1, 6, 0])),
        ("well", _stack_board(env, [6, 6, 6, 6, 6, 6, 6, 6, 6, 0])),
    ]
    with np.load(SPIN_BOARDS) as d:
        out += [(f"spin_{k}", d[k].astype(np.float32)) for k in sorted(d.files)]
    return out


def _spin_fixture_pieces():
    """Piece names for which a spin-producing board exists, read from the fixture keys."""
    with np.load(SPIN_BOARDS) as d:
        return {k.split("_")[0] for k in d.files}


def _descriptors(env, piece):
    """Every enumerated placement for `piece` on `env._board`, as descriptors."""
    finder = env._key_sequence_finder
    rots, ncols, lrs, spins = finder.find_unique_placements(
        env._board, piece, MAX_LEN, False, 512
    )
    return [
        (0, int(rots[i]), int(ncols[i]), int(lrs[i]), int(spins[i]))
        for i in range(len(rots))
    ]


def _has_key_sequence(env, desc):
    """True when the key-sequence finder can actually reproduce this descriptor. The
    fallback in `descriptor_key_sequence` is a bare START+HARD_DROP, which is a different
    placement, so comparing against it would be meaningless."""
    _is_hold, rot, ncol, lr, spin = desc
    rots, ncols, lrs, spins, _seqs = env._key_sequence_finder.find_unique_placements(
        board=env._board,
        piece=env._active_piece,
        max_len=MAX_LEN,
        is_hold=False,
        max_out=512,
        with_sequences=True,
    )
    return any(
        int(rots[i]) == rot
        and int(ncols[i]) == ncol
        and int(lrs[i]) == lr
        and int(spins[i]) == spin
        for i in range(len(rots))
    )


def _state(env):
    return (
        env._board.copy(),
        int(env._scorer._b2b),
        int(env._scorer._combo),
    )


@pytest.mark.parametrize("piece_type", list(PieceType)[1:8])  # I J L O S T Z
def test_descriptor_step_matches_key_sequence(piece_type):
    """placement_step(desc) == _step(descriptor_key_sequence(desc)) on board, b2b, and
    combo -- across the fixture boards and every enumerated placement."""
    searcher = CB2BSearch()
    base = _env()
    base._reset()

    checked = 0
    spins_checked = 0
    unreachable = []
    mismatches = []

    for name, board in _boards(base):
        for b2b, combo in ((-1, -1), (3, 2), (6, 0)):
            env = _env()
            env._reset()
            env._board = board.copy()
            env._active_piece = env._spawn_piece(piece_type)
            env._scorer._b2b, env._scorer._combo = b2b, combo

            for desc in _descriptors(env, env._active_piece):
                if not _has_key_sequence(env, desc):
                    unreachable.append((name, piece_type.name, desc))
                    continue

                a = clone_sim_env(env)
                b = clone_sim_env(env)
                seq = descriptor_key_sequence(env, desc)

                _r, _atk, _clears, _died = placement_step(a, searcher, desc)
                b._step(np.asarray(seq, dtype=np.int64))

                checked += 1
                spins_checked += desc[4] != 0
                board_a, b2b_a, combo_a = _state(a)
                board_b, b2b_b, combo_b = _state(b)
                if (
                    not np.array_equal(board_a != 0, board_b != 0)
                    or b2b_a != b2b_b
                    or combo_a != combo_b
                ):
                    mismatches.append(
                        (
                            name,
                            piece_type.name,
                            desc,
                            (b2b_a, combo_a),
                            (b2b_b, combo_b),
                        )
                    )

    # Floor guards against a vacuous run; O collapses 4:1 under dedup, bottoming near 360.
    assert checked >= 300, f"only {checked} placements checked for {piece_type.name}"
    # The gate must score at least one spin. O is absent from the fixtures: 2x2 symmetric
    # with trivial kicks, it yields no spin-labelled placements.
    if piece_type.name in _spin_fixture_pieces():
        assert spins_checked > 0, (
            f"no spin-labelled placement exercised for {piece_type.name}; the spin "
            f"fixtures no longer produce one"
        )
    assert not mismatches, (
        f"{len(mismatches)}/{checked} descriptor/key mismatches for "
        f"{piece_type.name}, first 3: {mismatches[:3]}"
    )
    # An enumerated placement the key path cannot reproduce means the descriptor path can
    # commit a move the real game cannot make.
    assert not unreachable, (
        f"{len(unreachable)}/{checked + len(unreachable)} enumerated placements have no "
        f"key sequence for {piece_type.name}, first 3: {unreachable[:3]}"
    )


def test_enumerator_and_scorer_agree_on_difficulty():
    """The enumerator's spin label and the env scorer's own classification must agree.

    Two live classifiers exist: `pathfinder.c:343-425` labels the descriptor, and
    `Scorer.py:33-98` re-derives the label when the keys are executed. They are independent
    implementations. Difficulty is the observable: for
    a 1-3 line clear, b2b increments iff the classifier saw a spin.

    A third copy lives at `b2b_search.c:669-750`, but it is `static` and reachable only
    through the beam/oracle datagen path, which the MCTS never calls. It is therefore not
    covered here; the two above are the ones on live paths.
    """
    searcher = CB2BSearch()
    base = _env()
    base._reset()

    disagreements = []
    spin_clears = 0
    with np.load(SPIN_BOARDS) as d:
        boards = [(k, d[k].astype(np.float32)) for k in sorted(d.files)]

    for name, board in boards:
        for piece_type in list(PieceType)[1:8]:
            env = _env()
            env._reset()
            env._board = board.copy()
            env._active_piece = env._spawn_piece(piece_type)
            env._scorer._b2b, env._scorer._combo = 0, -1  # b2b active, so a break shows

            for desc in _descriptors(env, env._active_piece):
                if not _has_key_sequence(env, desc):
                    continue
                probe = clone_sim_env(env)
                _r, _atk, clears, _died = placement_step(probe, searcher, desc)
                if not 1 <= clears <= 3:
                    continue  # a tetris is difficult regardless of spin
                keyed = clone_sim_env(env)
                keyed._step(np.asarray(descriptor_key_sequence(env, desc), np.int64))

                enum_difficult = desc[4] != 0
                scorer_difficult = keyed._scorer._b2b > 0
                spin_clears += enum_difficult
                if enum_difficult != scorer_difficult:
                    disagreements.append((name, piece_type.name, desc, clears))

    assert spin_clears > 0, "no spin-labelled clearing placement exercised"
    assert not disagreements, (
        f"{len(disagreements)} enumerator/scorer difficulty disagreements, "
        f"first 3: {disagreements[:3]}"
    )
