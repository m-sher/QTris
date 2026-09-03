"""4-wide mode: the side walls sit at one fixed height, and only the middle can top out."""

import numpy as np
import pytest
import tensorflow as tf

from TetrisEnv.Moves import Keys
from TetrisEnv.Pieces import PieceType
from TetrisEnv.PyTetrisEnv import (
    DEATH_HEIGHT_CAP,
    FOUR_WIDE_WALL_COLS,
    FOUR_WIDE_WALL_HEIGHT,
    PyTetrisEnv,
    apply_four_wide_walls,
)
from TetrisEnv.CB2BSearch import CB2BSearch
from qtris.search.cmcts import (
    CANDIDATE_CAPACITY,
    CMCTS,
    four_wide_wall_height,
    residual_match,
)
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS
from qtris.search.placement_search import (
    clone_sim_env,
    descriptor_key_sequence,
    placement_step,
)

PLAY_COLS = [3, 4, 5, 6]
DROP = np.array([Keys.START, Keys.HARD_DROP], dtype=np.int64)
SIMS = 16
LPR = 8


def _make_env(**kwargs) -> PyTetrisEnv:
    opts = dict(
        queue_size=5,
        max_holes=None,
        max_steps=None,
        max_len=15,
        pathfinding=False,
        garbage_chance=0.0,
        seed=0,
        idx=0,
        four_wide=True,
    )
    opts.update(kwargs)
    return PyTetrisEnv(**opts)


def _assert_walls_level(board: np.ndarray) -> None:
    """The wall columns hold exactly FOUR_WIDE_WALL_HEIGHT rows, filled bottom-up."""
    top = board.shape[0] - FOUR_WIDE_WALL_HEIGHT
    assert not board[:top, FOUR_WIDE_WALL_COLS].any()
    assert board[top:, FOUR_WIDE_WALL_COLS].all()


def _engine(four_wide: bool) -> CMCTS:
    return CMCTS(
        1,
        board_height=40,
        queue_size=5,
        max_holes=None,
        garbage_push_delay=1,
        auto_push_garbage=1,
        auto_fill_queue=1,
        c_puct=1.5,
        gamma=1.0,
        w_attack=0.0,
        w_death=1.0,
        return_scale=1.0,
        max_len=15,
        num_simulations=SIMS,
        leaves_per_round=LPR,
        vloss=1.0,
        w_b2b=0.0,
        q_norm=True,
        four_wide=four_wide,
    )


def _one_round(env: PyTetrisEnv, four_wide: bool):
    """Root the real C engine on `env`, expand it with flat priors and zero values, and run one
    round. Returns (number of leaves, their model-visible boards)."""
    engine = _engine(four_wide)
    try:
        engine.set_root(0, env)
        nv, _ = engine.collect_roots()
        assert nv == 1
        zeros = np.zeros(nv * CANDIDATE_CAPACITY, np.float32)
        engine.apply_roots(zeros, np.zeros(nv, np.float32), zeros.copy(), 0.0)
        nl, req = engine.collect_leaves()
        return nl, (req[0].copy() if nl else None)
    finally:
        engine.destroy()


class _FlatNet:
    """Stands in for the trained net: flat priors and a zero leaf value, so the visit counts
    are decided by the search's own edge rewards."""

    def policy_value(self, inputs):
        n = int(inputs[0].shape[0])
        logits = tf.zeros((n, CANDIDATE_CAPACITY), tf.float32)
        return logits, tf.zeros((n, 1), tf.float32)


def test_the_walls_stand_before_the_first_reset():
    env = _make_env()
    _assert_walls_level(env._board)


def test_reset_builds_the_walls():
    env = _make_env()
    env.reset()

    _assert_walls_level(env._board)
    assert not env._board[:, PLAY_COLS].any()
    top = env._board.shape[0] - FOUR_WIDE_WALL_HEIGHT
    assert (env._vis_board[top:, FOUR_WIDE_WALL_COLS] == PieceType.G.value).all()

    # As tall as they can be: one row under the cap, measured by the env's own height calc.
    assert int(np.max(env._get_heights(env._board))) == DEATH_HEIGHT_CAP - 1
    assert not env._is_top_out(env._board)


def test_walls_are_off_by_default():
    env = _make_env(four_wide=False)
    env.reset()
    assert not env._board.any()


def test_clearing_a_line_refills_the_walls():
    """A flat I spans exactly the four playable columns, so one hard drop completes a row on a
    fresh 4-wide board. The clear drops the walls a row, and the step must put it back."""
    env = _make_env()
    env.reset()
    env._active_piece = env._spawn_piece(PieceType.I)

    time_step = env._step(DROP)

    assert int(time_step.reward["clear"]) == 1
    _assert_walls_level(env._board)


def test_garbage_push_trims_the_walls_and_does_not_kill():
    """A J cannot fill four columns of a row, so the step takes its queued garbage. That shifts
    the walls past the death line, and trimming them back is what keeps the board alive."""
    env = _make_env()
    env.reset()
    env._active_piece = env._spawn_piece(PieceType.J)
    env._garbage_queue = [(4, 5, 0)]

    time_step = env._step(DROP)

    assert int(time_step.reward["clear"]) == 0
    assert float(time_step.reward["garbage_pushed"]) == 1.0
    assert not env._episode_ended
    _assert_walls_level(env._board)
    assert (env._board[-4:, 5] == 0.0).all()


def test_the_walls_stay_level_across_a_whole_game():
    """The re-level runs on every step, so play a real game and check after each move."""
    env = _make_env(garbage_chance=0.4, garbage_min=1, garbage_max=4, max_holes=100)
    env.reset()
    searcher = CB2BSearch()
    clears = pushes = 0
    for _ in range(60):
        _idx, seq, *_ = searcher.search_with_scores(
            board=env._board,
            active_piece=env._active_piece.piece_type.value,
            hold_piece=env._hold_piece.value,
            queue=np.array([p.value for p in env._queue], dtype=np.int32),
            b2b=int(env._scorer._b2b),
            combo=int(env._scorer._combo),
            total_garbage=0,
            garbage_push_delay=env._garbage_push_delay,
            search_depth=2,
            beam_width=64,
            max_len=15,
        )
        time_step = env._step(np.asarray(seq, dtype=np.int64))
        _assert_walls_level(env._board)
        clears += int(time_step.reward["clear"])
        pushes += int(float(time_step.reward["garbage_pushed"]))
        if time_step.is_last():
            break

    # Both directions have to have been exercised for the check above to mean anything.
    assert clears > 0 and pushes > 0, (clears, pushes)


def test_garbage_gap_lands_in_a_playable_column():
    env = _make_env(garbage_chance=1.0, garbage_min=1, garbage_max=1)
    env.reset()
    for _ in range(200):
        env._add_to_garbage_queue()
    cols = {col for _, col, _ in env._garbage_queue}
    assert cols and cols <= set(PLAY_COLS)

    ten_wide = _make_env(
        four_wide=False, garbage_chance=1.0, garbage_min=1, garbage_max=1
    )
    ten_wide.reset()
    for _ in range(200):
        ten_wide._add_to_garbage_queue()
    assert not {col for _, col, _ in ten_wide._garbage_queue} <= set(PLAY_COLS)


def test_only_the_middle_can_top_out():
    env = _make_env()
    board = np.zeros((40, 10), dtype=np.float32)
    apply_four_wide_walls(board)
    assert not env._is_top_out(board)

    stacked = board.copy()
    stacked[board.shape[0] - DEATH_HEIGHT_CAP :, PLAY_COLS] = 1.0
    assert env._is_top_out(stacked)

    blocked = board.copy()
    blocked[18, 4] = 1.0
    assert env._is_top_out(blocked)


def test_mcts_levels_the_walls_whatever_height_they_arrive_at():
    """Rooted on walls well under their legal height, the search must lift every in-tree board
    back to it, so the model-visible slice is solid in the wall columns."""
    env = _make_env()
    env.reset()
    env._board[:] = 0.0
    env._board[20:, FOUR_WIDE_WALL_COLS] = 1.0
    env._active_piece = env._spawn_piece(PieceType.J)

    nl, boards = _one_round(env, four_wide=True)
    assert nl > 0
    assert boards[:, :, FOUR_WIDE_WALL_COLS, 0].all()

    nl_off, boards_off = _one_round(env, four_wide=False)
    assert nl_off > 0
    assert not boards_off[:, :, FOUR_WIDE_WALL_COLS, 0].all()


def test_mcts_does_not_top_out_on_garbage_under_level_walls():
    """With the walls at their legal height, one queued row lifts them over the death line. Only
    trimming them in-tree keeps a child alive to be evaluated."""
    env = _make_env()
    env.reset()
    # Nothing reachable can complete a row, so every child takes the garbage push.
    env._active_piece = env._spawn_piece(PieceType.J)
    env._hold_piece = PieceType.J
    env._queue = [PieceType.J] * 5
    env._garbage_queue = [(1, 5, 0)]

    assert _one_round(env, four_wide=True)[0] > 0
    assert _one_round(env, four_wide=False)[0] == 0


def test_descriptor_step_matches_key_sequence_under_garbage():
    """The descriptor stepper is the AZ and eval commit path, and it runs the garbage and death
    block itself rather than going through `_step`. Both have to level the walls, or the two
    paths disagree the moment garbage arrives."""
    searcher = CB2BSearch()
    checked = 0
    for piece_type in (PieceType.I, PieceType.J, PieceType.T):
        env = _make_env(pathfinding=True)
        env.reset()
        env._active_piece = env._spawn_piece(piece_type)
        env._garbage_queue = [(4, 5, 0)]
        finder = env._key_sequence_finder
        rots, ncols, lrs, spins = finder.find_unique_placements(
            env._board, env._active_piece, 15, False, 512
        )
        for i in range(len(rots)):
            desc = (0, int(rots[i]), int(ncols[i]), int(lrs[i]), int(spins[i]))
            by_desc = clone_sim_env(env)
            by_keys = clone_sim_env(env)
            seq = descriptor_key_sequence(env, desc)

            placement_step(by_desc, searcher, desc)
            by_keys._step(np.asarray(seq, dtype=np.int64))

            checked += 1
            assert np.array_equal(by_desc._board != 0, by_keys._board != 0), desc
            _assert_walls_level(by_desc._board)

    assert checked >= 20, checked


def test_the_search_levels_to_the_same_height_the_env_builds():
    assert four_wide_wall_height() == FOUR_WIDE_WALL_HEIGHT


def test_the_flag_reaches_the_search_through_placement_mcts():
    """Config plumbing: the same root searched with the flag on and off must visit differently,
    since without it every child tops out on the queued row."""
    counts = []
    for four_wide in (True, False):
        env = _make_env()
        env.reset()
        env._active_piece = env._spawn_piece(PieceType.J)
        env._hold_piece = PieceType.J
        env._queue = [PieceType.J] * 5
        env._garbage_queue = [(1, 5, 0)]
        mcts = PlacementMCTS(
            _FlatNet(),
            MCTSConfig(
                num_simulations=SIMS,
                dirichlet_eps=0.0,
                leaves_per_round=LPR,
                gamma=1.0,
                w_attack=0.0,
                w_death=1.0,
                w_b2b=0.0,
                four_wide=four_wide,
            ),
        )
        res = mcts.search([env], 1.0, 0.0)[0]
        assert not res["dead"]
        counts.append(np.asarray(res["counts"], dtype=np.float64))

    assert not np.array_equal(counts[0], counts[1])


def test_the_cli_refuses_four_wide_outside_the_single_player_demo():
    """The 1v1 demo is stubbed in sys.modules so a missing guard fails on the absent
    SystemExit instead of launching the real thing."""
    import sys
    import types

    from qtris.cli.demo import main

    stub = types.SimpleNamespace(main=lambda args: None)
    argv = [
        "demo",
        "--checkpoint",
        "ckpt",
        "--four-wide",
        "--mode",
        "1v1",
        "--opponent",
        "opp",
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", argv)
        mp.setitem(sys.modules, "qtris.demo.placement_1v1", stub)
        with pytest.raises(SystemExit):
            main()


def test_four_wide_is_off_in_every_search_default():
    import inspect

    assert MCTSConfig().four_wide is False
    assert inspect.signature(CMCTS.__init__).parameters["four_wide"].default is False
    assert MCTSConfig().w_residual == 0.0
    assert inspect.signature(CMCTS.__init__).parameters["w_residual"].default == 0.0


def _walled_bit_board(mid_nibbles):
    """(40,) uint16 rows: level walls plus the given middle nibbles (top first) stacked
    on the floor. Nibble bit 0 is column 3."""
    rows = np.zeros(40, dtype=np.uint16)
    rows[40 - FOUR_WIDE_WALL_HEIGHT :] = 0x0387
    for i, nibble in enumerate(mid_nibbles):
        rows[40 - len(mid_nibbles) + i] |= nibble << 3
    return rows


def test_residual_matcher_recognizes_a_residual_and_demands_its_support():
    """4res #1 (col 3 for three rows, col 6 joining, cols 4-5 supported beneath): present
    support matches, an empty cell in the support row does not."""
    assert residual_match(_walled_bit_board([0x1, 0x1, 0x9, 0x6])) == 1
    assert residual_match(_walled_bit_board([0x1, 0x1, 0x9, 0x4])) == 0


def test_residual_matcher_uses_the_floor_as_support():
    """The same residual sitting directly on the floor: its support row falls below the
    board, and the floor supplies the required filled cells."""
    assert residual_match(_walled_bit_board([0x1, 0x1, 0x9])) == 1


def test_residual_matcher_ignores_rows_below_a_short_template():
    """4res #5 (three rows) on top of two full middle rows: the window rows past the
    template's height carry no requirement."""
    assert residual_match(_walled_bit_board([0x8, 0xE, 0x1, 0xF, 0xF])) == 1


def test_residual_matcher_rejects_non_residual_stacks():
    assert residual_match(_walled_bit_board([])) == 0
    assert residual_match(_walled_bit_board([0xF])) == 0


def test_residual_matcher_reads_an_env_board():
    """The 2D occupancy path packs like set_root: an O dropped against the left wall of a
    fresh 4-wide board leaves 4res #14."""
    env = _make_env(pathfinding=True)
    env.reset()
    env._active_piece = env._spawn_piece(PieceType.O)
    seq = np.full(15, Keys.PAD, dtype=np.int64)
    seq[:3] = [Keys.START, Keys.DAS_LEFT, Keys.HARD_DROP]
    env._step(seq)

    assert env._board[-2:, [3, 4]].all()
    assert residual_match(env._board) == 1


def _residual_counts(prime, piece, four_wide, w_residual):
    """Visit counts from a deterministic search: flat priors, zero leaf values, `prime`
    middle nibbles stacked on the floor, and a five-deep queue of one piece type. One leaf
    per round, so the counts are decided by the edge rewards alone."""
    env = _make_env()
    env.reset()
    for i, nibble in enumerate(prime):
        for j in range(4):
            if nibble & (1 << j):
                env._board[40 - len(prime) + i, 3 + j] = 1.0
    env._active_piece = env._spawn_piece(piece)
    env._hold_piece = piece
    env._queue = [piece] * 5
    mcts = PlacementMCTS(
        _FlatNet(),
        MCTSConfig(
            num_simulations=SIMS,
            dirichlet_eps=0.0,
            leaves_per_round=1,
            gamma=1.0,
            w_attack=0.0,
            w_death=1.0,
            w_b2b=0.0,
            four_wide=four_wide,
            w_residual=w_residual,
        ),
    )
    res = mcts.search([env], 1.0, 0.0)[0]
    assert not res["dead"]
    return np.asarray(res["counts"], dtype=np.float64)


def test_the_residual_weight_steers_the_search_only_in_four_wide():
    """Bottom rows primed so that the clearing placements are exactly the placements
    leaving a residual (a vertical I in column 3 clears into 5res #37, a flat I clears
    the row it lands on): the weight must re-rank the visits, and with the four_wide
    flag off it must change nothing."""
    prime = [0xC, 0xE]

    on = _residual_counts(prime, PieceType.I, True, 0.05)
    off = _residual_counts(prime, PieceType.I, True, 0.0)
    assert not np.array_equal(on, off)

    plain_on = _residual_counts(prime, PieceType.I, False, 0.05)
    plain_off = _residual_counts(prime, PieceType.I, False, 0.0)
    assert np.array_equal(plain_on, plain_off)


def test_the_residual_weight_ignores_non_clearing_matches():
    """O pieces on a fresh board: no single placement clears, and the two-piece clear
    empties the middle, so every residual match this budget reaches is a non-clearing
    one and the weight must not move the visits."""
    on = _residual_counts([], PieceType.O, True, 0.05)
    off = _residual_counts([], PieceType.O, True, 0.0)
    assert np.array_equal(on, off)
