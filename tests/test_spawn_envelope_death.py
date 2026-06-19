import numpy as np

from TetrisEnv.PyTetrisEnv import DEATH_HEIGHT_CAP, PyTetrisEnv, spawn_envelope_blocked


def _make_env() -> PyTetrisEnv:
    return PyTetrisEnv(
        queue_size=5,
        max_holes=50,
        max_height=18,
        max_steps=None,
        max_len=15,
        pathfinding=False,
        seed=0,
        idx=0,
    )


def _filled_board(height: int, empty_cols) -> np.ndarray:
    """A (40,10) occupancy board with every column filled to `height` rows except `empty_cols`,
    which are left completely empty."""
    board = np.zeros((40, 10), dtype=np.float32)
    # row 0 = top, so the bottom `height` rows are the deepest indices
    board[board.shape[0] - height :, :] = 1.0
    board[:, list(empty_cols)] = 0.0
    return board


def test_full_side_columns_just_below_cap_is_not_death():
    """Columns 0,1,2,7,8,9 filled to height 34 (one below the cap) with the middle columns
    3,4,5,6 empty: the spawn box is clear and no column hits the cap, so the env must NOT
    consider this a top-out."""
    env = _make_env()
    board = _filled_board(height=34, empty_cols=(3, 4, 5, 6))

    # Sanity, measured by the env's own height calc: tallest column is exactly 34, middle empty.
    assert int(np.max(env._get_heights(board))) == 34
    assert not board[:, [3, 4, 5, 6]].any()

    # The env's real top-out logic must report alive.
    assert not spawn_envelope_blocked(board)
    assert not env._is_top_out(board)


def test_height_cap_triggers_death():
    """Positive control for the separate height-cap check: the same middle-empty board taken
    one row higher (to the cap) is a top-out, via the cap, not the (still-empty) spawn box."""
    env = _make_env()
    board = _filled_board(height=DEATH_HEIGHT_CAP, empty_cols=(3, 4, 5, 6))

    assert int(np.max(env._get_heights(board))) == DEATH_HEIGHT_CAP
    assert not spawn_envelope_blocked(board)  # death comes from the cap, not the box
    assert env._is_top_out(board)


def test_blocked_spawn_box_triggers_death():
    """Positive control for the envelope check: a single cell inside the spawn box is a top-out
    even though the board is far below the height cap."""
    env = _make_env()
    board = np.zeros((40, 10), dtype=np.float32)
    board[18, 4] = 1.0  # inside the spawn box (row 18, col 4); height here is only 22

    assert spawn_envelope_blocked(board)
    assert int(np.max(env._get_heights(board))) < DEATH_HEIGHT_CAP
    assert env._is_top_out(board)
