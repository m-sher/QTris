"""Multi-landing placement enum + MCTS packing tests."""

from TetrisEnv.CKeySequences import CKeySequenceFinder
from TetrisEnv.Pieces import PieceType
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from qtris.data.placement_features import (
    CANDIDATE_CAPACITY as BC_CAPACITY,
    MCTS_CANDIDATE_CAPACITY,
)
from qtris.search.cmcts import CMCTS, CANDIDATE_CAPACITY as MCTS_CAP
from qtris.search.placement_search import descriptor_key_sequence


def _overhang_board(env):
    b = env._board.copy()
    b[-2:, :] = 1.0
    b[-12:-10, 0:6] = 1.0
    b[-20:-18, 4:10] = 1.0
    return b


def test_bc_capacity_unchanged():
    assert BC_CAPACITY == 128
    assert MCTS_CANDIDATE_CAPACITY == 256
    assert MCTS_CAP == MCTS_CANDIDATE_CAPACITY


def test_unique_exceeds_dense_on_overhang():
    finder = CKeySequenceFinder()
    env = PyTetrisEnv(
        queue_size=5,
        max_holes=50,
        max_steps=None,
        max_len=15,
        pathfinding=True,
        garbage_chance=0.0,
        auto_push_garbage=False,
        auto_fill_queue=True,
        seed=0,
        idx=0,
        num_row_tiers=2,
    )
    b = _overhang_board(env)
    piece = env._spawn_piece(PieceType.T)
    _, lr_d = finder.find_placement_candidates(b, piece, 15, False)
    rots, ncols, lrs, spins = finder.find_unique_placements(b, piece, 15, False, 512)
    assert len(rots) > int((lr_d >= 0).sum())
    # dense keeps deepest landing only for a multi-landing pose
    from collections import defaultdict

    by = defaultdict(list)
    for i in range(len(rots)):
        by[(int(rots[i]), int(ncols[i]), int(spins[i]))].append(int(lrs[i]))
    multi = {k: v for k, v in by.items() if len(v) > 1}
    assert multi
    k = next(iter(multi))
    slot = k[0] * 40 + k[1] * 4 + k[2]
    assert int(lr_d[slot]) == max(multi[k])


def test_mcts_packs_multiple_landings_and_handshake():
    env = PyTetrisEnv(
        queue_size=5,
        max_holes=50,
        max_steps=None,
        max_len=15,
        pathfinding=False,
        garbage_chance=0.0,
        auto_push_garbage=False,
        auto_fill_queue=False,
        seed=0,
        idx=0,
        num_row_tiers=2,
    )
    env._board = _overhang_board(env)
    env._active_piece = env._spawn_piece(PieceType.T)
    env._hold_piece = PieceType.I
    env._queue = [PieceType.J] * 5
    engine = CMCTS(
        1,
        board_height=40,
        queue_size=5,
        max_holes=50,
        garbage_push_delay=1,
        auto_push_garbage=0,
        auto_fill_queue=0,
        c_puct=1.5,
        gamma=1.0,
        w_attack=0.05,
        w_death=1.0,
        return_scale=1.0,
        max_len=15,
        num_simulations=32,
        leaves_per_round=4,
        vloss=1.0,
    )
    try:
        engine.set_root(0, env)
        nv, req = engine.collect_roots()
        assert nv == 1
        n_legal = int(req[4][0].sum())
        assert n_legal > 68
        pi, counts, desc, dead = engine.result()
        landings = {int(desc[0, j, 3]) for j in range(MCTS_CAP) if desc[0, j, 0] >= 0}
        assert len(landings) > 1
        # resolve sequence for a no-hold mid-height landing when present
        nh_lands = sorted(
            {int(desc[0, j, 3]) for j in range(MCTS_CAP) if desc[0, j, 0] == 0}
        )
        assert nh_lands
        target = nh_lands[len(nh_lands) // 2]
        for j in range(MCTS_CAP):
            if desc[0, j, 0] == 0 and int(desc[0, j, 3]) == target:
                d = tuple(int(x) for x in desc[0, j])
                seq = descriptor_key_sequence(env, d, 15)
                assert seq[0] == 0  # START
                break
        else:
            raise AssertionError("no no-hold target landing")
    finally:
        engine.destroy()
