"""Expert replay frames measured the same way as scripts/mirror_gauge.py.

Usage: uv run python scripts/expert_frames.py OUT.json REPLAY.json [REPLAY.json ...].
Records per frame the board height, holes, bumpiness, bank, pending garbage, what the
expert did next, and how many candidates offered a difficult clear."""

import json
import sys

import numpy as np
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Pieces import PieceType

from qtris.search.cmcts import CMCTS
from qtris.training._1v1_placement_az import _build_game_pairs

out = sys.argv[1]
files = sys.argv[2:]
env = _build_game_pairs(1, 5, 50, 15)[0][0]
env._reset()
searcher = CB2BSearch()


def load(env, fr):
    stored = np.asarray(fr["board"], dtype=np.float32)
    b = np.zeros((40, 10), dtype=np.float32)
    rows = min(stored.shape[0], 40)
    b[-rows:] = stored[-rows:]
    env._board = b
    env._vis_board = b.copy()
    pieces = [int(p) for p in fr["pieces"]]
    env._active_piece = env._spawn_piece(PieceType(pieces[0]))
    env._hold_piece = PieceType(pieces[1])
    env._queue = [PieceType(p) for p in pieces[2 : 2 + env._queue_size]]
    env._next_bag = [
        PieceType(p) for p in pieces[2 + env._queue_size : 2 + env._queue_size + 7]
    ]
    b2b, combo, _ = fr["b2b_combo_garbage"]
    env._scorer._b2b = int(b2b)
    env._scorer._combo = int(combo)
    env._garbage_queue = [
        (int(g[0]), 0, env._garbage_push_delay) for g in fr.get("garbage_queue") or []
    ]


def candidates(env):
    engine = CMCTS(1, board_height=40, queue_size=5, max_holes=50, max_len=15)
    try:
        engine.set_root(0, env)
        nv, req = engine.collect_roots()
        if not nv:
            return []
        _pi, _c, desc, _d, _rv = engine.result()
        mask = np.array(req[4][0], dtype=bool)
        desc = np.array(desc[0], np.int64)
    finally:
        engine.destroy()
    res = []
    for slot in np.flatnonzero(mask):
        is_hold, rot, norm_col, landing_row, spin = (int(x) for x in desc[slot])
        if is_hold:
            placed = (
                env._queue[0] if env._hold_piece == PieceType.N else env._hold_piece
            )
        else:
            placed = env._active_piece.piece_type
        _b, clears, attack, new_b2b, _combo = searcher.lock_score(
            env._board,
            placed.value,
            rot,
            norm_col,
            landing_row,
            spin,
            env._scorer._b2b,
            env._scorer._combo,
        )
        res.append((int(clears), float(attack), int(new_b2b), int(spin)))
    return res


rows = []
for fn in files:
    frames = json.load(open(fn))["frames"]
    for i, fr in enumerate(frames):
        load(env, fr)
        h, holes, sky, bump = env._board_stats(env._board)
        b2b = env._scorer._b2b
        cands = candidates(env)
        nxt = frames[i + 1] if i + 1 < len(frames) else None
        row = dict(
            file=fn.split("/")[-1],
            idx=i,
            used=fr.get("used") is not False,
            h=int(h),
            holes=int(holes),
            bump=int(bump),
            pre_b2b=int(b2b),
            pre_combo=int(env._scorer._combo),
            pending=int(env._get_total_garbage()),
            n_cands=len(cands),
            n_clear=sum(1 for c in cands if c[0] > 0),
            n_difficult=sum(1 for c in cands if c[0] > 0 and c[2] == b2b + 1),
            n_spin_clear=sum(1 for c in cands if c[0] > 0 and c[3] > 0),
            n_tetris=sum(1 for c in cands if c[0] == 4),
            n_break=sum(1 for c in cands if c[0] > 0 and c[2] == -1),
        )
        if nxt is not None:
            nb2b, ncombo, _ = nxt["b2b_combo_garbage"]
            nb2b, ncombo = int(nb2b), int(ncombo)
            row.update(
                post_b2b=nb2b,
                difficult=bool(nb2b == b2b + 1),
                broke=bool(b2b >= 0 and nb2b == -1),
                cleared=bool(ncombo == env._scorer._combo + 1),
            )
        rows.append(row)
json.dump(rows, open(out, "w"))
print("rows", len(rows))
