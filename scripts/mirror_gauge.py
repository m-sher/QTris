"""Mirror self-play gauge of the current search on the latest 1v1 checkpoint.

Usage: uv run python scripts/mirror_gauge.py OUT.json [games] [steps] [w_holes=0.16 ...].
Searches noise-free at 256 simulations, 8 leaves per round, gamma=1 and w_death=1,
taking every other setting from MCTSConfig unless a key=value argument overrides it.
Records per placement the board height, holes, bumpiness, bank, pending garbage, what
the placement did, and how many candidates offered a difficult clear."""

import json
import random
import sys

import numpy as np
import tensorflow as tf
from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Pieces import PieceType

from qtris.search.cmcts import CMCTS
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS
from qtris.training._1v1_placement_az import (
    _build_game_pairs,
    _build_net,
    _commit_and_exchange,
)

out = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 220
over = {k: float(v) for k, v in (a.split("=") for a in sys.argv[4:])}
cfg = MCTSConfig(
    num_simulations=256,
    leaves_per_round=8,
    gamma=1.0,
    w_death=1.0,
    dirichlet_eps=0.0,
    **over,
)
print(
    "weights",
    {
        k: getattr(cfg, k)
        for k in (
            "w_attack",
            "w_b2b",
            "w_height",
            "w_bumpiness",
            "w_holes",
            "w_plain",
            "fpu",
        )
    },
    flush=True,
)
net = _build_net(N, 8, 64, 4, 4, 5)
ck = tf.train.Checkpoint(model=net)
ck.restore(tf.train.latest_checkpoint("checkpoints/1v1_placement_az")).expect_partial()
mcts = PlacementMCTS(net, cfg)
pairs = _build_game_pairs(N, 5, 50, 15, seed0=777)
for e1, e2 in pairs:
    e1._reset()
    e2._reset()
searcher = CB2BSearch()
rng = random.Random(1)


def candidates(env):
    """Per legal candidate: (clears, attack, new_b2b, spin) via the C lock core."""
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
alive = [True] * N
for t in range(STEPS):
    idx = [g for g in range(N) if alive[g]]
    if not idx:
        break
    temps = np.zeros(len(idx), np.float32)
    r1 = mcts.search([pairs[g][0] for g in idx], 1.0, temps)
    r2 = mcts.search([pairs[g][1] for g in idx], 1.0, temps)
    for j, g in enumerate(idx):
        a, b = r1[j], r2[j]
        e1, e2 = pairs[g]
        if a["dead"] or b["dead"]:
            alive[g] = False
            rows.append(
                dict(
                    game=g,
                    step=t,
                    event="end",
                    p1_dead=bool(a["dead"]),
                    p2_dead=bool(b["dead"]),
                )
            )
            continue
        pre = []
        for env in (e1, e2):
            h, holes, sky, bump = env._board_stats(env._board)
            cands = candidates(env)
            b2b = env._scorer._b2b
            pre.append(
                dict(
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
            )
        d1, d2 = a["descriptor"], b["descriptor"]
        pre1b, pre2b = e1._scorer._b2b, e2._scorer._b2b
        p1_died, p2_died, atk1, atk2 = _commit_and_exchange(
            e1, e2, searcher, d1, d2, rng
        )
        for k, (env, d, atk, died, preb) in enumerate(
            ((e1, d1, atk1, p1_died, pre1b), (e2, d2, atk2, p2_died, pre2b))
        ):
            post_b2b = env._scorer._b2b
            row = dict(
                game=g,
                step=t,
                player=k + 1,
                event="place",
                spin=int(d[4]),
                attack=float(atk),
                post_b2b=int(post_b2b),
                died=bool(died),
                difficult=bool(post_b2b == preb + 1),
                broke=bool(preb >= 0 and post_b2b == -1),
                cleared=bool(
                    post_b2b == preb + 1
                    or (preb >= 0 and post_b2b == -1)
                    or (preb == -1 and env._scorer._combo >= 0)
                ),
            )
            row.update(pre[k])
            rows.append(row)
        if p1_died or p2_died:
            alive[g] = False
    if t % 20 == 0:
        print("step", t, "alive", sum(alive), flush=True)
json.dump(rows, open(out, "w"))
print("rows", len(rows))
