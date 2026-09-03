"""Mirror self-play under training noise, dumping both players' positions as
replay-style frames for scripts/search_stats.py.

Searches at 256 simulations, 8 leaves per round, gamma=1 and w_death=1, taking every
other setting from MCTSConfig.

Usage: uv run python scripts/onpolicy_frames.py OUT.json [games] [steps]."""

import json
import random
import sys

import numpy as np
import tensorflow as tf
from TetrisEnv.CB2BSearch import CB2BSearch

from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS
from qtris.training._1v1_placement_az import (
    _build_game_pairs,
    _build_net,
    _commit_and_exchange,
)

out = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 150
cfg = MCTSConfig(num_simulations=256, leaves_per_round=8, gamma=1.0, w_death=1.0)
net = _build_net(N, 8, 64, 4, 4, 5)
tf.train.Checkpoint(model=net).restore(
    tf.train.latest_checkpoint("checkpoints/1v1_placement_az")
).expect_partial()
mcts = PlacementMCTS(net, cfg)
pairs = _build_game_pairs(N, 5, 50, 15, seed0=4242)
for e1, e2 in pairs:
    e1._reset()
    e2._reset()
searcher = CB2BSearch()
rng = random.Random(3)
np.random.seed(3)


def frame(env, game, step, player):
    pieces = (
        [env._active_piece.piece_type.value, env._hold_piece.value]
        + [p.value for p in env._queue]
        + [p.value for p in env._next_bag]
    )
    return dict(
        board=env._board[-24:].astype(int).tolist(),
        pieces=pieces,
        b2b_combo_garbage=[
            float(env._scorer._b2b),
            float(env._scorer._combo),
            float(env._get_total_garbage()),
        ],
        garbage_queue=[[int(g[0]), int(g[1]), int(g[2])] for g in env._garbage_queue],
        used=True,
        game=game,
        step=step,
        player=player,
    )


frames = []
alive = [True] * N
moves = np.zeros(N, np.int64)
for t in range(STEPS):
    idx = [g for g in range(N) if alive[g]]
    if not idx:
        break
    temps = np.where(moves[idx] < cfg.temp_moves, 1.0, 0.0).astype(np.float32)
    for g in idx:
        frames.append(frame(pairs[g][0], g, t, 1))
        frames.append(frame(pairs[g][1], g, t, 2))
    r1 = mcts.search([pairs[g][0] for g in idx], 1.0, temps)
    r2 = mcts.search([pairs[g][1] for g in idx], 1.0, temps)
    for j, g in enumerate(idx):
        a, b = r1[j], r2[j]
        if a["dead"] or b["dead"]:
            alive[g] = False
            continue
        d1, d2 = _commit_and_exchange(
            pairs[g][0], pairs[g][1], searcher, a["descriptor"], b["descriptor"], rng
        )[:2]
        moves[g] += 1
        if d1 or d2:
            alive[g] = False
    if t % 25 == 0:
        print("step", t, "alive", sum(alive), "frames", len(frames), flush=True)
json.dump({"mode": "onpolicy", "frames": frames}, open(out, "w"))
print("frames", len(frames))
