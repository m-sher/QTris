"""Measure the AZ placement bot's outgoing attack distribution, and optionally
record attack-stream traces for trace-replay garbage (see --garbage-traces in
`train placement --algo az`).

Measurement:  uv run python scripts/measure_attack_dist.py
Trace library: uv run python scripts/measure_attack_dist.py \
    --save-traces garbage_traces --tiers 16,64,128,256
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import tensorflow as tf

from TetrisEnv.CB2BSearch import CB2BSearch
from TetrisEnv.Moves import Keys
from qtris.data.placement_features import CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM
from qtris.models.placement.model import PlacementPolicyValueNet
from qtris.search.placement_mcts import MCTSConfig, PlacementMCTS
from qtris.training.placement_az import _build_envs, placement_step

CHUNK_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 9), (10, 99)]


def _load_net(checkpoint, num_envs):
    net = PlacementPolicyValueNet(
        batch_size=num_envs,
        piece_dim=8,
        depth=64,
        num_heads=4,
        num_layers=4,
        dropout_rate=0.0,
    )
    net(
        (
            keras.Input(shape=(24, 10, 1), dtype=tf.float32),
            keras.Input(shape=(7,), dtype=tf.int64),
            keras.Input(shape=(3,), dtype=tf.float32),
            keras.Input(
                shape=(CANDIDATE_CAPACITY, PLACEMENT_FEATURE_DIM), dtype=tf.float32
            ),
            keras.Input(shape=(CANDIDATE_CAPACITY,), dtype=tf.bool),
        )
    )
    scale = tf.Variable(1.0, trainable=False)
    ck = tf.train.latest_checkpoint(checkpoint)
    tf.train.Checkpoint(model=net, return_scale=scale).restore(ck).expect_partial()
    print(f"ckpt: {ck}  return_scale={float(scale):.2f}")
    return net, float(scale)


def _rollout(net, scale, sims, level, num_envs, pieces, seed=7):
    """Greedy MCTS self-play at one garbage level; returns the per-placement
    attack streams (pieces, num_envs), deaths, and mean b2b."""
    np.random.seed(seed)
    args = SimpleNamespace(
        garbage_chance_min=level,
        garbage_chance_max=level,
        garbage_rows_min=1,
        garbage_rows_max=4,
    )
    envs = _build_envs(
        num_envs, 5, max_holes=50, max_height=18, max_steps=None, max_len=15, args=args
    )
    for e in envs:
        e._reset()
    mcts = PlacementMCTS(
        net,
        MCTSConfig(
            num_simulations=sims, c_puct=1.5, dirichlet_eps=0.0, leaves_per_round=8
        ),
    )
    searcher = CB2BSearch()
    forced = np.full(15, Keys.PAD, dtype=np.int64)
    forced[0], forced[1] = Keys.START, Keys.HARD_DROP

    atk = np.zeros((pieces, num_envs), np.float32)
    deaths = 0
    b2bs = []
    for t in range(pieces):
        res = mcts.search(envs, scale, np.zeros(num_envs, np.float32))
        for i, r in enumerate(res):
            b2bs.append(envs[i]._scorer._b2b)
            if r["dead"]:
                envs[i]._step(forced.copy())
                deaths += 1
                envs[i]._reset()
                continue
            _total, a, _clear, died = placement_step(envs[i], searcher, r["descriptor"])
            atk[t, i] = a
            if died or envs[i]._episode_ended:
                deaths += 1
                envs[i]._reset()
    return atk, deaths, float(np.mean(b2bs))


def _report(level, atk, deaths, b2b):
    a = atk.reshape(-1)
    ev = a[a > 0]
    spikes = []
    for i in range(atk.shape[1]):
        run = 0.0
        for v in atk[:, i]:
            if v > 0:
                run += v
            elif run > 0:
                spikes.append(run)
                run = 0.0
        if run > 0:
            spikes.append(run)
    hist = " ".join(
        f"{((ev >= lo) & (ev <= hi)).mean() if ev.size else 0.0:5.2f}"
        for lo, hi in CHUNK_BINS
    )
    print(
        f"{level:5.2f} {a.mean():6.3f} {(a > 0).mean():7.3f} "
        f"{ev.mean() if ev.size else 0.0:6.2f} {hist} "
        f"{np.mean(spikes) if spikes else 0.0:8.2f} "
        f"{np.max(spikes) if spikes else 0.0:8.1f} {deaths:6d} {b2b:5.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/placement_az")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--pieces", type=int, default=256)
    ap.add_argument("--sims", type=int, default=128, help="measurement-mode sims.")
    ap.add_argument(
        "--levels",
        default="0,0.05,0.1,0.15,0.2",
        help="garbage-chance levels for measurement mode.",
    )
    ap.add_argument(
        "--save-traces",
        default=None,
        help="trace-library dir; switches to generation mode (one tier per --tiers).",
    )
    ap.add_argument(
        "--tiers",
        default="16,64,128,256",
        help="sims per tier in generation mode (sorted = difficulty order).",
    )
    ap.add_argument(
        "--gen-level",
        type=float,
        default=0.15,
        help="garbage level the trace games are played under in generation mode.",
    )
    args = ap.parse_args()

    net, scale = _load_net(args.checkpoint, args.envs)

    if args.save_traces:
        out = Path(args.save_traces)
        for idx, sims in enumerate(int(s) for s in args.tiers.split(",")):
            tier = out / f"{idx:02d}_sims{sims}"
            tier.mkdir(parents=True, exist_ok=True)
            atk, deaths, b2b = _rollout(
                net, scale, sims, args.gen_level, args.envs, args.pieces, seed=11 + idx
            )
            for i in range(args.envs):
                if atk[:, i].any():
                    np.save(tier / f"e{i:02d}.npy", atk[:, i])
            print(
                f"{tier}: {int((atk.any(axis=0)).sum())} traces, "
                f"APP {atk.mean():.3f}, deaths {deaths}, b2b {b2b:.2f}"
            )
        return

    print(
        f"\n{'gch':>5s} {'APP':>6s} {'evrate':>7s} {'chunk':>6s} "
        + " ".join(
            f"{f'p{lo}' if lo == hi else f'p{lo}-{hi}':>5s}" for lo, hi in CHUNK_BINS
        )
        + f" {'spike_mu':>8s} {'spike_mx':>8s} {'deaths':>6s} {'b2b':>5s}"
    )
    for level in (float(x) for x in args.levels.split(",")):
        atk, deaths, b2b = _rollout(
            net, scale, args.sims, level, args.envs, args.pieces
        )
        _report(level, atk, deaths, b2b)


if __name__ == "__main__":
    main()
