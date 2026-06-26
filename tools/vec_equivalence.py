"""Vectorized-parity harness: compares the AsyncVectorEnv + TFVecEnv stack bit-exact
against the committed golden over a deterministic action stream crossing autoreset
boundaries. Isolates the wrapper (batching + autoreset + tensor I/O); per-step env logic
is covered by env_equivalence.py.

  compare: python tools/vec_equivalence.py --kind single
  capture: python tools/vec_equivalence.py --capture --kind single
"""

import argparse
import hashlib
import json
import random

import numpy as np

from TetrisEnv.Moves import Keys
from TetrisEnv.PyTetris1v1Env import PyTetris1v1Env
from TetrisEnv.PyTetrisEnv import PyTetrisEnv

ENV_PARAMS = dict(
    queue_size=5,
    max_holes=50,
    max_len=15,
    pathfinding=True,
    num_row_tiers=2,
)
MOVE_KEYS = [Keys.HOLD, 2, 3, 4, 5, 6, 7, 8, 9]  # holds/laterals/rotations/soft drop


def _make_env(kind, seed, idx):
    cls = PyTetrisEnv if kind == "single" else PyTetris1v1Env
    return cls(seed=seed, idx=idx, max_steps=None, **ENV_PARAMS)


def _action_batch(rng, num_envs, action_dim):
    """State-independent key-sequence batch; every 15-key segment ends in HARD_DROP.
    1v1 action_dim=30 is two segments."""
    out = np.full((num_envs, action_dim), Keys.PAD, dtype=np.int64)
    for e in range(num_envs):
        for base in range(0, action_dim, 15):
            seq = [rng.choice(MOVE_KEYS) for _ in range(rng.randrange(5))] + [
                Keys.HARD_DROP
            ]
            out[e, base : base + len(seq)] = seq
    return out


def _build_new(kind, num_envs):
    import tensorflow as tf

    from TetrisEnv.tf_vec_env import make_tf_vec_env

    constructors = [(lambda i=i: _make_env(kind, 2000 + i, i)) for i in range(num_envs)]
    venv = make_tf_vec_env(constructors)

    class NewVec:
        def reset(self):
            res = venv.reset()
            return {k: np.asarray(v) for k, v in res.observation.items()}

        def step(self, actions):
            res = venv.step(tf.constant(actions))
            obs = {k: np.asarray(v) for k, v in res.observation.items()}
            reward = {k: np.asarray(v) for k, v in res.reward.items()}
            return obs, reward, np.asarray(res.done)

    return NewVec()


def _digest(obs, reward, done):
    h = hashlib.sha256()
    for k in sorted(obs):
        a = np.ascontiguousarray(obs[k])
        h.update(f"{k}{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    for k in sorted(reward):
        h.update(k.encode())
        h.update(np.ascontiguousarray(reward[k]).tobytes())
    h.update(np.ascontiguousarray(done).astype(np.int8).tobytes())
    return h.hexdigest()


def run(kind, num_envs, steps):
    venv = _build_new(kind, num_envs)
    action_dim = 15 if kind == "single" else 30
    rng = random.Random(f"vec-{kind}-{num_envs}")

    chain = hashlib.sha256()
    per_step = []
    venv.reset()
    for _ in range(steps):
        obs, reward, done = venv.step(_action_batch(rng, num_envs, action_dim))
        d = _digest(obs, reward, done)
        per_step.append(d)
        chain.update(d.encode())
    return {
        "meta": {
            "kind": kind,
            "num_envs": num_envs,
            "steps": steps,
            "env_params": ENV_PARAMS,
        },
        "chain": chain.hexdigest(),
        "per_step": per_step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["single", "1v1"], default="single")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--golden", default=None)
    args = ap.parse_args()

    golden_path = args.golden or f"tools/vec_golden_{args.kind}.json"
    fresh = run(args.kind, args.envs, args.steps)
    print(f"vec {args.kind}: {args.envs} envs x {args.steps} steps")

    if args.capture:
        with open(golden_path, "w") as f:
            json.dump(fresh, f)
        print(f"wrote golden -> {golden_path}")
        return 0

    with open(golden_path) as f:
        golden = json.load(f)
    if golden["meta"] != fresh["meta"]:
        print(f"MISMATCH meta: golden={golden['meta']} fresh={fresh['meta']}")
        return 1
    if golden["chain"] != fresh["chain"]:
        for j, (a, b) in enumerate(zip(golden["per_step"], fresh["per_step"])):
            if a != b:
                print(
                    f"MISMATCH: batched step {j} differs (of {len(golden['per_step'])})"
                )
                return 1
        print("MISMATCH: chain differs but per-step equal (length?)")
        return 1
    print(
        f"EQUIVALENT: vec {args.kind} {args.steps} batched steps bit-exact vs {golden_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
