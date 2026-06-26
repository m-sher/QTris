"""Differential equivalence harness for the tf-agents -> gymnasium env migration.

Drives PyTetrisEnv / PyTetris1v1Env over a large seeded sweep with a deterministic
valid-placement policy, hashing every (observation, reward channels, done) per step.
Capture golden on the current (tf-agents) code, then compare bit-exact after each phase.

  capture: python tools/env_equivalence.py --capture --kind single --n 1000 --t 200
  compare: python tools/env_equivalence.py --kind single --n 1000 --t 200

Version-agnostic: detects gymnasium.Env vs the tf-agents PyEnvironment and extracts the
same canonical (obs dict, channel dict, done bool) from either API.
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
SUBSET = 10  # rollouts whose per-step digests are stored, to localize a mismatch


def _is_gym(env):
    try:
        import gymnasium

        return isinstance(env, gymnasium.Env)
    except Exception:
        return False


def _reset(env, seed):
    """Return the observation dict from a reset (seed=None means autoreset-style re-seed)."""
    if _is_gym(env):
        obs, _info = env.reset(seed=seed)
        return dict(obs)
    return dict(env._reset().observation)


def _step(env, action):
    """Return (obs dict, channel dict, done) from one step, from either API."""
    if _is_gym(env):
        obs, _reward, term, trunc, info = env.step(action)
        return dict(obs), dict(info), bool(term or trunc)
    ts = env._step(action)
    return dict(ts.observation), dict(ts.reward), bool(ts.is_last())


_FORCED = np.array([Keys.START, Keys.HARD_DROP] + [Keys.PAD] * 13, dtype=np.int64)


def _pick_single(env, rng):
    """A valid placement key sequence for `env`, or a forced hard drop if none.
    Valid slots are scores > -1e30 (the env's sentinel); unfilled slots keep rows=0,
    so rows is not a validity test. Valid seqs always carry an executable HARD_DROP."""
    scores, _rows, seqs = env._enumerate_placement_candidates()
    valid = np.flatnonzero(scores > -1e30)
    if valid.size == 0:
        return _FORCED.copy()
    seq = seqs[valid[rng.randrange(valid.size)]].astype(np.int64)
    return seq if (seq == Keys.HARD_DROP).any() else _FORCED.copy()


def _pick_1v1(env, rng):
    return np.concatenate([_pick_single(env._env1, rng), _pick_single(env._env2, rng)])


def _digest_step(obs, channels, done):
    h = hashlib.sha256()
    for k in sorted(obs):
        a = np.ascontiguousarray(obs[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    for k in sorted(channels):
        h.update(k.encode())
        h.update(np.ascontiguousarray(channels[k]).tobytes())
    h.update(bytes([int(done)]))
    return h.hexdigest()


def _rollout(kind, seed, max_steps):
    """One rollout: reset(seed), then step a valid-placement policy up to max_steps,
    resetting on done (re-seed-from-RNG path). Returns (chain_hex, [per-step hex])."""
    if kind == "single":
        env = PyTetrisEnv(seed=seed, idx=0, max_steps=None, **ENV_PARAMS)
        pick = _pick_single
    else:
        env = PyTetris1v1Env(seed=seed, idx=0, max_steps=None, **ENV_PARAMS)
        pick = _pick_1v1
    action_rng = random.Random(f"act-{kind}-{seed}")

    chain = hashlib.sha256()
    steps = []
    obs = _reset(env, seed)
    chain.update(_digest_step(obs, {}, False).encode())
    for _ in range(max_steps):
        action = pick(env, action_rng)
        obs, channels, done = _step(env, action)
        d = _digest_step(obs, channels, done)
        steps.append(d)
        chain.update(d.encode())
        if done:
            obs = _reset(env, None)
            chain.update(_digest_step(obs, {}, False).encode())
    return chain.hexdigest(), steps


def run_sweep(kind, n_seeds, max_steps):
    rollups, subset_steps = {}, {}
    total = 0
    for i in range(n_seeds):
        seed = 10_000 + i
        chain_hex, steps = _rollout(kind, seed, max_steps)
        rollups[str(seed)] = chain_hex
        total += len(steps)
        if i < SUBSET:
            subset_steps[str(seed)] = steps
    return {
        "meta": {
            "kind": kind,
            "n_seeds": n_seeds,
            "max_steps": max_steps,
            "env_params": ENV_PARAMS,
            "total_steps": total,
        },
        "rollups": rollups,
        "subset_steps": subset_steps,
    }


def _compare(golden, fresh):
    """Compare fresh rollouts against golden. fresh may be a subset of golden's seeds
    (fewer --n) for a fast partial gate, but kind/max_steps/env_params must match so the
    same seed yields the same rollout."""
    gm, fm = golden["meta"], fresh["meta"]
    for k in ("kind", "max_steps", "env_params"):
        if gm[k] != fm[k]:
            return [f"meta mismatch {k}: golden={gm[k]} fresh={fm[k]}"]
    diffs = []
    for seed, fh in fresh["rollups"].items():
        gh = golden["rollups"].get(seed)
        if gh is None:
            diffs.append(f"rollout {seed}: not in golden (golden too small)")
        elif fh != gh:
            msg = f"rollout {seed}: chain differs"
            gsteps = golden["subset_steps"].get(seed)
            fsteps = fresh["subset_steps"].get(seed)
            if gsteps and fsteps:
                for j, (a, b) in enumerate(zip(gsteps, fsteps)):
                    if a != b:
                        msg += f" (first diff at step {j})"
                        break
            diffs.append(msg)
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["single", "1v1"], default="single")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--t", type=int, default=200)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--golden", default=None)
    args = ap.parse_args()

    golden_path = args.golden or f"tools/env_golden_{args.kind}.json"
    fresh = run_sweep(args.kind, args.n, args.t)
    print(
        f"{args.kind}: {args.n} seeds x {args.t} -> {fresh['meta']['total_steps']} steps"
    )

    if args.capture:
        with open(golden_path, "w") as f:
            json.dump(fresh, f)
        print(f"wrote golden -> {golden_path}")
        return 0

    with open(golden_path) as f:
        golden = json.load(f)
    diffs = _compare(golden, fresh)
    if diffs:
        print(f"MISMATCH ({len(diffs)} rollouts):")
        for d in diffs[:20]:
            print("  " + d)
        return 1
    print(
        f"EQUIVALENT: {len(fresh['rollups'])}/{len(golden['rollups'])} golden rollouts "
        f"bit-exact ({fresh['meta']['total_steps']} steps) vs {golden_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
