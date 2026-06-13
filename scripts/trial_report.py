"""Trial discriminator report for the collapse campaign (see collapse_plan.md).

Usage: uv run python trials/analyze.py tb_logs/Tetris/<run> [<run> ...]
"""

import sys

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {
    "update_kl": "optimization/update_kl",
    "policy_kl": "optimization/policy_kl",
    "vloss": "optimization/value_loss",
    "entropy": "optimization/entropy",
    "reward": "rewards/avg_total_reward",
    "attacks": "rewards/avg_attacks",
    "deaths": "rewards/avg_deaths",
    "garb_in": "gameplay/garbage_in_app",
    "scale": "optimization/return_scale",
}


def series(acc, tag):
    return np.array(
        [float(tf.make_ndarray(e.tensor_proto)) for e in acc.Tensors(tag)]
    )


def report(run_dir):
    acc = EventAccumulator(run_dir, size_guidance={"tensors": 0})
    acc.Reload()
    s = {k: series(acc, t) for k, t in TAGS.items()}
    n = len(s["reward"])
    print(f"\n=== {run_dir}  ({n} gens, scale={s['scale'][0]:.2f}) ===")
    print(
        f"{'gen':>4s} {'reward':>8s} {'attacks':>8s} {'deaths':>7s} {'entropy':>8s} "
        f"{'upd_kl':>8s} {'pol_kl':>7s} {'vloss':>7s} {'garb_in':>8s}"
    )
    for g in range(n):
        print(
            f"{g:4d} {s['reward'][g]:8.1f} {s['attacks'][g]:8.1f} {s['deaths'][g]:7.2f} "
            f"{s['entropy'][g]:8.3f} {s['update_kl'][g]:8.4f} {s['policy_kl'][g]:7.3f} "
            f"{s['vloss'][g]:7.3f} {s['garb_in'][g]:8.3f}"
        )
    d1 = s["update_kl"][1:15].mean() if n >= 15 else float("nan")
    d3 = s["deaths"][15:].mean() if n >= 16 else float("nan")
    d4 = s["reward"][18:].mean() if n >= 19 else float("nan")
    ent = s["entropy"]
    d5 = (ent[5:].max() - ent[:5].mean()) if n >= 6 else float("nan")
    print(
        f"D1 update_kl g1-14 = {d1:.4f} (healthy >=.030 / sick <=.025) | "
        f"D3 deaths g15+ = {d3:.2f} (<=0.5 / >=1.5) | "
        f"D4 reward g18+ = {d4:.1f} | D5 entropy rise vs g0-4 = {d5:+.3f} (spiral >=+0.3)"
    )


for r in sys.argv[1:]:
    report(r)
