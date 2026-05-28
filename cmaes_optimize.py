"""CMA-ES optimizer for b2b_search weights, with W&B logging.

CMA-ES is the right tool for this shape of problem (~28 noisy continuous
dims, expensive evaluation): it learns parameter correlations into its
covariance matrix and converges much faster per evaluation than GP-Bayes
in this dim range.  W&B is used for experiment tracking only — the
optimization loop runs locally.

Objective (lexicographic, maximized):
    objective = 1e6 * survival_rate + 1e4 * APP + mean_max_b2b

Each tier strictly dominates the next: any 1/N change in survival beats
any plausible APP delta; 0.001 APP beats the full max_b2b range.

Usage:
    uv run python cmaes_optimize.py --popsize 20 --max-gens 25
"""

import argparse
import time

import cma
import numpy as np
import wandb

from TetrisEnv.CB2BSearch import CB2BSearch, GameConfig

# Default values come from the C source.  Bounds default to [0, 2*default]
# so the search can also deactivate weights (matches the ablation finding
# that several weights improved quality when removed).
DEFAULTS: dict[str, float] = {
    "W_NEAR_DEATH":      5000.0,
    "W_HEIGHT_QUARTIC":  80.0,
    "W_AVG_HEIGHT":      40.0,
    "W_BUMPINESS":       1.0,
    "W_HOLES":           6.0,
    "W_WASTED_HOLE":     3.0,
    "W_HOLE_CEILING":    1.5,
    "W_HOLE_FORGIVE":    1.5,
    "W_B2B_FLAT":        5.0,
    "W_B2B_SQRT":        8.0,
    "W_B2B_LINEAR":      20.0,
    "W_ATTACK_TOTAL":    1.2,
    "W_MAX_SINGLE":      0.5,
    "W_B2B_ATTACK":      1.5,
    "W_APP":             3.0,
    "W_GARBAGE_PREVENT": 4.0,
    "W_COMBO":           2.5,
    "W_DOWNSTACK":       2.0,
    "W_WELL_ALIGNED_9":  1.5,
    "W_CASCADE":         5.0,
    "W_SURGE_POT":       2.0,
    "W_BREAK_READY":     6.0,
    "W_TSLOT":           6.0,
    "W_IMMOBILE_CLEAR":  5.0,
    "W_IMMOBILE_LINES":  1.0,
    "W_CHAIN_ROLLOUT":   10.0,
    "W_FUTURE_B2B":      2.0,
    "W_TSPIN_MULTILINE": 3.0,
}
WEIGHT_NAMES: list[str] = list(DEFAULTS.keys())
N_DIMS = len(WEIGHT_NAMES)
HIGHS = np.array([2.0 * DEFAULTS[n] for n in WEIGHT_NAMES], dtype=float)
LOWS = np.zeros(N_DIMS, dtype=float)


def denormalize(x_norm: np.ndarray) -> np.ndarray:
    return LOWS + np.clip(x_norm, 0.0, 1.0) * (HIGHS - LOWS)


def evaluate(searcher: CB2BSearch, weights_real: np.ndarray, args, game_configs):
    searcher.reset_weights()
    for name, val in zip(WEIGHT_NAMES, weights_real):
        searcher.set_weight(name, float(val))

    t0 = time.time()
    results = searcher.run_eval_games(
        game_configs,
        num_steps=args.steps,
        search_depth=args.depth,
        beam_width=args.width,
        queue_size=args.queue_size,
    )
    elapsed = time.time() - t0

    n = len(results)
    survived = sum(r.survived for r in results) / n
    pieces = sum(r.steps_completed for r in results)
    attack = sum(r.total_attack for r in results)
    app = (attack / pieces) if pieces > 0 else 0.0
    max_b2b = sum(r.max_b2b for r in results) / n
    max_h = sum(r.max_height for r in results) / n

    objective = 1_000_000.0 * survived + 10_000.0 * app + max_b2b
    return {
        "objective": objective,
        "survival": survived,
        "app": app,
        "mean_max_b2b": max_b2b,
        "mean_max_height": max_h,
        "mean_pieces": pieces / n,
        "search_time_sec": elapsed,
    }


def cma_internals(es: cma.CMAEvolutionStrategy) -> dict:
    out = {
        "cmaes/sigma": float(es.sigma),
        "cmaes/generation": int(es.countiter),
    }
    try:
        eigvals = np.linalg.eigvalsh(es.C)
        eigvals = np.clip(eigvals, 1e-20, None)
        cond = float(eigvals.max() / eigvals.min())
        out["cmaes/condition_number"] = cond
        out["cmaes/axis_ratio"] = float(np.sqrt(cond))
        out["cmaes/min_eigval"] = float(eigvals.min())
        out["cmaes/max_eigval"] = float(eigvals.max())
    except Exception:
        pass
    try:
        out["cmaes/mean_norm"] = float(np.linalg.norm(es.mean))
    except Exception:
        pass
    return out


def signed_correlations(history: list[tuple[np.ndarray, dict]]) -> np.ndarray:
    """Return Pearson correlation per weight against objective. Sign = direction."""
    if len(history) < 3:
        return np.zeros(N_DIMS)
    X = np.array([w for w, _ in history])
    y = np.array([m["objective"] for _, m in history])
    out = np.zeros(N_DIMS)
    if y.std() < 1e-12:
        return out
    for j in range(N_DIMS):
        xj = X[:, j]
        if xj.std() < 1e-12:
            out[j] = 0.0
        else:
            out[j] = float(np.corrcoef(xj, y)[0, 1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=12,
                        help="Eval games per fitness evaluation")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--width", type=int, default=76)
    parser.add_argument("--queue-size", type=int, default=5)
    parser.add_argument("--garbage-chance", type=float, default=0.15)
    parser.add_argument("--garbage-min", type=int, default=1)
    parser.add_argument("--garbage-max", type=int, default=4)
    parser.add_argument("--garbage-push-delay", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=2000,
                        help="All evaluations use the same seed set so fitness comparisons are noise-controlled")
    parser.add_argument("--popsize", type=int, default=20,
                        help="CMA-ES population per generation. Default lambda for N=28 is 14; "
                             "we enlarge to 20 because the fitness is noisy.")
    parser.add_argument("--sigma0", type=float, default=0.3,
                        help="Initial step size in normalized [0,1] space (~30%% of bound width)")
    parser.add_argument("--max-gens", type=int, default=25)
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Hard cap on total evaluations (overrides max-gens if hit first)")
    parser.add_argument("--project", type=str, default="b2b-cmaes")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    config_dump = vars(args).copy()
    config_dump["weight_names"] = WEIGHT_NAMES
    config_dump["weight_lows"] = LOWS.tolist()
    config_dump["weight_highs"] = HIGHS.tolist()
    wandb.init(project=args.project, name=args.name, config=config_dump)

    searcher = CB2BSearch()
    game_configs = [
        GameConfig(
            seed=args.seed_base + i,
            garbage_chance=args.garbage_chance,
            garbage_min=args.garbage_min,
            garbage_max=args.garbage_max,
            garbage_push_delay=args.garbage_push_delay,
        )
        for i in range(args.games)
    ]

    # Start CMA-ES from the existing default weights (centered in normalized space).
    x0 = np.full(N_DIMS, 0.5, dtype=float)
    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        args.sigma0,
        {
            "popsize": args.popsize,
            "bounds": [[0.0] * N_DIMS, [1.0] * N_DIMS],
            "verbose": -9,
            "verb_log": 0,
        },
    )

    history: list[tuple[np.ndarray, dict]] = []
    eval_count = 0
    best_obj = -np.inf
    best_weights = denormalize(x0)
    best_metrics: dict | None = None

    eval_table_columns = [
        "generation", "candidate_idx", "global_step",
        "objective", "survival", "app", "mean_max_b2b",
        "mean_max_height", "mean_pieces", "search_time_sec",
    ] + WEIGHT_NAMES

    while es.countiter < args.max_gens and not es.stop():
        if args.max_evals is not None and eval_count >= args.max_evals:
            break

        gen = es.countiter + 1
        xs_norm = es.ask()
        gen_metrics: list[dict] = []
        gen_weights: list[np.ndarray] = []

        for i, x in enumerate(xs_norm):
            x_arr = np.asarray(x, dtype=float)
            weights_real = denormalize(x_arr)
            m = evaluate(searcher, weights_real, args, game_configs)
            eval_count += 1
            gen_metrics.append(m)
            gen_weights.append(weights_real)
            history.append((weights_real.copy(), m))

            if m["objective"] > best_obj:
                best_obj = m["objective"]
                best_weights = weights_real.copy()
                best_metrics = m

            wandb.log({
                "eval/objective":       m["objective"],
                "eval/survival":        m["survival"],
                "eval/app":             m["app"],
                "eval/mean_max_b2b":    m["mean_max_b2b"],
                "eval/mean_max_height": m["mean_max_height"],
                "eval/search_time_sec": m["search_time_sec"],
                "eval/generation":      gen,
                "eval/candidate_idx":   i,
            }, step=eval_count)

            if args.max_evals is not None and eval_count >= args.max_evals:
                break

        es.tell(xs_norm[: len(gen_metrics)], [-m["objective"] for m in gen_metrics])

        objs = np.array([m["objective"]    for m in gen_metrics])
        apps = np.array([m["app"]          for m in gen_metrics])
        b2bs = np.array([m["mean_max_b2b"] for m in gen_metrics])
        survs = np.array([m["survival"]    for m in gen_metrics])
        times = np.array([m["search_time_sec"] for m in gen_metrics])

        gen_log: dict = {
            "gen/best_objective":    float(objs.max()),
            "gen/mean_objective":    float(objs.mean()),
            "gen/median_objective":  float(np.median(objs)),
            "gen/std_objective":     float(objs.std()),
            "gen/best_app":          float(apps.max()),
            "gen/mean_app":          float(apps.mean()),
            "gen/best_max_b2b":      float(b2bs.max()),
            "gen/mean_max_b2b":      float(b2bs.mean()),
            "gen/best_survival":     float(survs.max()),
            "gen/mean_survival":     float(survs.mean()),
            "gen/mean_eval_time":    float(times.mean()),
            "best_so_far/objective": best_obj,
            "best_so_far/survival":  best_metrics["survival"]     if best_metrics else 0.0,
            "best_so_far/app":       best_metrics["app"]          if best_metrics else 0.0,
            "best_so_far/max_b2b":   best_metrics["mean_max_b2b"] if best_metrics else 0.0,
            "cmaes/evaluations":     eval_count,
        }
        gen_log.update(cma_internals(es))

        # Per-weight signed correlation against objective.
        corrs = signed_correlations(history)
        corr_table = wandb.Table(columns=[
            "weight", "correlation_with_objective", "abs_correlation",
            "best_value", "default", "delta_from_default",
        ])
        for name, corr, best_val in zip(WEIGHT_NAMES, corrs, best_weights):
            corr_table.add_data(
                name, float(corr), float(abs(corr)),
                float(best_val), DEFAULTS[name], float(best_val) - DEFAULTS[name],
            )
        wandb.log({"weight_correlations": corr_table}, step=eval_count)

        # Best weights at the end of this generation, scalar-logged so they
        # show up as W&B line charts (one per weight).
        for name, val in zip(WEIGHT_NAMES, best_weights):
            gen_log[f"best_weight/{name}"] = float(val)

        wandb.log(gen_log, step=eval_count)

        print(
            f"[gen {gen:3d}] best_obj={objs.max():12.1f}  "
            f"mean_obj={objs.mean():12.1f}  "
            f"sigma={es.sigma:.4f}  evals={eval_count}",
            flush=True,
        )

    # ── Final tables ──────────────────────────────────────────────────
    eval_table = wandb.Table(columns=eval_table_columns)
    for k, (w, m) in enumerate(history):
        # k is 0-indexed; reconstruct gen/idx via popsize for clarity
        gen_k = (k // args.popsize) + 1
        idx_k = k % args.popsize
        eval_table.add_data(
            gen_k, idx_k, k + 1,
            m["objective"], m["survival"], m["app"], m["mean_max_b2b"],
            m["mean_max_height"], m["mean_pieces"], m["search_time_sec"],
            *[float(v) for v in w],
        )
    wandb.log({"eval_history": eval_table})

    best_table = wandb.Table(columns=["weight", "best_value", "default", "delta", "ratio"])
    for name, val in zip(WEIGHT_NAMES, best_weights):
        d = DEFAULTS[name]
        best_table.add_data(name, float(val), d, float(val) - d,
                            float(val) / d if d != 0 else float("nan"))
    wandb.log({"best_weights": best_table})

    final_corr = signed_correlations(history)
    final_corr_table = wandb.Table(columns=[
        "weight", "correlation_with_objective", "abs_correlation",
        "best_value", "default",
    ])
    for name, corr, best_val in zip(WEIGHT_NAMES, final_corr, best_weights):
        final_corr_table.add_data(name, float(corr), float(abs(corr)),
                                  float(best_val), DEFAULTS[name])
    wandb.log({"final_weight_correlations": final_corr_table})

    if best_metrics:
        wandb.summary.update({
            "final/best_objective":    best_obj,
            "final/best_survival":     best_metrics["survival"],
            "final/best_app":          best_metrics["app"],
            "final/best_max_b2b":      best_metrics["mean_max_b2b"],
            "final/total_evaluations": eval_count,
            "final/total_generations": int(es.countiter),
        })
        for name, val in zip(WEIGHT_NAMES, best_weights):
            wandb.summary[f"final/{name}"] = float(val)

    print("\nBest weights found:")
    for name, val in zip(WEIGHT_NAMES, best_weights):
        d = DEFAULTS[name]
        print(f"  {name:<22} {val:10.3f}  (default {d:8.2f}, delta {val - d:+8.2f})")
    print(f"\nBest objective: {best_obj:.1f}")
    if best_metrics:
        print(f"  survival={best_metrics['survival']:.3f}  "
              f"APP={best_metrics['app']:.3f}  "
              f"max_b2b={best_metrics['mean_max_b2b']:.2f}")

    wandb.finish()


if __name__ == "__main__":
    main()
