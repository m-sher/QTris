"""Entrypoint for Phase 2 training (solo + random garbage scaffolding).

Run from the QTris/ directory.  TFTetrisEnv must be installed (pip install
../TFTetrisEnv) so `from TetrisEnv...` resolves.  Examples:

  # All defaults — solo, garbage pressure, WandB on, project=Tetris-Alpha
  python -m Alpha.train

  # Override common knobs
  python -m Alpha.train --gens 200 --games-per-gen 64 --sims 200 --batch 512

  # Resume-style: name the run, tweak garbage
  python -m Alpha.train --run-name solo-v0.1 --garbage-chance 0.2

  # No-WandB (e.g. for smoke tests or air-gapped runs)
  python -m Alpha.train --no-wandb --gens 3 --games-per-gen 4

The defaults are conservative and machine-agnostic; tune to your training
machine's GPU memory + CPU count.
"""

from __future__ import annotations
import argparse
import dataclasses
import sys

from .trainer import Phase2Config, run_phase2


def _build_config(argv: list) -> Phase2Config:
    p = argparse.ArgumentParser(
        description="Run AlphaZero-style training (Phase 2: solo + scaffolding).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core schedule
    p.add_argument("--gens", "--num-generations", type=int, default=None,
                   dest="num_generations", help="Number of generations.")
    p.add_argument("--games-per-gen", type=int, default=None,
                   dest="games_per_generation",
                   help="Self-play games per generation.")
    p.add_argument("--train-steps", type=int, default=None,
                   dest="train_steps_per_generation",
                   help="Gradient steps per generation.")
    p.add_argument("--batch", "--batch-size", type=int, default=None,
                   dest="batch_size", help="Mini-batch size.")
    p.add_argument("--sims", "--num-simulations", type=int, default=None,
                   dest="num_simulations", help="MCTS sims per move.")
    p.add_argument("--steps-per-game", type=int, default=None,
                   dest="num_steps_per_game",
                   help="Max placements per self-play game.")
    p.add_argument("--buffer-cap", type=int, default=None,
                   dest="buffer_capacity",
                   help="Replay buffer capacity (transitions).")
    p.add_argument("--anneal-gens", type=int, default=None,
                   dest="anneal_generations",
                   help="Linear oracle-weight anneal length (gens).")
    p.add_argument("--lr", type=float, default=None,
                   dest="learning_rate")
    p.add_argument("--weight-decay", type=float, default=None,
                   dest="weight_decay")
    p.add_argument("--seed", type=int, default=None)

    # Network
    p.add_argument("--depth", type=int, default=None, dest="model_depth")
    p.add_argument("--heads", type=int, default=None, dest="model_num_heads")
    p.add_argument("--layers", type=int, default=None, dest="model_num_layers")
    p.add_argument("--dropout", type=float, default=None, dest="model_dropout")

    # Garbage pressure
    p.add_argument("--garbage-chance", type=float, default=None)
    p.add_argument("--garbage-min", type=int, default=None)
    p.add_argument("--garbage-max", type=int, default=None)

    # WandB
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable WandB logging (overrides --use-wandb).")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--run-name", "--wandb-run-name", type=str, default=None,
                   dest="wandb_run_name")

    # Warm start from autoregressive checkpoint
    p.add_argument("--warm-start", type=str, default=None,
                   dest="warm_start_ckpt",
                   help="Path to a tf.train.CheckpointManager directory "
                        "for QTris/Autoregressive/TetrisModel.PolicyModel. "
                        "Encoder weights will be copied; heads start fresh. "
                        "Geometry must match (depth/num_layers).")
    p.add_argument("--warm-start-piece-dim", type=int, default=None,
                   dest="warm_start_piece_dim")
    p.add_argument("--warm-start-dropout", type=float, default=None,
                   dest="warm_start_dropout")

    args = p.parse_args(argv)

    cfg = Phase2Config()
    for f in dataclasses.fields(cfg):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    if args.no_wandb:
        cfg.use_wandb = False
    return cfg


def main(argv=None) -> int:
    cfg = _build_config(argv if argv is not None else sys.argv[1:])

    print(f"[phase2] starting — {cfg.num_generations} gens × "
          f"{cfg.games_per_generation} games/gen × "
          f"{cfg.num_steps_per_game} steps × {cfg.num_simulations} sims",
          flush=True)
    print(f"[phase2] anneal oracle 1.0 → 0.0 over {cfg.anneal_generations} gens",
          flush=True)
    print(f"[phase2] garbage chance={cfg.garbage_chance} "
          f"rows=[{cfg.garbage_min}, {cfg.garbage_max}]", flush=True)
    if cfg.use_wandb:
        print(f"[phase2] WandB project={cfg.wandb_project} "
              f"run={cfg.wandb_run_name or '<auto>'}", flush=True)
    else:
        print("[phase2] WandB disabled (console logging only)", flush=True)
    if cfg.warm_start_ckpt:
        print(f"[phase2] warm-start from {cfg.warm_start_ckpt}", flush=True)

    run_phase2(cfg)
    print("[phase2] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
