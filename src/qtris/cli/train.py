import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("--mode", choices=["single", "1v1"], default="single")
    parser.add_argument("--num-generations", type=int, default=1_000_000)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="mirror the TensorBoard run to wandb (sync_tensorboard).",
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=16,
        help="concurrent self-play games (net batch size).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=32,
        help="moves collected per game per generation.",
    )
    parser.add_argument(
        "--max-game-steps",
        type=int,
        default=512,
        help="1v1 only: hard per-game placement cap; a game still alive at the "
        "cap is flushed as a draw.",
    )
    parser.add_argument(
        "--max-pool-size",
        type=int,
        default=30,
        help="1v1 only: max opponent-pool snapshots kept on disk (gen_0 pinned).",
    )
    parser.add_argument(
        "--pool-interval",
        type=int,
        default=10,
        help="1v1 only: generations between gated opponent-pool snapshots.",
    )
    parser.add_argument(
        "--pool-wr-gate",
        type=float,
        default=0.55,
        help="1v1 only: decisive-WR EMA the learner must beat to add a snapshot.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="1v1 only: generations between win_rate_vs_ref evals (vs frozen gen_0).",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=32,
        help="1v1 only: games per win_rate_vs_ref eval.",
    )
    parser.add_argument(
        "--td-lambda",
        type=float,
        default=0.9,
        help="1v1 only: TD(lambda) for the value target (1=raw outcome z on "
        "every position; lower bootstraps toward near-term root value).",
    )
    parser.add_argument(
        "--num-simulations", type=int, default=64, help="MCTS simulations per move."
    )
    parser.add_argument(
        "--c-puct", type=float, default=1.5, help="PUCT exploration constant."
    )
    parser.add_argument(
        "--leaves-per-round",
        type=int,
        default=4,
        help="intra-tree leaf batching: leaves collected per tree per net call (virtual "
        "loss). Higher = fewer net calls (~L x faster) but more search distortion; 1 = "
        "sequential. Default 4.",
    )
    parser.add_argument(
        "--vloss",
        type=float,
        default=1.0,
        help="virtual-loss magnitude (scaled-Q units).",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="root Dirichlet noise concentration.",
    )
    parser.add_argument(
        "--dirichlet-eps",
        type=float,
        default=0.25,
        help="root Dirichlet noise mix weight.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="single-player only: discount for MCTS backup + MC return target "
        "(default 0.99). Rejected for 1v1, whose reward is terminal-only.",
    )
    parser.add_argument(
        "--temp-moves",
        type=int,
        default=12,
        help="opening moves sampled at temperature 1 before greedy.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="training minibatch size."
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="optimization epochs over each self-play buffer.",
    )
    parser.add_argument(
        "--value-coef",
        type=float,
        default=1.0,
        help="value-loss weight in the AZ loss.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Adam learning rate."
    )
    parser.add_argument(
        "--w-death",
        type=float,
        default=100.0,
        help="terminal-edge death penalty (raw attack units; also the realized death reward).",
    )
    parser.add_argument(
        "--q-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="1v1 only: rank PUCT on per-tree min-max normalised Q (MuZero style); "
        "--no-q-norm ranks on raw return units.",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=8_000,
        help="max positions kept in the multi-generation replay buffer.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=1.0,
        help="lambda for the value-target return (1.0 = MC return + horizon bootstrap).",
    )
    parser.add_argument(
        "--garbage-traces",
        type=str,
        default=None,
        help="trace-replay garbage: dir of tier subdirs of .npy attack streams "
        "(sorted tier name = difficulty). Replaces the chance sweep when set.",
    )
    parser.add_argument(
        "--trace-free-envs",
        type=int,
        default=2,
        help="number of garbage-free envs when --garbage-traces is set.",
    )
    parser.add_argument(
        "--trace-harvest-cap",
        type=int,
        default=256,
        help="max files kept in the rolling 99_recent harvest tier.",
    )
    parser.add_argument(
        "--return-scale",
        type=float,
        default=None,
        help="force the frozen return_scale (skips the warm-start seeding rollout; "
        "ignored when resuming a checkpoint).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/placement_az",
        help="AZ checkpoint directory (a non-empty dir silently resumes).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="suffix for the TensorBoard run dir.",
    )
    parser.add_argument(
        "--save-states",
        default=None,
        help="1v1 only: dir to dump per-generation state shards (both players) for "
        "offline oracle relabeling via `datagen --label-states`.",
    )
    parser.add_argument(
        "--no-harvest",
        action="store_true",
        help="trace mode: don't write this run's attacks into 99_recent and don't "
        "rescan the library (static pools for the whole run).",
    )
    parser.add_argument(
        "--trace-tiers",
        default=None,
        help="comma-separated tier subdirs to load from --garbage-traces (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the RNG (Dirichlet noise, temperature sampling).",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="trace mode: feedback difficulty curriculum (ramp trace tiers toward a deaths "
        "deadband) instead of the fixed even tier split.",
    )
    parser.add_argument(
        "--curriculum-start",
        type=float,
        default=0.0,
        help="--curriculum: starting difficulty index (0 = weakest tier; pass the last value "
        "when resuming a chained run).",
    )
    parser.add_argument(
        "--garbage-chance-min",
        type=float,
        default=0.0,
        help="per-step garbage probability of game 0 (swept up to max).",
    )
    parser.add_argument(
        "--garbage-chance-max",
        type=float,
        default=0.2,
        help="per-step garbage probability of the last game.",
    )
    parser.add_argument(
        "--garbage-rows-min", type=int, default=1, help="min garbage rows per spawn."
    )
    parser.add_argument(
        "--garbage-rows-max", type=int, default=4, help="max garbage rows per spawn."
    )

    args = parser.parse_args()

    if args.mode == "1v1":
        if args.gamma is not None:
            parser.error(
                "1v1 does not accept --gamma: its reward is terminal-only "
                "(z in {-1,0,1}) and the TD(lambda) target is undiscounted. Use "
                "--td-lambda to trade outcome grounding against bootstrap."
            )
        from qtris.training._1v1_placement_az import main as run
    else:
        from qtris.training.placement_az import main as run

    import sys
    import tf_agents

    tf_agents.system.multiprocessing.handle_main(
        lambda _argv: run(args),
        argv=[sys.argv[0]],
    )


if __name__ == "__main__":
    main()
