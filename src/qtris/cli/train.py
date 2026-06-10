import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("family", choices=["ar", "flat", "placement"])
    parser.add_argument("--mode", choices=["single", "1v1"], default="single")
    parser.add_argument(
        "--algo",
        choices=["ppo", "az"],
        default="ppo",
        help="placement single-player only: 'ppo' (default) or 'az' (AlphaZero MCTS "
        "self-play).",
    )
    parser.add_argument("--num-generations", type=int, default=1_000_000)
    parser.add_argument(
        "--expert-dataset",
        default=None,
        help="placement ppo only: path to the BC dataset for the PPO expert anchor "
        "(omit to train plain PPO with no expert anchor).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="mirror the TensorBoard run to wandb (sync_tensorboard).",
    )

    az = parser.add_argument_group(
        "AlphaZero (placement --algo az)", "Knobs for the MCTS self-play trainer."
    )
    az.add_argument(
        "--num-games",
        type=int,
        default=16,
        help="concurrent self-play games (net batch size).",
    )
    az.add_argument(
        "--horizon",
        type=int,
        default=32,
        help="moves collected per game per generation.",
    )
    az.add_argument(
        "--num-simulations", type=int, default=64, help="MCTS simulations per move."
    )
    az.add_argument(
        "--leaves-per-round",
        type=int,
        default=4,
        help="intra-tree leaf batching: leaves collected per tree per net call (virtual "
        "loss). Higher = fewer net calls (~L x faster) but more search distortion; 1 = "
        "sequential. Default 4.",
    )
    az.add_argument(
        "--vloss",
        type=float,
        default=1.0,
        help="virtual-loss magnitude (scaled-Q units).",
    )
    az.add_argument(
        "--c-puct", type=float, default=1.5, help="PUCT exploration constant."
    )
    az.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="root Dirichlet noise concentration.",
    )
    az.add_argument(
        "--dirichlet-eps",
        type=float,
        default=0.25,
        help="root Dirichlet noise mix weight.",
    )
    az.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount for MCTS backup + MC return target.",
    )
    az.add_argument(
        "--temp-moves",
        type=int,
        default=12,
        help="opening moves sampled at temperature 1 before greedy.",
    )
    az.add_argument(
        "--mini-batch-size", type=int, default=256, help="training minibatch size."
    )
    az.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="optimization epochs over each self-play buffer.",
    )
    az.add_argument(
        "--value-coef",
        type=float,
        default=1.0,
        help="value-loss weight in the AZ loss.",
    )
    az.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Adam learning rate."
    )
    az.add_argument(
        "--w-attack",
        type=float,
        default=1.0,
        help="per-edge search reward weight on attack (also scales the realized return).",
    )
    az.add_argument(
        "--w-b2b",
        type=float,
        default=1.0,
        help="search leaf-bootstrap weight on max(0, b2b). 0 lets the learned value carry "
        "b2b's worth instead of the unbounded hoard crutch.",
    )
    az.add_argument(
        "--w-death",
        type=float,
        default=100.0,
        help="terminal-edge death penalty (raw attack units; also the realized death reward).",
    )
    az.add_argument(
        "--replay-capacity",
        type=int,
        default=25_000,
        help="max positions kept in the multi-generation replay buffer.",
    )
    az.add_argument(
        "--gae-lambda",
        type=float,
        default=1.0,
        help="lambda for the value-target return (1.0 = MC return + horizon bootstrap).",
    )
    az.add_argument(
        "--garbage-chance-min",
        type=float,
        default=0.0,
        help="per-step garbage probability of game 0 (swept up to max).",
    )
    az.add_argument(
        "--garbage-chance-max",
        type=float,
        default=0.2,
        help="per-step garbage probability of the last game.",
    )
    az.add_argument(
        "--garbage-rows-min", type=int, default=1, help="min garbage rows per spawn."
    )
    az.add_argument(
        "--garbage-rows-max", type=int, default=4, help="max garbage rows per spawn."
    )

    args = parser.parse_args()

    if args.algo == "az" and (args.family != "placement" or args.mode != "single"):
        parser.error("--algo az is only supported for `train placement` (single mode).")

    if args.mode == "1v1":
        if args.family == "placement":
            parser.error("placement 1v1 not supported; use `train placement` (single).")
        from qtris.training._1v1 import main as run
    elif args.family == "ar":
        from qtris.training.ar import main as run
    elif args.family == "placement":
        if args.algo == "az":
            from qtris.training.placement_az import main as run
        else:
            from qtris.training.placement import main as run
    else:
        from qtris.training.flat import main as run

    import sys
    import tf_agents

    tf_agents.system.multiprocessing.handle_main(
        lambda _argv: run(args),
        argv=[sys.argv[0]],
    )


if __name__ == "__main__":
    main()
