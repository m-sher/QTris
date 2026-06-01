import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("family", choices=["ar", "flat", "placement"])
    parser.add_argument("--mode", choices=["single", "1v1"], default="single")
    parser.add_argument("--num-generations", type=int, default=1_000_000)
    parser.add_argument(
        "--expert-dataset",
        default=None,
        help="placement only: path to the BC dataset for the PPO expert anchor "
        "(default: datasets/tetris_oracle_placement; anchor is skipped if the path is absent).",
    )
    args = parser.parse_args()

    if args.mode == "1v1":
        if args.family == "placement":
            parser.error("placement 1v1 not supported; use `train placement` (single).")
        from qtris.training._1v1 import main as run
    elif args.family == "ar":
        from qtris.training.ar import main as run
    elif args.family == "placement":
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
