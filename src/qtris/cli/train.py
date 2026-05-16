import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("family", choices=["ar", "flat"])
    parser.add_argument("--mode", choices=["single", "1v1"], default="single")
    parser.add_argument("--num-generations", type=int, default=1_000_000)
    args = parser.parse_args()

    if args.mode == "1v1":
        from qtris.training._1v1 import main as run
    elif args.family == "ar":
        from qtris.training.ar import main as run
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
