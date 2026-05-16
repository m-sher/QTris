import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("family", choices=["ar", "flat", "vs"])
    parser.add_argument("--mode", choices=["single", "1v1"], default="single")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Required for `demo ar` / `demo flat` / `demo ar --mode 1v1`. "
                             "Path to a tf.train.CheckpointManager directory.")
    parser.add_argument("--opponent", type=Path, default=None,
                        help="Required for `demo <family> --mode 1v1`. Opponent's checkpoint directory.")
    parser.add_argument("--left", type=Path, default=None,
                        help="Required for `demo vs`. Left player's checkpoint directory.")
    parser.add_argument("--right", type=Path, default=None,
                        help="Required for `demo vs`. Right player's checkpoint directory.")
    args = parser.parse_args()

    if args.family == "vs":
        if args.left is None or args.right is None:
            parser.error("`demo vs` requires --left and --right.")
        from qtris.demo.vs import main as run
    elif args.mode == "1v1":
        if args.checkpoint is None or args.opponent is None:
            parser.error("`demo <family> --mode 1v1` requires --checkpoint and --opponent.")
        if args.family == "flat":
            parser.error("flat 1v1 demo not yet implemented; only `demo ar --mode 1v1` is supported.")
        from qtris.demo.ar_1v1 import main as run
    else:
        if args.checkpoint is None:
            parser.error(f"`demo {args.family}` requires --checkpoint.")
        if args.family == "ar":
            from qtris.demo.ar import main as run
        else:
            from qtris.demo.flat import main as run
    run(args)


if __name__ == "__main__":
    main()
