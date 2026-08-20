import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("--mode", choices=["single", "1v1"], default="single")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a tf.train.CheckpointManager directory.",
    )
    parser.add_argument(
        "--opponent",
        type=Path,
        default=None,
        help="Required for `--mode 1v1`. Opponent's checkpoint directory.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="pick moves with the neural-guided beam search (policy prior + value "
        "leaf) instead of greedy top-1.",
    )
    parser.add_argument(
        "--depth", type=int, default=1, help="--search: lookahead plies."
    )
    parser.add_argument("--beam", type=int, default=8, help="--search: beam width.")
    parser.add_argument(
        "--gate",
        type=int,
        default=8,
        help="--search: top-K candidates expanded per node.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=0,
        help="play with PUCT MCTS (net policy priors + value leaves) "
        "at this simulation budget, greedy by visit count - the AlphaZero way to play "
        "an AZ checkpoint. 0 = off (use greedy top-1 or --search).",
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
        "--garbage-chance",
        type=float,
        default=0.15,
        help="per-step garbage chance for the chance model "
        "(0 = no garbage at all; ignored when --garbage-traces is set).",
    )
    parser.add_argument(
        "--garbage-traces",
        type=str,
        default=None,
        help="trace-replay garbage from this library dir "
        "(replaces the chance model; see train --garbage-traces).",
    )
    parser.add_argument(
        "--trace-tier",
        type=str,
        default=None,
        help="--garbage-traces: tier subdir to draw from "
        "(default: last sorted = strongest).",
    )
    parser.add_argument(
        "--max-game-steps",
        type=int,
        default=500,
        help="--mode 1v1: max moves before the duel is called a draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="--mode 1v1: RNG seed (piece sequence + garbage columns).",
    )
    args = parser.parse_args()

    if getattr(args, "search", False) and getattr(args, "num_simulations", 0) > 0:
        parser.error("use either --search or --num-simulations, not both.")

    if args.mode == "1v1":
        if args.checkpoint is None or args.opponent is None:
            parser.error("`demo --mode 1v1` requires --checkpoint and --opponent.")
        from qtris.demo.placement_1v1 import main as run
    else:
        if args.checkpoint is None:
            parser.error("`demo` requires --checkpoint.")
        from qtris.demo.placement import main as run
    run(args)


if __name__ == "__main__":
    main()
