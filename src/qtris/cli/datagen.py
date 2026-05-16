import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="datagen")
    parser.add_argument("family", choices=["ar", "flat"])
    parser.add_argument("--dagger", action="store_true",
                        help="Use DAgger collection (policy rollout + beam relabel) instead of pure beam-search.")
    parser.add_argument("--policy-checkpoint", type=Path, default=None,
                        help="Required for --dagger. Path to PolicyModel checkpoint directory to roll out.")
    parser.add_argument("--steps", type=int, default=200_000,
                        help="Number of env steps to collect.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output dataset path (defaults to datasets/tetris_expert_dataset_b2b for ar; "
                             "datasets/tetris_expert_dataset_flat for flat).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed offset. Default: 0 for non-dagger; 10_000_000 for --dagger.")
    args = parser.parse_args()

    if args.dagger:
        if args.policy_checkpoint is None:
            parser.error("--dagger requires --policy-checkpoint.")
        if args.seed is None:
            args.seed = 10_000_000  # match DAgger's old default (far from beam seeds)
        from qtris.data.dagger import main as run
    elif args.family == "ar":
        if args.seed is None:
            args.seed = 0
        from qtris.data.gen_ar import main as run
    else:
        if args.seed is None:
            args.seed = 0
        from qtris.data.gen_flat import main as run
    run(args)


if __name__ == "__main__":
    main()
