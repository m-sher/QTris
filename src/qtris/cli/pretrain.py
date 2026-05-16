import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="pretrain")
    parser.add_argument("family", choices=["ar", "flat"])
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to expert dataset (defaults: datasets/tetris_expert_dataset_b2b for ar; "
                             "datasets/tetris_expert_dataset_flat for flat).")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Per-step batch size. Default 512 for ar; pass 256 for flat to match prior runs.")
    parser.add_argument("--policy-only", action="store_true",
                        help="Train only the policy head; skip the value head.")
    args = parser.parse_args()

    if args.family == "ar":
        from qtris.pretraining.ar import main as run
    else:
        from qtris.pretraining.flat import main as run
    run(args)


if __name__ == "__main__":
    main()
