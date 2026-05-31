import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="pretrain")
    parser.add_argument("family", choices=["ar", "flat", "placement"])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to expert dataset (defaults: datasets/tetris_expert_dataset_b2b for ar; "
        "datasets/tetris_expert_dataset_flat for flat).",
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Per-step batch size. ar scores batch x --cand-topk sequences through "
        "the key decoder, so keep it modest (128 ~ 5.8GB at depth 64, K 32); flat "
        "has no such multiplier and can go higher (pass 256-512).",
    )
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="Train only the policy head; skip the value head.",
    )
    parser.add_argument(
        "--cand-topk",
        type=int,
        default=32,
        help="ar: number of top-scored candidate moves distilled per position "
        "(memory/compute lever).",
    )
    parser.add_argument(
        "--policy-temp",
        type=float,
        default=10.0,
        help="ar: temperature applied to candidate scores when forming the policy "
        "target weights. Scores span O(tens-thousands) (b2b-dependent); ~10 keeps "
        "the best move ~56%% of the target mass, higher flattens it.",
    )
    args = parser.parse_args()

    if args.family == "ar":
        from qtris.pretraining.ar import main as run
    elif args.family == "placement":
        from qtris.pretraining.placement import main as run
    else:
        from qtris.pretraining.flat import main as run
    run(args)


if __name__ == "__main__":
    main()
