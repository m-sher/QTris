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
        "datasets/tetris_expert_dataset_flat for flat; "
        "datasets/tetris_oracle_placement for placement).",
    )
    parser.add_argument(
        "--val-dataset",
        type=Path,
        default=None,
        help="placement only: path to a SEPARATE, frozen held-out dataset for "
        "validation top1/top3 (collect it once to its own path and NEVER merge it into "
        "--dataset). An in-dataset split is not valid here because warm-started runs "
        "have already trained on the whole training file. Omit to skip val reporting.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint to resume/warm-start the policy from: a checkpoint directory "
        "(its latest is used) or a specific ckpt prefix (e.g. "
        "checkpoints/placement_pretrained_policy/ckpt-2335). Defaults to the family's "
        "own save-dir latest. New checkpoints still save to the family's default dir; "
        "for ar/flat this targets the policy checkpoint (value resumes from its own dir).",
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Per-step batch size. ar scores batch x --cand-topk sequences through "
        "the key decoder, so keep it modest (128 ~ 5.8GB at depth 64, K 32); flat and "
        "placement score the candidate set in one cheap pass (encoder stays O(batch)) "
        "and can go higher (pass 256-512).",
    )
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="ar/flat: train only the policy head, skip the separate value model "
        "(no effect for placement, which trains its shared policy+value net jointly).",
    )
    parser.add_argument(
        "--cand-topk",
        type=int,
        default=32,
        help="ar only (flat/placement score every candidate, so it does not apply): "
        "number of top-scored candidate moves distilled per position (memory/compute lever).",
    )
    parser.add_argument(
        "--policy-temp",
        type=float,
        default=1.0,
        help="ar/placement: temperature applied to candidate scores when forming the "
        "policy target weights. Scores are raw search magnitude O(hundreds-thousands); "
        "lower sharpens the target onto the search's best move, higher flattens it toward "
        "the full distribution.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="placement only: AdamW decoupled weight decay (0.0 = off, ~ plain Adam). "
        "XLA-safe regularizer for the overfitting placement net; sweep ~1e-3..1e-2 "
        "watching the held-out val top1/top3.",
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
