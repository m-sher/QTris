import argparse
from pathlib import Path

from qtris.config import PretrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="pretrain")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to expert dataset (default: datasets/tetris_oracle_placement).",
    )
    parser.add_argument(
        "--val-dataset",
        type=Path,
        default=None,
        help="Path to a SEPARATE, frozen held-out dataset for "
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
        "checkpoints/placement_pretrained_policy/ckpt-2335). Defaults to the save-dir "
        "latest. New checkpoints still save to the default dir.",
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Per-step batch size. The candidate set is scored in one pass (encoder "
        "stays O(batch)), so this can go high (pass 256-512).",
    )
    parser.add_argument(
        "--policy-temp",
        type=float,
        default=1.0,
        help="Temperature applied to candidate scores when forming the "
        "policy target weights. Scores are raw search magnitude O(hundreds-thousands); "
        "lower sharpens the target onto the search's best move, higher flattens it toward "
        "the full distribution.",
    )
    parser.add_argument(
        "--value-anchor",
        type=float,
        default=PretrainConfig().value_anchor_t,
        help="tanh output that the oracle score's 95th percentile maps "
        "to. The value label is tanh((score - median) / scale) with scale set from this, "
        "so lower leaves more headroom above the anchor and starts the pretrained head "
        "less confident. Must be in (0, 1).",
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=1.0,
        help="Weight on the value loss, which is normalized by the label "
        "variance (so it reads as fraction of variance unexplained and this weight stays "
        "comparable across recalibrations).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW decoupled weight decay (0.0 = off, ~ plain Adam). "
        "XLA-safe regularizer for the overfitting placement net; sweep ~1e-3..1e-2 "
        "watching the held-out val top1/top3.",
    )
    args = parser.parse_args()

    from qtris.pretraining.placement import main as run

    run(args)


if __name__ == "__main__":
    main()
