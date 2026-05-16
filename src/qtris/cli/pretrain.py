import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="pretrain")
    parser.add_argument("family", choices=["ar", "flat"])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--policy-only", action="store_true")
    args = parser.parse_args()

    print(f"[stub] would run pretrain {args.family} with args={vars(args)}")


if __name__ == "__main__":
    main()
