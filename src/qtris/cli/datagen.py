import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="datagen")
    parser.add_argument("family", choices=["ar", "flat"])
    parser.add_argument("--dagger", action="store_true")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    print(f"[stub] would run datagen {args.family} with args={vars(args)}")


if __name__ == "__main__":
    main()
