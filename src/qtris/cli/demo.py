import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("family", choices=["ar", "flat", "vs"])
    parser.add_argument("--mode", choices=["single", "1v1"], default="single")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--opponent", type=Path)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    args = parser.parse_args()

    print(f"[stub] would run demo {args.family} with args={vars(args)}")


if __name__ == "__main__":
    main()
