import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("family", choices=["ar", "flat"])
    parser.add_argument("--mode", choices=["single", "1v1"], default="single")
    parser.add_argument("--num-generations", type=int, default=1000)
    args = parser.parse_args()

    print(f"[stub] would run train {args.family} with args={vars(args)}")


if __name__ == "__main__":
    main()
