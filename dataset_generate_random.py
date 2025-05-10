import numpy as np
from numpy.random import Generator, PCG64


import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate random integers and save to a file.")
    parser.add_argument("--num", type=int, required=True, help="Number of integers to generate")
    parser.add_argument("--min", type=int, required=True, help="Minimum integer value (inclusive)")
    parser.add_argument("--max", type=int, required=True, help="Maximum integer value (inclusive)")
    parser.add_argument("--outfile", type=str, required=True, help="Output file to save the integers")
    parser.add_argument("--seed", type=int, required=False, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = Generator(PCG64(args.seed))
    random_integers = rng.integers(low=args.min, high=args.max + 1, size=args.num)

    # Save to file, all numbers appended together with no separators
    with open(args.outfile, 'w') as f:
        f.write(''.join(str(num) for num in random_integers))
    print(f"Saved {args.num} random integers to {args.outfile}")


if __name__ == "__main__":
    main()
