#!/usr/bin/env python3
"""Materialize an exact released FFN-refinement mask without recomputing selection."""
import argparse

from cocurve.refined_artifacts import materialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--ratio", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(materialize(args.model_key, args.ratio, args.output))


if __name__ == "__main__":
    main()
