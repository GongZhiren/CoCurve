#!/usr/bin/env python3

import argparse

from cocurve.pipeline import CoCurvePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all single-unit ablations.")
    parser.add_argument("--config", required=True, help="Path to default config yaml")
    parser.add_argument("--model-key", default=None, help="Optional model key override")
    parser.add_argument("--run-dir", default=None, help="Resume or write to a fixed run directory")
    args = parser.parse_args()

    pipeline = CoCurvePipeline(args.config)
    pipeline.run(stages=["ablation"], model_key=args.model_key, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
