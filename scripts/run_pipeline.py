#!/usr/bin/env python3

import argparse

from cocurve.pipeline import CoCurvePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoCurve pipeline stages.")
    parser.add_argument("--config", required=True, help="Path to default config yaml")
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma separated stage names: collect,ablation,matrix,solve,quality,json_eval,std_prep,apply or all",
    )
    parser.add_argument("--model-key", default=None, help="Optional model key override")
    parser.add_argument("--run-dir", default=None, help="Resume or write to a fixed run directory")
    parser.add_argument("--run-base-dir", default=None, help="Base directory for multi-model runs")
    parser.add_argument(
        "--multi-models",
        default=None,
        help="Comma separated model keys for multi-model execution",
    )
    args = parser.parse_args()

    pipeline = CoCurvePipeline(args.config)
    stage_list = [s.strip() for s in args.stages.split(",") if s.strip()]

    if args.multi_models:
        models = [m.strip() for m in args.multi_models.split(",") if m.strip()]
        pipeline.run_multi_models(models, stages=stage_list, run_base_dir=args.run_base_dir)
    else:
        pipeline.run(stages=stage_list, model_key=args.model_key, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
