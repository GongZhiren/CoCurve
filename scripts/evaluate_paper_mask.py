#!/usr/bin/env python3
"""Evaluate a released paper mask without recomputing CoCurve's H matrix."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cocurve.artifacts import resolve, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ratio", type=int, required=True)
    parser.add_argument("--precision", choices=("auto", "bf16", "nf4", "int8"), default="auto")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--output", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()

    artifact = resolve(args.model, args.ratio, args.artifact_root)
    verify(args.artifact_root, [args.model])
    if artifact.kind != "llm":
        raise SystemExit("use scripts/run_vlm.py for vision-language artifacts")
    precision = artifact.precision if args.precision == "auto" else args.precision
    output = Path(args.output or f"outputs/reproduction/{args.model}/r{args.ratio}/{precision}")
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_standard_eval.py")),
        "--config", "configs/paper.yaml",
        "--model-key", args.model,
        "--run-dir", str(output),
        "--mask-run-dir", str(artifact.mask_path.parent.parent),
        "--tasks", args.tasks,
        "--max-samples", str(args.max_samples),
    ]
    if precision == "bf16":
        command.append("--bf16")
    elif precision == "nf4":
        command.append("--load-in-4bit")
    elif precision == "int8":
        command.append("--load-in-8bit")
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
