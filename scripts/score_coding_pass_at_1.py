#!/usr/bin/env python3
"""Compute pass@1 for saved HumanEval/MBPP generations and patch the report.

Usage:
  PYTHONPATH=src python scripts/score_coding_pass_at_1.py \
      --run-dir outputs/experiments/llama-3.1-8b/<run> [--timeout 10] [--workers 8]

Reads <run>/standard_eval/{humaneval,mbpp}_generations.json, executes the unit
tests in isolated subprocesses, writes {task}_pass_at_1.json, and updates the
matching entries in standard_eval_report.json from the placeholder
`generation_saved_no_execution` metric to real pass@1 scores.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

from cocurve.code_execution import build_humaneval_program, build_mbpp_program, run_program


def _score_one(args):
    program, timeout = args
    return run_program(program, timeout=timeout)


def score_task(task: str, gen_path: Path, timeout: float, workers: int) -> Dict[str, object]:
    payload = json.loads(gen_path.read_text(encoding="utf-8"))
    generations: List[Dict[str, object]] = payload.get("generations", [])
    programs = []
    for g in generations:
        if task == "humaneval":
            programs.append(build_humaneval_program(str(g["prompt"]), str(g["prediction"]), str(g["test"]), str(g["entry_point"])))
        else:
            tests = g.get("tests") or g.get("test_list") or []
            programs.append(build_mbpp_program(str(g["prediction"]), tests))
    results: List[Dict[str, object]]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_score_one, [(p, timeout) for p in programs]))
    else:
        results = [run_program(p, timeout=timeout) for p in programs]
    passed = sum(int(bool(r["passed"])) for r in results)
    total = len(generations)
    rows = [{"task_id": g.get("task_id"), **r} for g, r in zip(generations, results)]
    return {"task": task, "metric": "pass@1", "num_samples": total,
            "pass@1": passed / max(1, total), "num_passed": passed, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    se = Path(args.run_dir) / "standard_eval"
    report_path = se / "standard_eval_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    for task in ("humaneval", "mbpp"):
        gen_path = se / f"{task}_generations.json"
        if not gen_path.exists():
            print(f"[skip] {gen_path} not found")
            continue
        result = score_task(task, gen_path, args.timeout, args.workers)
        (se / f"{task}_pass_at_1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        # Patch the summary report: keep elapsed_sec from the original generation entry.
        prev = report.get(task, {})
        report[task] = {
            "task": task,
            "metric": "pass@1",
            "num_samples": result["num_samples"],
            "pass@1": result["pass@1"],
            "num_passed": result["num_passed"],
            "generation_elapsed_sec": prev.get("elapsed_sec"),
        }
        print(f"[{task}] pass@1 = {result['pass@1']:.4f}  ({result['num_passed']}/{result['num_samples']})")

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] updated {report_path}")


if __name__ == "__main__":
    main()
