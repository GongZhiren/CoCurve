#!/usr/bin/env python3
"""Materialize the fixed local dataset snapshots used by paper evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def limited(dataset: Any, limit: int) -> Any:
    if limit <= 0:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def build_piqa(root: Path) -> int:
    dataset = load_dataset("lighteval/piqa", split="validation")
    rows = ({"goal": row["goal"], "sol1": row["sol1"], "sol2": row["sol2"],
             "label": int(row["label"])} for row in dataset)
    return write_jsonl(root / "reasoning/piqa/validation.jsonl", rows)


def build_openbookqa(root: Path) -> int:
    dataset = load_dataset("allenai/openbookqa", "main", split="test")
    rows = ({"question_stem": row["question_stem"], "choices": row["choices"],
             "answerKey": row["answerKey"]} for row in dataset)
    return write_jsonl(root / "reasoning/openbookqa/test.jsonl", rows)


def build_boolq(root: Path, limit: int) -> int:
    dataset = limited(load_dataset("google/boolq", split="validation"), limit)
    rows = ({"question": row["question"], "passage": row["passage"],
             "answer": bool(row["answer"])} for row in dataset)
    return write_jsonl(root / "reasoning/boolq/validation.jsonl", rows)


def build_ptb(root: Path) -> int:
    dataset = load_dataset(
        "parquet",
        data_files=("hf://datasets/ptb-text-only/ptb_text_only@refs%2Fconvert%2Fparquet/"
                    "penn_treebank/test/0000.parquet"),
        split="train",
    )
    rows = ({"text": row["sentence"]} for row in dataset)
    return write_jsonl(root / "language_modeling/ptb/test.jsonl", rows)


def build_c4(root: Path, limit: int) -> int:
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    def rows() -> Iterable[dict[str, str]]:
        for index, row in enumerate(dataset):
            if limit > 0 and index >= limit:
                break
            yield {"text": row["text"]}

    return write_jsonl(root / "language_modeling/c4/validation.jsonl", rows())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--boolq-limit", type=int, default=2000)
    parser.add_argument("--c4-limit", type=int, default=500)
    parser.add_argument(
        "--only", default="", help="comma-separated subset: piqa,openbookqa,boolq,ptb,c4"
    )
    args = parser.parse_args()
    root = Path(args.project_root) / "data" / "datasets"
    requested = {name.strip() for name in args.only.split(",") if name.strip()}
    jobs = {
        "piqa": lambda: build_piqa(root),
        "openbookqa": lambda: build_openbookqa(root),
        "boolq": lambda: build_boolq(root, args.boolq_limit),
        "ptb": lambda: build_ptb(root),
        "c4": lambda: build_c4(root, args.c4_limit),
    }
    unknown = requested - set(jobs)
    if unknown:
        parser.error(f"unknown datasets: {', '.join(sorted(unknown))}")
    for name, job in jobs.items():
        if requested and name not in requested:
            continue
        print(f"{name}: {job()} rows")


if __name__ == "__main__":
    main()
