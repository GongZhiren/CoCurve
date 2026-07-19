#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def choice_label(idx: int) -> str:
    return chr(ord("A") + idx)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, {"__parse_error__": str(exc)}


def prompt_and_answer(record: Dict[str, Any]) -> Tuple[str, str]:
    prompt_keys = ("prompt", "question", "query", "instruction", "text", "input")
    answer_keys = ("answer", "target", "gold", "label", "output", "answerKey")
    prompt = ""
    answer = ""
    for key in prompt_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            prompt = value.strip()
            break
    for key in answer_keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            answer = str(value).strip()
            break
    return prompt, answer


def choices_from_record(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    choices = record.get("choices")
    if isinstance(choices, list):
        return [(choice_label(i), str(choice)) for i, choice in enumerate(choices)]
    raw = record.get("raw")
    if isinstance(raw, dict):
        raw_choices = raw.get("choices")
        if isinstance(raw_choices, dict):
            texts = raw_choices.get("text")
            labels = raw_choices.get("label")
            if isinstance(texts, list):
                if not isinstance(labels, list) or len(labels) != len(texts):
                    labels = [choice_label(i) for i in range(len(texts))]
                return [(str(label), str(text)) for label, text in zip(labels, texts)]
        endings = raw.get("endings")
        if isinstance(endings, list):
            return [(choice_label(i), str(choice)) for i, choice in enumerate(endings)]
        out = []
        for i in range(1, 6):
            key = f"option{i}"
            if key in raw:
                out.append((choice_label(i - 1), str(raw[key])))
        return out
    return []


def prompt_contains_options(prompt: str) -> bool:
    upper = prompt.upper()
    return any(marker in upper for marker in ("A.", "A)", "CHOICES:", "OPTION A"))


def validate_file(path: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "rows": 0,
        "parse_errors": [],
        "missing_prompt": [],
        "missing_answer": [],
        "lm_empty_text": [],
        "multiple_choice_rows": 0,
        "multiple_choice_without_options": [],
        "multiple_choice_options_in_side_fields": 0,
        "task_types": {},
        "examples": [],
    }
    for line_no, row in load_jsonl(path):
        stats["rows"] += 1
        if "__parse_error__" in row:
            stats["parse_errors"].append({"line": line_no, "error": row["__parse_error__"]})
            continue
        prompt, answer = prompt_and_answer(row)
        task_type = str(row.get("task_type", "unknown"))
        stats["task_types"][task_type] = stats["task_types"].get(task_type, 0) + 1
        if task_type == "lm":
            if not prompt:
                stats["lm_empty_text"].append(line_no)
            continue
        if not prompt:
            stats["missing_prompt"].append(line_no)
        if not answer:
            stats["missing_answer"].append(line_no)
        choices = choices_from_record(row)
        if task_type == "multiple_choice":
            stats["multiple_choice_rows"] += 1
            if choices:
                stats["multiple_choice_options_in_side_fields"] += 1
            if not choices and not prompt_contains_options(prompt):
                stats["multiple_choice_without_options"].append(line_no)
        if len(stats["examples"]) < 3:
            stats["examples"].append(
                {
                    "line": line_no,
                    "task_type": task_type,
                    "has_prompt": bool(prompt),
                    "has_answer": bool(answer),
                    "num_choices": len(choices),
                    "prompt_contains_options": prompt_contains_options(prompt),
                }
            )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate project-local JSONL datasets for evaluation readiness.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="data/datasets/validation_report.json")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    data_root = root / "data" / "datasets"
    files = sorted(data_root.rglob("*.jsonl"))
    report: Dict[str, Any] = {"data_root": str(data_root), "num_files": len(files), "files": {}, "summary": {}}
    total_rows = 0
    total_errors = 0
    total_mc_without_options = 0
    for path in files:
        stats = validate_file(path)
        rel = str(path.relative_to(root))
        report["files"][rel] = stats
        total_rows += int(stats["rows"])
        total_errors += len(stats["parse_errors"]) + len(stats["missing_prompt"]) + len(stats["missing_answer"])
        total_mc_without_options += len(stats["multiple_choice_without_options"])

    report["summary"] = {
        "total_rows": total_rows,
        "total_structural_errors": total_errors,
        "total_multiple_choice_without_options": total_mc_without_options,
        "status": "ok" if total_errors == 0 and total_mc_without_options == 0 else "needs_attention",
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
