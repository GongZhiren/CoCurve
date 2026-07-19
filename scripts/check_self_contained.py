#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any

import yaml


def walk(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            yield from walk(v, p)
    else:
        yield path, obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Check configs for external absolute paths.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cfg_dir = root / "configs"
    allow_keys = {
        "models.llama-3.1-8b.path",
        "models.llama-2-13b.path",
        "models.qwen2.5-7b.path",
        "models.mixtral-8x7b.path",
    }
    issues = []

    for cfg_file in sorted(cfg_dir.glob("*.yaml")):
        data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        for k, v in walk(data):
            if not isinstance(v, str):
                continue
            if v.startswith("/") and k not in allow_keys:
                issues.append((cfg_file.name, k, v))

    if issues:
        print("Found disallowed absolute paths:")
        for f, k, v in issues:
            print(f"  {f} :: {k} = {v}")
        raise SystemExit(1)

    print("Self-contained check passed: no disallowed absolute paths.")


if __name__ == "__main__":
    main()
