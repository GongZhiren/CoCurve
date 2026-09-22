#!/usr/bin/env python3

import argparse
import re
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
    issues = []

    for cfg_file in sorted(cfg_dir.glob("*.yaml")):
        data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        for k, v in walk(data):
            if not isinstance(v, str):
                continue
            if v.startswith("/"):
                issues.append((cfg_file.name, k, v))

    if issues:
        print("Found disallowed absolute paths:")
        for f, k, v in issues:
            print(f"  {f} :: {k} = {v}")
        raise SystemExit(1)

    private_patterns = {
        "private filesystem path": re.compile(r"/(?:scratch|home)/[A-Za-z0-9_.-]+/"),
        "Overleaf token": re.compile(r"\bolp_[A-Za-z0-9]+"),
        "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
        "literal IPv4 address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }
    text_suffixes = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".cff", ".txt"}
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in text_suffixes:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in private_patterns.items():
            if pattern.search(content):
                issues.append((str(path.relative_to(root)), label, "redacted"))

    if issues:
        print("Release hygiene failures:")
        for filename, key, value in issues:
            print(f"  {filename} :: {key} = {value}")
        raise SystemExit(1)

    print("Release hygiene passed: portable configs and no private paths, hosts, or tokens.")


if __name__ == "__main__":
    main()
