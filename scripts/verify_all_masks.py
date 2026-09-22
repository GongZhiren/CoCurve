#!/usr/bin/env python3
"""Reconstruct every released paper mask from H and require exact agreement."""
from __future__ import annotations

import argparse

from cocurve.artifacts import load_manifest
from reproduce_paper_mask import reproduce


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=None)
    args = parser.parse_args()
    root, manifest = load_manifest(args.artifact_root)
    count = 0
    for model, entry in sorted(manifest["models"].items()):
        for ratio in sorted(entry["ratios"], key=int):
            report = reproduce(model, int(ratio), str(root))
            if not report["match"]:
                raise RuntimeError(
                    f"{model} r{ratio}: {report['symmetric_difference']} units differ"
                )
            count += 1
            print(f"PASS {model} r{ratio}")
    print(f"reproduced all {count} publication masks exactly")


if __name__ == "__main__":
    main()
