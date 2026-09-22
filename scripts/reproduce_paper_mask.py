#!/usr/bin/env python3
"""Re-solve CoCurve from a released H matrix and verify the paper mask.

This path does not load model weights or calibration data.  It is the fastest
way to audit the solver, budget accounting, registry, and artifact provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cocurve.artifacts import resolve
from cocurve.solver import greedy_budget_prune


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def solver_geometry(registry: dict, protect: int = 2) -> tuple[list[float], list[int], set[int]]:
    units = registry["units"]
    costs = [float(unit["cost"]) for unit in units]
    if "tower" not in units[0]:
        layers = [int(unit["layer_idx"]) for unit in units]
        n_layers = max(layers) + 1
        protected = set(range(protect)) | set(range(n_layers - protect, n_layers))
        return costs, layers, protected

    # Each VLM tower owns its boundary layers.  Offset the vision indices so
    # same-numbered language and vision layers do not share a cap.
    towers = ("lm", "vis")
    depth = {tower: 1 + max(int(u["layer_idx"]) for u in units if u["tower"] == tower)
             for tower in towers}
    offset = {"lm": 0, "vis": depth["lm"]}
    layers = [offset[u["tower"]] + int(u["layer_idx"]) for u in units]
    protected = set()
    for tower in towers:
        start, count = offset[tower], depth[tower]
        protected.update(start + i for i in range(protect))
        protected.update(start + count - 1 - i for i in range(protect))
    return costs, layers, protected


def reproduce(model: str, ratio: int, artifact_root: str | None = None) -> dict:
    artifact = resolve(model, ratio, artifact_root)
    h = np.load(artifact.h_path).astype(np.float64)
    registry = load_json(artifact.registry_path)
    expected = load_json(artifact.mask_path)
    costs, layers, protected = solver_geometry(registry)
    result = greedy_budget_prune(
        h=h,
        costs=np.asarray(costs, dtype=np.float64),
        prune_ratio=float(expected["target_prune_ratio"]),
        allow_budget_overshoot=True,
        normalize_by_cost=True,
        unit_layers=layers,
        max_pruned_cost_fraction_per_layer=1.5 * float(expected["target_prune_ratio"]),
        protected_layers=protected,
        interaction_strength=float(expected.get(
            "reproduction_interaction_strength", expected["interaction_strength"])),
    )
    got, want = set(result.pruned_units), set(map(int, expected["pruned_units"]))
    return {
        "model": model,
        "ratio": ratio,
        "match": got == want,
        "published_units": len(want),
        "reproduced_units": len(got),
        "symmetric_difference": len(got ^ want),
        "published_actual_ratio": float(expected["actual_prune_ratio"]),
        "reproduced_actual_ratio": float(result.actual_prune_ratio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ratio", required=True, type=int)
    parser.add_argument("--artifact-root", default=None)
    args = parser.parse_args()
    report = reproduce(args.model, args.ratio, args.artifact_root)
    print(json.dumps(report, indent=2))
    if not report["match"]:
        raise SystemExit("released H/registry/solver did not reproduce the published mask")


if __name__ == "__main__":
    main()
