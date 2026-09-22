"""Resolve and validate publication-locked CoCurve artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


@dataclass(frozen=True)
class PaperArtifact:
    model_key: str
    model_id: str
    ratio: int
    kind: str
    precision: str
    h_path: Path
    registry_path: Path
    mask_path: Path
    metrics_path: Path


def default_root() -> Path:
    """Return the repository artifact root for editable/source installations."""
    package = Path(str(files("cocurve"))).resolve()
    candidate = package.parents[1] / "artifacts" / "paper-v1"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        "paper artifacts were not found next to the source tree; pass --artifact-root "
        "or clone the repository with Git LFS enabled"
    )


def load_manifest(root: str | Path | None = None) -> tuple[Path, Dict[str, Any]]:
    base = Path(root).expanduser().resolve() if root else default_root()
    path = base / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"artifact manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported artifact schema: {data.get('schema_version')}")
    return base, data


def resolve(model_key: str, ratio: int, root: str | Path | None = None) -> PaperArtifact:
    base, manifest = load_manifest(root)
    try:
        model = manifest["models"][model_key]
        paths = model["ratios"][str(int(ratio))]
    except KeyError as exc:
        available = sorted(manifest.get("models", {}))
        raise KeyError(
            f"no paper artifact for model={model_key!r}, ratio={ratio}; models={available}"
        ) from exc
    return PaperArtifact(
        model_key=model_key,
        model_id=model["model_id"],
        ratio=int(ratio),
        kind=model["kind"],
        precision=model.get("precision", "bf16"),
        h_path=base / model["h"],
        registry_path=base / model["registry"],
        mask_path=base / paths["mask"],
        metrics_path=base / paths["reference_metrics"],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: str | Path | None = None, selected: Iterable[str] | None = None) -> int:
    base, manifest = load_manifest(root)
    wanted = set(selected or ())
    failures = []
    checked = 0
    for rel, expected in manifest["files"].items():
        if wanted and not any(rel.startswith(f"llm/{m}/") or rel.startswith(f"vlm/{m}/") for m in wanted):
            continue
        path = base / rel
        if not path.is_file():
            failures.append(f"missing: {rel}")
            continue
        if path.stat().st_size != int(expected["bytes"]):
            failures.append(f"wrong size: {rel}")
            continue
        if _sha256(path) != expected["sha256"]:
            failures.append(f"checksum mismatch: {rel}")
            continue
        checked += 1
    if failures:
        raise RuntimeError("artifact verification failed:\n  " + "\n  ".join(failures))
    return checked


def validate(root: str | Path | None = None, selected: Iterable[str] | None = None) -> int:
    """Validate matrix, registry, mask, and budget semantics for paper artifacts."""
    base, manifest = load_manifest(root)
    wanted = set(selected or ())
    failures = []
    checked = 0
    for model_key, model in sorted(manifest["models"].items()):
        if wanted and model_key not in wanted:
            continue
        registry = json.loads((base / model["registry"]).read_text(encoding="utf-8"))
        units = registry.get("units", [])
        unit_ids = [int(unit["unit_id"]) for unit in units]
        if unit_ids != list(range(len(units))):
            failures.append(f"{model_key}: registry unit IDs are not contiguous and ordered")
            continue
        costs = {int(unit["unit_id"]): float(unit["cost"]) for unit in units}
        total_cost = sum(costs.values())
        h = np.load(base / model["h"], mmap_mode="r")
        if h.shape != (len(units), len(units)):
            failures.append(f"{model_key}: H shape {h.shape} != {(len(units), len(units))}")
        if not np.isfinite(h).all():
            failures.append(f"{model_key}: H contains non-finite values")
        if not np.allclose(h, h.T, rtol=0.0, atol=1e-7):
            failures.append(f"{model_key}: H is not symmetric")
        if np.any(np.diag(h) < -1e-8):
            failures.append(f"{model_key}: H has negative diagonal entries")
        universe = set(unit_ids)
        for ratio, paths in sorted(model["ratios"].items(), key=lambda item: int(item[0])):
            mask = json.loads((base / paths["mask"]).read_text(encoding="utf-8"))
            pruned = {int(value) for value in mask["pruned_units"]}
            kept = {int(value) for value in mask["selected_units"]}
            prefix = f"{model_key} r{ratio}"
            if mask.get("model_key") != model_key or mask.get("model_id") != model["model_id"]:
                failures.append(f"{prefix}: model provenance mismatch")
            if pruned & kept or pruned | kept != universe:
                failures.append(f"{prefix}: selected/pruned units do not partition the registry")
            pruned_cost = sum(costs.get(unit_id, float("nan")) for unit_id in pruned)
            actual_ratio = pruned_cost / total_cost
            if not np.isclose(pruned_cost, float(mask["pruned_cost"]), rtol=0.0, atol=1e-3):
                failures.append(f"{prefix}: pruned cost does not match the registry")
            if not np.isclose(total_cost, float(mask["total_cost"]), rtol=0.0, atol=1e-3):
                failures.append(f"{prefix}: total cost does not match the registry")
            if not np.isclose(actual_ratio, float(mask["actual_prune_ratio"]), rtol=0.0, atol=1e-12):
                failures.append(f"{prefix}: actual pruning ratio is inconsistent")
            checked += 1
    if failures:
        raise RuntimeError("artifact semantic validation failed:\n  " + "\n  ".join(failures))
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or verify CoCurve paper artifacts")
    parser.add_argument("command", choices=("list", "verify", "path"))
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--ratio", type=int, default=20)
    args = parser.parse_args()
    base, manifest = load_manifest(args.artifact_root)
    if args.command == "list":
        for key, model in sorted(manifest["models"].items()):
            ratios = ",".join(sorted(model["ratios"], key=int))
            print(f"{key:28s} {model['kind']:3s} ratios={ratios}  {model['model_id']}")
    elif args.command == "verify":
        file_count = verify(base, args.model)
        mask_count = validate(base, args.model)
        print(f"verified {file_count} files and validated {mask_count} masks under {base}")
    else:
        if len(args.model) != 1:
            parser.error("path requires exactly one --model")
        artifact = resolve(args.model[0], args.ratio, base)
        print(json.dumps({
            "H": str(artifact.h_path),
            "registry": str(artifact.registry_path),
            "mask": str(artifact.mask_path),
            "reference_metrics": str(artifact.metrics_path),
        }, indent=2))


if __name__ == "__main__":
    main()
