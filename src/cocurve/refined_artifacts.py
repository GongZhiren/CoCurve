"""Exact evaluated FFN-refinement structures, compactly stored as channel bits."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np


def artifact_root():
    return Path(__file__).resolve().parents[2] / "artifacts/reconstruction-v1"


def load_structure(model_key, ratio, *, root=None):
    """Load an evaluated structure and verify every removed unit ID."""
    root = Path(root) if root is not None else artifact_root()
    manifest = json.loads((root / "manifest.json").read_text())
    point = next((p for p in manifest["points"]
                  if p["model_key"] == model_key and p["ratio"] == int(ratio)), None)
    if point is None:
        raise KeyError(f"No evaluated refinement snapshot for {model_key} at {ratio}%")
    path = root / point["structure"]
    raw = path.read_bytes()
    expected = manifest["files"][point["structure"]]
    if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("Refinement structure checksum mismatch")
    data = json.loads(raw)
    layers, heads, width = (int(data[k]) for k in ("num_layers", "num_kv_heads", "intermediate_size"))
    if layers < 1 or heads < 1 or width < 1:
        raise ValueError("Invalid structure dimensions")
    packed = np.frombuffer(base64.b64decode(data["removed_ffn_bits_base64"], validate=True), dtype=np.uint8)
    if len(packed) != layers * ((width + 7) // 8):
        raise ValueError("Invalid FFN channel-bit dimensions")
    ffn = np.unpackbits(packed.reshape(layers, -1), axis=1, bitorder="little")[:, :width]
    removed = []
    if len(data["removed_attention_groups"]) != layers:
        raise ValueError("Invalid attention layer count")
    for layer, groups in enumerate(data["removed_attention_groups"]):
        if len(set(groups)) != len(groups) or any(g < 0 or g >= heads for g in groups):
            raise ValueError("Invalid attention group indices")
        offset = layer * (heads + width)
        removed.extend(offset + int(g) for g in groups)
        removed.extend(offset + heads + int(c) for c in np.flatnonzero(ffn[layer]))
    removed.sort()
    return data, removed


def materialize(model_key, ratio, output, *, root=None):
    """Write a standard run-directory mask without selecting units again."""
    data, removed = load_structure(model_key, ratio, root=root)
    size = data["num_layers"] * (data["num_kv_heads"] + data["intermediate_size"])
    payload = {"schema_version": 1, "method": "cocurve_ffn_refinement",
               "model_key": model_key, "ffn_groups_per_layer": data["intermediate_size"],
               "pruned_units": removed, "selected_units": sorted(set(range(size)) - set(removed)),
               "actual_prune_ratio": data["actual_prune_ratio"],
               "structure_version": "reconstruction-v1"}
    path = Path(output) / "masks/prune_solution.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2) + "\n"
    if path.exists() and path.read_text() != raw:
        raise FileExistsError("Output already contains a different mask")
    path.write_text(raw)
    return path
