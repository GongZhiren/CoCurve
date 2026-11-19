"""Released refinement masks must preserve the original heads and layer quotas."""
import json

import numpy as np

from cocurve.refined_artifacts import artifact_root, load_structure, materialize


def test_evaluated_refinement_snapshots_preserve_original_allocation(tmp_path):
    root = artifact_root()
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["points"]) == 6
    paper = root.parent / "paper-v1/llm"
    for point in manifest["points"]:
        model, ratio = point["model_key"], point["ratio"]
        meta, removed = load_structure(model, ratio)
        registry = json.loads((paper / model / "registry.json").read_text())["units"]
        coarse = json.loads((paper / model / f"r{ratio}/masks/prune_solution.json").read_text())
        old = set(coarse["pruned_units"])
        stride = meta["num_kv_heads"] + meta["intermediate_size"]
        bits = np.zeros((meta["num_layers"], stride), dtype=np.uint8)
        bits.reshape(-1)[removed] = 1
        for layer in range(meta["num_layers"]):
            units = [u for u in registry if u["layer_idx"] == layer]
            expected_heads = {u["metadata"]["kv_head_index"] for u in units
                              if u["unit_type"] == "attn_head" and u["unit_id"] in old}
            expected_ffn = sum(u["metadata"]["group_size"] for u in units
                               if u["unit_type"] == "ffn_group" and u["unit_id"] in old)
            assert set(np.flatnonzero(bits[layer, :meta["num_kv_heads"]])) == expected_heads
            assert bits[layer, meta["num_kv_heads"]:].sum() == expected_ffn
        path = materialize(model, ratio, tmp_path / model / str(ratio))
        saved = json.loads(path.read_text())
        assert set(saved["pruned_units"]) == set(removed)
        assert len(saved["selected_units"]) + len(removed) == bits.size
