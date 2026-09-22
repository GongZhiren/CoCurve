import json

from cocurve.artifacts import load_manifest, resolve


def test_manifest_resolves_every_operating_point():
    root, manifest = load_manifest()
    assert len(manifest["models"]) == 9
    for model_key, model in manifest["models"].items():
        for ratio in model["ratios"]:
            artifact = resolve(model_key, int(ratio), root)
            assert artifact.h_path.is_file()
            assert artifact.registry_path.is_file()
            assert artifact.mask_path.is_file()
            assert artifact.metrics_path.is_file()
            payload = json.loads(artifact.mask_path.read_text())
            assert payload["model_key"] == model_key
            assert not (set(payload["selected_units"]) & set(payload["pruned_units"]))
            registry = json.loads(artifact.registry_path.read_text())
            assert len(payload["selected_units"]) + len(payload["pruned_units"]) == len(registry["units"])
