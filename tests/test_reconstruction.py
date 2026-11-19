"""Recovery mathematics, structure invariants, and reloadable sliced checkpoints."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from cocurve.checkpoints import (
    export_checkpoint,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from cocurve.model import ModelBundle
from cocurve.prune import apply_structural_prune_inplace, clear_runtime_masks, register_runtime_masks
from cocurve.reconstruction import constrained_obs, reconstruct
from cocurve.units import build_unit_registry


def _example():
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                         max_position_embeddings=64)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    bundle = ModelBundle(model, None, "cpu", 2, 4, 32, 8, 64, 2, "gqa", "llama", "synthetic")
    registry = build_unit_registry(2, 4, 32, 64, 16, "params", 2, head_dim=8)
    attention, ffn = {}, {}
    for unit in registry.units:
        (attention if unit.unit_type == "attn_head" else ffn).setdefault(unit.layer_idx, []).append(unit)
    return bundle, registry, attention, ffn


def test_obs_matches_independent_float64_solution():
    torch.manual_seed(3)
    weight = torch.randn(5, 16)
    inputs = torch.randn(128, 16)
    gram = inputs.T @ inputs
    removed = torch.arange(16) % 3 == 0
    actual = constrained_obs(weight, gram, removed)
    h = gram.double() + torch.eye(16) * (0.01 * gram.diagonal().mean().double())
    inverse = h.inverse()
    gold = weight.double()[:, ~removed] - weight.double()[:, removed] @ torch.linalg.solve(
        inverse[removed][:, removed], inverse[removed][:, ~removed])
    torch.testing.assert_close(actual[:, ~removed].double(), gold, atol=2e-6, rtol=2e-6)
    assert not actual[:, removed].count_nonzero()
    assert ((inputs @ actual.T - inputs @ weight.T).square().mean()
            < (inputs @ (weight * (~removed)).T - inputs @ weight.T).square().mean())
    assert torch.equal(constrained_obs(weight, gram, torch.zeros(16, dtype=torch.bool)), weight)
    assert not constrained_obs(weight, gram, torch.ones(16, dtype=torch.bool)).count_nonzero()


@pytest.mark.parametrize("method", ["obs", "ffn_refinement"])
def test_sequential_recovery_preserves_attention_and_channel_budgets(tmp_path, method):
    torch.manual_seed(0)
    bundle, reg, _, _ = _example()
    removed = {u.unit_id for u in reg.units
               if (u.unit_type == "attn_head" and u.index_within_layer == 1)
               or (u.unit_type == "ffn_group" and u.index_within_layer < 4)}
    path = tmp_path / "mask.json"
    path.write_text(json.dumps({"pruned_units": sorted(removed),
                               "selected_units": sorted(set(range(reg.num_units)) - removed)}))
    rows = [torch.randint(0, 64, (1, 8)) for _ in range(3)]
    result = reconstruct(bundle, path, rows, method=method, device="cpu")
    assert result["parameters_before"] - result["parameters_after"] == sum(
        u.cost for u in reg.units if u.unit_id in removed)
    for layer in bundle.model.model.layers:
        assert layer.self_attn.k_proj.out_features == 8
        assert layer.mlp.down_proj.in_features == 48
    with torch.no_grad():
        assert torch.isfinite(bundle.model(rows[0]).logits).all()
    dest = tmp_path / "checkpoint"
    save_checkpoint(bundle.model, dest)
    loaded = load_checkpoint(dest)
    with torch.no_grad():
        torch.testing.assert_close(bundle.model(rows[0]).logits, loaded(rows[0]).logits, rtol=0, atol=0)


def test_empty_ffn_matches_masking_and_survives_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(2)
    bundle, reg, attention, ffn = _example()
    dense = copy.deepcopy(bundle.model)
    removed = {u.unit_id for u in reg.units if u.layer_idx == 0 and u.unit_type == "ffn_group"}
    kept = set(range(reg.num_units)) - removed
    masks = register_runtime_masks(bundle, attention, ffn, kept)
    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        reference = bundle.model(ids).logits
    clear_runtime_masks(masks)
    before = sum(p.numel() for p in bundle.model.parameters())
    apply_structural_prune_inplace(bundle, attention, ffn, kept)
    assert before - sum(p.numel() for p in bundle.model.parameters()) == 3 * 32 * 64
    with torch.no_grad():
        torch.testing.assert_close(reference, bundle.model(ids).logits)
    target = tmp_path / "empty-ffn"
    save_checkpoint(bundle.model, target)
    loaded = load_checkpoint(target)
    with torch.no_grad():
        torch.testing.assert_close(reference, loaded(ids).logits)
    assert dense.model.layers[0].mlp.down_proj.in_features == 64


def test_checkpoint_inspection_rejects_truncation(tmp_path):
    bundle, _, _, _ = _example()
    save_checkpoint(bundle.model, tmp_path / "checkpoint")
    weight = tmp_path / "checkpoint/model.safetensors"
    with weight.open("r+b") as stream:
        stream.truncate(weight.stat().st_size - 10)
    with pytest.raises(ValueError):
        inspect_checkpoint(weight.parent)


def test_checkpoint_checksum_rejects_changed_payload(tmp_path):
    bundle, _, _, _ = _example()
    save_checkpoint(bundle.model, tmp_path / "checkpoint")
    weight = tmp_path / "checkpoint/model.safetensors"
    with weight.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    inspect_checkpoint(weight.parent)
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(weight.parent)


def test_streaming_export_preserves_every_weight(tmp_path):
    bundle, _, _, _ = _example()
    source, target = tmp_path / "source", tmp_path / "exported"
    save_checkpoint(bundle.model, source)
    manifest = export_checkpoint(source, target, max_shard_bytes=110_000)
    assert manifest["tensor_bytes_preserved"]
    restored = load_checkpoint(target)
    for key, value in bundle.model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_common_lora_can_start_from_reconstructed_checkpoint(tmp_path, monkeypatch):
    """Exercise the CLI's real PEFT training and merged serialization on a tiny model."""
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    bundle, _, _, _ = _example()
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, **{f"word{i}": i for i in range(3, 64)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    source = tmp_path / "reconstructed"
    save_checkpoint(bundle.model, source, tokenizer=tokenizer)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("recovery_cli", root / "scripts/run_lora_recovery.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "alpaca_tokens", lambda *args: torch.randint(3, 64, (2, 8)))
    out = tmp_path / "recovered"
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["run_lora_recovery.py", "--model-key", "llama-3.1-8b-instruct",
                                    "--reconstructed-checkpoint", str(source), "--out", str(out),
                                    "--seq-len", "8", "--batch", "1", "--accum", "1",
                                    "--skip-eval", "--save-merged"])
    runner.main()
    recovered = load_checkpoint(out)
    ids = torch.randint(3, 64, (1, 8))
    with torch.no_grad():
        assert torch.isfinite(recovered(ids).logits).all()
    meta = json.loads((out / "recovery_meta.json").read_text())
    assert meta["steps"] == 2 and meta["checkpoint_format"] == "merged"
