"""True structural slicing must preserve the fixed-mask computation."""
from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from cocurve.model import ModelBundle
from cocurve.prune import (
    apply_structural_prune_inplace,
    clear_runtime_masks,
    register_runtime_masks,
)
from cocurve.units import build_unit_registry


def _bundle(model: LlamaForCausalLM) -> ModelBundle:
    return ModelBundle(
        model=model,
        tokenizer=None,
        device="cpu",
        num_layers=3,
        num_heads=8,
        hidden_size=64,
        head_dim=8,
        intermediate_size=96,
        kv_heads=2,
        attn_pattern="gqa",
        family="llama",
        model_path="synthetic-test",
    )


def test_structural_slice_matches_runtime_mask() -> None:
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    masked_model = LlamaForCausalLM(config).eval()
    sliced_model = LlamaForCausalLM(config).eval()
    sliced_model.load_state_dict(masked_model.state_dict())

    registry = build_unit_registry(
        num_layers=3,
        num_heads=8,
        hidden_size=64,
        intermediate_size=96,
        ffn_groups_per_layer=6,
        cost_type="params",
        kv_heads=2,
        head_dim=8,
    )
    attention, ffn = {}, {}
    for unit in registry.units:
        target = attention if unit.unit_type == "attn_head" else ffn
        target.setdefault(unit.layer_idx, []).append(unit)

    pruned = {
        unit.unit_id
        for unit in registry.units
        if (
            unit.layer_idx == 0
            and unit.unit_type == "attn_head"
            and unit.index_within_layer == 1
        )
        or (
            unit.layer_idx in (0, 1)
            and unit.unit_type == "ffn_group"
            and unit.index_within_layer in (2, 4)
        )
    }
    kept = {unit.unit_id for unit in registry.units} - pruned
    inputs = torch.randint(0, 128, (2, 16))

    state = register_runtime_masks(_bundle(masked_model), attention, ffn, kept)
    with torch.no_grad():
        masked_logits = masked_model(inputs).logits
    clear_runtime_masks(state)

    before = sum(parameter.numel() for parameter in sliced_model.parameters())
    apply_structural_prune_inplace(_bundle(sliced_model), attention, ffn, kept)
    after = sum(parameter.numel() for parameter in sliced_model.parameters())
    with torch.no_grad():
        sliced_logits = sliced_model(inputs).logits

    assert after < before
    assert torch.max(torch.abs(masked_logits - sliced_logits)).item() < 1e-4
