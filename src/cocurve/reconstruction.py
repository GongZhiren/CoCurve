"""Sequential recovery of a CoCurve structure without baseline dependencies.

OBS solves the constrained least-squares problem for a fixed mask. FFN
refinement retains the selected attention groups and per-layer channel counts,
and reselects FFN channels with the block-OBS procedure of SlimGPT (Ling et al.,
NeurIPS 2024). This is an optional recovery stage, not the CoCurve selector.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import attention_module, attention_projections, layer_modules, mlp_module, mlp_projections
from .prune import apply_structural_prune_inplace
from .units import build_unit_registry


def _prepared(weight, gram, damping):
    if not torch.isfinite(torch.tensor(damping)) or damping <= 0:
        raise ValueError("damping must be finite and positive")
    if gram.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Input Gram shape does not match projection input width")
    w, h = weight.detach().float().clone(), gram.detach().float().clone()
    if not torch.isfinite(w).all() or not torch.isfinite(h).all():
        raise ValueError("Non-finite projection or calibration statistics")
    dead = h.diagonal() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(damping * h.diagonal().mean())
    return w, h


@torch.no_grad()
def constrained_obs(weight, gram, removed, damping=0.01):
    """Compensate retained columns, with removed columns constrained to zero."""
    w, h = _prepared(weight, gram, damping)
    removed = removed.to(device=h.device, dtype=torch.bool)
    if removed.shape != (w.shape[1],):
        raise ValueError("Removal mask must contain one entry per input column")
    keep, gone = (~removed).nonzero().flatten(), removed.nonzero().flatten()
    if not len(gone):
        return w
    if not len(keep):
        return torch.zeros_like(w)
    rr = h[keep][:, keep]
    rhs = h[keep][:, gone] @ w[:, gone].T
    delta = torch.cholesky_solve(rhs, torch.linalg.cholesky(rr))
    if (rr @ delta - rhs).norm() > 5e-4 * rhs.norm().clamp_min(1e-30):
        raise RuntimeError("OBS normal-equation residual exceeds tolerance")
    w[:, keep] += delta.T
    w[:, gone] = 0
    return w


@torch.no_grad()
def refine_ffn_columns(weight, gram, count, damping=0.01):
    """Reselect exactly count FFN channels with iterative block OBS."""
    if not 0 <= count <= weight.shape[1]:
        raise ValueError("Invalid FFN removal count")
    w, h = _prepared(weight, gram, damping)
    removed = torch.zeros(w.shape[1], dtype=torch.bool, device=w.device)
    if count == w.shape[1]:
        return torch.zeros_like(w), ~removed
    pruned, block_size = 0, (count - 512) // 2
    while pruned < count:
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
        error = (w.square() / inverse.diagonal().unsqueeze(0)).sum(0)
        error[removed] = torch.inf
        order = error.argsort()
        take = min(count - pruned, max(block_size, 64), 1024)
        w = w[:, order]
        upper = torch.linalg.cholesky(inverse[order][:, order], upper=True)[:take]
        local = w[:, :take].clone()
        correction = torch.zeros_like(local)
        for index in range(take):
            correction[:, index:index + 1] = local[:, index:index + 1] / upper[index, index]
            local[:, index:] -= correction[:, index:index + 1] @ upper[index:index + 1, index:take]
        w[:, :take] = 0
        w[:, take:w.shape[1] - pruned] -= correction @ upper[:, take:w.shape[1] - pruned]
        w = w[:, order.argsort()]
        gone = order[:take]
        h[gone, :] = 0
        h[:, gone] = 0
        h[gone, gone] = 1
        removed[gone] = True
        pruned += take
        block_size = (block_size - 512) // 2
    return w, removed


class _InputGram:
    """Match the sequence-normalized input statistics used in the experiments."""

    def __init__(self, width, device):
        self.gram = torch.zeros(width, width, device=device, dtype=torch.float32)
        self.samples = 0

    def __call__(self, module, args, output):
        x = args[0].detach()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        self.gram *= self.samples / (self.samples + batch)
        self.samples += batch
        x = x.reshape(-1, x.shape[-1]).T.float() * (2 / self.samples) ** 0.5
        self.gram.add_(x @ x.T)


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(x, device) for x in value)
    if isinstance(value, list):
        return [_move(x, device) for x in value]
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    return value


def _forward(layer, state, kwargs, device):
    out = layer(state.to(device), **_move(kwargs, device))
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def _capture(bundle, inputs, device):
    class Captured(Exception):
        pass

    states, kwargs = [], {}
    first = layer_modules(bundle)[0]

    def hook(module, args, call_kwargs):
        states.append(args[0].detach().cpu())
        if not kwargs:
            kwargs.update(_move(call_kwargs, "cpu"))
        raise Captured

    handle = first.register_forward_pre_hook(hook, with_kwargs=True)
    embed = bundle.model.get_input_embeddings()
    rotary = getattr(bundle.model.model, "rotary_emb", None)
    embed.to(device)
    if rotary is not None:
        rotary.to(device)
    try:
        for ids in inputs:
            try:
                bundle.model(input_ids=ids.to(device), use_cache=False)
            except Captured:
                pass
    finally:
        handle.remove()
        embed.cpu()
        if rotary is not None:
            rotary.cpu()
    if len(states) != len(inputs):
        raise RuntimeError("Could not capture all calibration windows")
    return states, kwargs


@torch.no_grad()
def reconstruct(bundle, mask, inputs, *, method="obs", damping=0.01, device="cuda:0",
                ffn_groups=16):
    """Recover on one device, then physically slice the actual retained units.

    Start from an unsliced, unquantized CPU model and independent calibration
    token windows. Inputs contain shape [1, sequence_length] token tensors.
    Returns the actual channel-level mask and reconstruction settings.
    """
    if method not in ("obs", "ffn_refinement"):
        raise ValueError("method must be obs or ffn_refinement")
    if getattr(bundle.model, "is_quantized", False):
        raise ValueError("Reconstruct floating-point weights before quantization")
    if any(p.device.type != "cpu" for p in bundle.model.parameters()):
        raise ValueError("Load the model on CPU; reconstruction moves one layer at a time")
    if not inputs:
        raise ValueError("At least one calibration window is required")
    if any(ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] != inputs[0].shape[1] for ids in inputs):
        raise ValueError("Calibration windows must have the same shape [1, length]")
    source = json.loads(Path(mask).read_text())
    registry_args = {"num_layers": bundle.num_layers, "num_heads": bundle.num_heads,
                     "hidden_size": bundle.hidden_size, "intermediate_size": bundle.intermediate_size,
                     "cost_type": "params", "kv_heads": bundle.kv_heads, "head_dim": bundle.head_dim}
    coarse = build_unit_registry(ffn_groups_per_layer=ffn_groups, **registry_args)
    all_ids = set(range(coarse.num_units))
    gone = set(source["pruned_units"])
    if not gone <= all_ids or set(source.get("selected_units", all_ids - gone)) != all_ids - gone:
        raise ValueError("Mask partition does not match the model's unit registry")
    masks = {(i, k): torch.zeros(width, dtype=torch.bool)
             for i in range(bundle.num_layers)
             for k, width in (("attn", bundle.num_heads * bundle.head_dim),
                              ("ffn", bundle.intermediate_size))}
    for unit in coarse.units:
        if unit.unit_id not in gone:
            continue
        if unit.unit_type == "attn_head":
            for head in unit.metadata["query_head_indices"]:
                masks[unit.layer_idx, "attn"][head * bundle.head_dim:(head + 1) * bundle.head_dim] = True
        else:
            start, size = unit.metadata["start_channel"], unit.metadata["group_size"]
            masks[unit.layer_idx, "ffn"][start:start + size] = True
    if any(masks[i, "attn"].all() for i in range(bundle.num_layers)):
        raise ValueError("Whole-attention removal is not supported")
    bundle.model.eval()
    states, kwargs = _capture(bundle, inputs, device)
    records = []
    for i, layer in enumerate(layer_modules(bundle)):
        layer.to(device)
        projections = mlp_projections(mlp_module(layer))
        down = projections.get("down_proj") or projections.get("w2")
        o = attention_projections(attention_module(layer)).get("o_proj")
        if down is None or o is None:
            raise ValueError("Recovery requires separate attention output and FFN down projections")
        for kind, projection in (("attn", o), ("ffn", down)):
            removed = masks[i, kind]
            if not removed.any():
                continue
            stats = _InputGram(projection.in_features, device)
            handle = projection.register_forward_hook(stats)
            try:
                for state in states:
                    _forward(layer, state, kwargs, device)
            finally:
                handle.remove()
            if kind == "ffn" and method == "ffn_refinement":
                weight, actual = refine_ffn_columns(
                    projection.weight, stats.gram, int(removed.sum()), damping)
                masks[i, kind] = actual.cpu()
            else:
                weight = constrained_obs(projection.weight, stats.gram, removed, damping)
            projection.weight.copy_(weight.to(projection.weight.dtype))
            del stats, weight
        states = [_forward(layer, state, kwargs, device).detach().cpu() for state in states]
        layer.cpu()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        records.append({"layer": i, "attention_columns_removed": int(masks[i, "attn"].sum()),
                        "ffn_channels_removed": int(masks[i, "ffn"].sum())})
    fine = build_unit_registry(ffn_groups_per_layer=bundle.intermediate_size, **registry_args)
    removed_ids = set()
    attention, ffn = {}, {}
    for unit in fine.units:
        (attention if unit.unit_type == "attn_head" else ffn).setdefault(unit.layer_idx, []).append(unit)
        if unit.unit_type == "attn_head":
            col = unit.metadata["query_head_indices"][0] * bundle.head_dim
            remove = masks[unit.layer_idx, "attn"][col]
        else:
            remove = masks[unit.layer_idx, "ffn"][unit.metadata["start_channel"]]
        if remove:
            removed_ids.add(unit.unit_id)
    before = sum(p.numel() for p in bundle.model.parameters())
    result = apply_structural_prune_inplace(
        bundle, attention, ffn, set(range(fine.num_units)) - removed_ids)
    after = sum(p.numel() for p in bundle.model.parameters())
    expected_removed = sum(u.cost for u in coarse.units if u.unit_id in gone)
    if (not getattr(bundle.model.config, "attention_bias", False)
            and not getattr(bundle.model.config, "mlp_bias", False)
            and before - after != expected_removed):
        raise RuntimeError("Reconstructed structure does not match the original projection budget")
    return {"method": method,
            "ffn_groups_per_layer": bundle.intermediate_size, "pruned_units": sorted(removed_ids),
            "selected_units": sorted(set(range(fine.num_units)) - removed_ids),
            "actual_prune_ratio": expected_removed / sum(u.cost for u in coarse.units),
            "attention_identities_fixed": True, "per_layer_ffn_counts_fixed": True,
            "ffn_identities_fixed": method == "obs", "parameters_before": before,
            "parameters_after": after, "layers": records, "physical_structure": result}
