from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .model import ModelBundle, attention_module, attention_projections, layer_modules, mlp_module, mlp_projections
from .types import UnitSpec


@dataclass
class RuntimeMaskState:
    hooks: List[torch.utils.hooks.RemovableHandle]


@dataclass
class PhysicalBackupEntry:
    tensor: torch.Tensor
    index: Tuple
    original: torch.Tensor


def _attn_head_keep_mask(specs: List[UnitSpec], selected_unit_ids: Set[int], num_heads: int) -> torch.Tensor:
    mask = torch.ones(num_heads, dtype=torch.float32)
    if not specs:
        return mask
    for spec in specs:
        if spec.unit_id not in selected_unit_ids:
            for head_idx in spec.metadata["query_head_indices"]:
                if 0 <= head_idx < num_heads:
                    mask[head_idx] = 0.0
    return mask


def register_runtime_masks(
    bundle: ModelBundle,
    units_by_layer: Dict[int, List[UnitSpec]],
    ffn_by_layer: Dict[int, List[UnitSpec]],
    selected_unit_ids: Set[int],
    comp_means: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
    ffn_gains: Optional[Dict[int, float]] = None,
) -> RuntimeMaskState:
    # ffn_gains (optional): edge-derived COUPLING-AWARE compensation. Maps a KEPT FFN
    # unit_id -> multiplicative gain (1+c_v) applied to its channels, so surviving units
    # are rescaled to absorb the removed units' output contribution (gains solved in
    # closed form from the same edge matrix H; see scripts/compute_edge_compensation.py).
    # Default None leaves kept units at gain 1.0 (clean selection-only behaviour).
    # comp_means (optional): FLAP-style mean/bias compensation. When provided,
    # pruned channels are replaced by their calibration mean instead of zero, i.e.
    #   out = W(h*mask + mean*(1-mask)) = W(h_kept) + W*mean_removed (a constant bias).
    # This preserves the removed channels' EXPECTED contribution (only their
    # fluctuation is dropped). Closed-form, label-free, no fine-tuning. When None,
    # behaviour is identical to plain zeroing (the clean default).
    comp_ffn = (comp_means or {}).get("ffn", {})
    comp_attn = (comp_means or {}).get("attn", {})
    hooks: List[torch.utils.hooks.RemovableHandle] = []
    layers = layer_modules(bundle)

    for layer_idx, layer in enumerate(layers):
        attn_specs = units_by_layer.get(layer_idx, [])
        if attn_specs:
            keep_mask = _attn_head_keep_mask(attn_specs, selected_unit_ids, bundle.num_heads)
            attn = attention_module(layer)
            projs = attention_projections(attn) if attn is not None else {}
            o_proj = projs.get("o_proj")
            if o_proj is not None:
                attn_mean = comp_attn.get(layer_idx)

                def make_o_proj_prehook(mask: torch.Tensor, mean: Optional[torch.Tensor]):
                    def hook(_module: nn.Module, inputs):
                        if not inputs:
                            return inputs
                        hidden = inputs[0]
                        if hidden.dim() != 3:
                            return inputs
                        bsz, seq, width = hidden.shape
                        expected_width = bundle.num_heads * bundle.head_dim
                        if width != expected_width:
                            return inputs
                        reshaped = hidden.view(bsz, seq, bundle.num_heads, bundle.head_dim)
                        mask_local = mask.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, bundle.num_heads, 1)
                        masked = reshaped * mask_local
                        if mean is not None:
                            mean_local = mean.to(device=hidden.device, dtype=hidden.dtype).view(
                                1, 1, bundle.num_heads, bundle.head_dim)
                            masked = masked + mean_local * (1.0 - mask_local)
                        masked = masked.reshape(bsz, seq, width)
                        return (masked,) + tuple(inputs[1:])

                    return hook

                hooks.append(o_proj.register_forward_pre_hook(make_o_proj_prehook(keep_mask, attn_mean)))

        ffn_specs = ffn_by_layer.get(layer_idx, [])
        if ffn_specs:
            mlp = mlp_module(layer)
            if mlp is not None:
                channel_mask = torch.ones(bundle.intermediate_size, dtype=torch.float32)
                for spec in ffn_specs:
                    start = spec.metadata["start_channel"]
                    size = spec.metadata["group_size"]
                    if spec.unit_id not in selected_unit_ids:
                        channel_mask[start : start + size] = 0.0
                    elif ffn_gains is not None and spec.unit_id in ffn_gains:
                        channel_mask[start : start + size] = float(ffn_gains[spec.unit_id])

                projections = mlp_projections(mlp)
                down_proj = projections.get("down_proj") or projections.get("w2") or projections.get("c_proj")
                if down_proj is not None:
                    ffn_mean = comp_ffn.get(layer_idx)

                    def make_mlp_prehook(mask: torch.Tensor, mean: Optional[torch.Tensor]):
                        def hook(_module: nn.Module, inputs):
                            if not inputs:
                                return inputs
                            hidden = inputs[0]
                            if hidden.dim() != 3:
                                return inputs
                            if hidden.shape[-1] != mask.shape[0]:
                                return inputs
                            mask_local = mask.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                            masked = hidden * mask_local
                            if mean is not None:
                                mean_local = mean.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                                masked = masked + mean_local * (1.0 - mask_local)
                            return (masked,) + tuple(inputs[1:])

                        return hook

                    hooks.append(down_proj.register_forward_pre_hook(make_mlp_prehook(channel_mask, ffn_mean)))
    return RuntimeMaskState(hooks=hooks)


def clear_runtime_masks(mask_state: RuntimeMaskState) -> None:
    for hook in mask_state.hooks:
        hook.remove()
    mask_state.hooks.clear()


def apply_physical_prune_inplace(
    bundle: ModelBundle,
    units_by_layer: Dict[int, List[UnitSpec]],
    ffn_by_layer: Dict[int, List[UnitSpec]],
    selected_unit_ids: Set[int],
) -> List[PhysicalBackupEntry]:
    # Physical pruning here uses exact zeroing of pruned slices to preserve shape compatibility.
    # It is equivalent to runtime masking for functional checks and downstream export.
    layers = layer_modules(bundle)
    head_dim = bundle.head_dim
    backups: List[PhysicalBackupEntry] = []

    def backup_slice(tensor: torch.Tensor, idx: Tuple) -> None:
        backups.append(PhysicalBackupEntry(tensor=tensor, index=idx, original=tensor[idx].clone()))

    for layer_idx, layer in enumerate(layers):
        attn_specs = units_by_layer.get(layer_idx, [])
        if attn_specs:
            attn = attention_module(layer)
            projs = attention_projections(attn) if attn is not None else {}
            q_proj = projs.get("q_proj")
            k_proj = projs.get("k_proj")
            v_proj = projs.get("v_proj")
            o_proj = projs.get("o_proj")
            if o_proj is not None:
                weight_o = o_proj.weight.data
                keep_mask = _attn_head_keep_mask(attn_specs, selected_unit_ids, bundle.num_heads)
                for head_idx in range(bundle.num_heads):
                    if keep_mask[head_idx] == 0:
                        start = head_idx * head_dim
                        end = start + head_dim
                        idx_o = (slice(None), slice(start, end))
                        backup_slice(weight_o, idx_o)
                        weight_o[idx_o] = 0.0
                        if q_proj is not None:
                            wq = q_proj.weight.data
                            idx_q = (slice(start, end), slice(None))
                            backup_slice(wq, idx_q)
                            wq[idx_q] = 0.0

            # shared kv heads are zeroed once if all linked query heads are pruned.
            if k_proj is not None and v_proj is not None:
                for spec in attn_specs:
                    if spec.unit_id in selected_unit_ids:
                        continue
                    kv_idx = int(spec.metadata["kv_head_index"])
                    kv_start = kv_idx * head_dim
                    kv_end = kv_start + head_dim
                    wk = k_proj.weight.data
                    wv = v_proj.weight.data
                    idx_kv = (slice(kv_start, kv_end), slice(None))
                    backup_slice(wk, idx_kv)
                    backup_slice(wv, idx_kv)
                    wk[idx_kv] = 0.0
                    wv[idx_kv] = 0.0

        ffn_specs = ffn_by_layer.get(layer_idx, [])
        if ffn_specs:
            mlp = mlp_module(layer)
            if mlp is None:
                continue
            projections = mlp_projections(mlp)
            down_proj = projections.get("down_proj") or projections.get("w2") or projections.get("c_proj")
            up_proj = projections.get("up_proj") or projections.get("w3") or projections.get("c_fc")
            gate_proj = projections.get("gate_proj") or projections.get("w1")
            if down_proj is None or up_proj is None:
                continue
            for spec in ffn_specs:
                if spec.unit_id in selected_unit_ids:
                    continue
                start = spec.metadata["start_channel"]
                size = spec.metadata["group_size"]
                end = start + size
                wd = down_proj.weight.data
                wu = up_proj.weight.data
                idx_down = (slice(None), slice(start, end))
                idx_up = (slice(start, end), slice(None))
                backup_slice(wd, idx_down)
                backup_slice(wu, idx_up)
                wd[idx_down] = 0.0
                wu[idx_up] = 0.0
                if gate_proj is not None:
                    wg = gate_proj.weight.data
                    idx_gate = (slice(start, end), slice(None))
                    backup_slice(wg, idx_gate)
                    wg[idx_gate] = 0.0
    return backups


def restore_physical_prune(backups: List[PhysicalBackupEntry]) -> None:
    for entry in reversed(backups):
        entry.tensor[entry.index] = entry.original


def _sliced_linear(old: nn.Linear, keep_rows: List[int] | None, keep_cols: List[int] | None) -> nn.Linear:
    """Build a smaller nn.Linear keeping the given output rows / input cols.

    Used for true structural removal: the resulting layer has genuinely smaller
    weight matrices, so it does less compute and uses less memory (unlike the
    zeroing path, which keeps full shapes).
    """
    w = old.weight.data
    out_features, in_features = w.shape
    rows = list(range(out_features)) if keep_rows is None else keep_rows
    cols = list(range(in_features)) if keep_cols is None else keep_cols
    new = nn.Linear(len(cols), len(rows), bias=old.bias is not None)
    row_idx = torch.tensor(rows, dtype=torch.long, device=w.device)
    col_idx = torch.tensor(cols, dtype=torch.long, device=w.device)
    new_w = w.index_select(0, row_idx).index_select(1, col_idx).clone()
    new = new.to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new.weight.copy_(new_w)
        if old.bias is not None:
            new.bias.copy_(old.bias.data.index_select(0, row_idx).clone())
    return new


def apply_structural_prune_inplace(
    bundle: ModelBundle,
    units_by_layer: Dict[int, List[UnitSpec]],
    ffn_by_layer: Dict[int, List[UnitSpec]],
    selected_unit_ids: Set[int],
) -> Dict[str, object]:
    """Physically shrink weight matrices to realize true hardware speedup.

    Removes pruned attention KV-groups (query-head rows of q_proj/o_proj + the
    shared kv-head rows of k_proj/v_proj) and pruned FFN channel groups
    (gate/up rows + down cols), per layer. Because CoCurve's attention unit
    is a whole KV-aligned group, num_key_value_groups is preserved and the GQA
    head-count is re-inferred from the (smaller) projection shapes — so this is
    bit-for-bit equivalent to runtime masking while doing strictly less compute.

    This mutation is irreversible; use on a freshly loaded bundle.
    """
    layers = layer_modules(bundle)
    head_dim = bundle.head_dim
    info: Dict[str, object] = {"layers": []}

    for layer_idx, layer in enumerate(layers):
        layer_info: Dict[str, object] = {"layer_idx": layer_idx}
        attn_specs = units_by_layer.get(layer_idx, [])
        if attn_specs:
            attn = attention_module(layer)
            projs = attention_projections(attn) if attn is not None else {}
            q_proj, k_proj, v_proj, o_proj = (projs.get(n) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
            kept_q_heads: List[int] = []
            kept_kv_heads: List[int] = []
            for spec in attn_specs:
                if spec.unit_id in selected_unit_ids:
                    kept_q_heads.extend(int(h) for h in spec.metadata["query_head_indices"])
                    kept_kv_heads.append(int(spec.metadata["kv_head_index"]))
            kept_q_heads = sorted(set(kept_q_heads))
            kept_kv_heads = sorted(set(kept_kv_heads))
            did_attn = False
            if kept_q_heads and kept_kv_heads and len(kept_q_heads) < bundle.num_heads:
                q_rows = [h * head_dim + d for h in kept_q_heads for d in range(head_dim)]
                kv_rows = [h * head_dim + d for h in kept_kv_heads for d in range(head_dim)]
                if all(p is not None for p in (q_proj, k_proj, v_proj, o_proj)):
                    # Separate q/k/v/o projections (Llama / Mistral / Yi / Qwen / Falcon).
                    attn.q_proj = _sliced_linear(q_proj, q_rows, None)
                    attn.k_proj = _sliced_linear(k_proj, kv_rows, None)
                    attn.v_proj = _sliced_linear(v_proj, kv_rows, None)
                    attn.o_proj = _sliced_linear(o_proj, None, q_rows)
                    did_attn = True
                elif o_proj is not None and hasattr(attn, "qkv_proj"):
                    # Phi3-style FUSED qkv_proj: output rows are concatenated
                    #   [ q: num_heads*head_dim | k: kv_heads*head_dim | v: kv_heads*head_dim ].
                    nh, nkv = bundle.num_heads, bundle.kv_heads
                    q_off, k_off, v_off = 0, nh * head_dim, nh * head_dim + nkv * head_dim
                    keep_rows = ([q_off + r for r in q_rows]
                                 + [k_off + r for r in kv_rows]
                                 + [v_off + r for r in kv_rows])
                    attn.qkv_proj = _sliced_linear(attn.qkv_proj, keep_rows, None)
                    attn.o_proj = _sliced_linear(o_proj, None, q_rows)
                    did_attn = True
                if did_attn:
                    # Keep GQA grouping consistent across module/config attribute names.
                    new_groups = len(kept_q_heads) // max(1, len(kept_kv_heads))
                    for attr, val in (("num_key_value_groups", new_groups), ("num_heads", len(kept_q_heads)),
                                       ("num_attention_heads", len(kept_q_heads)),
                                       ("num_key_value_heads", len(kept_kv_heads))):
                        if hasattr(attn, attr):
                            setattr(attn, attr, val)
                    layer_info["kept_q_heads"] = len(kept_q_heads)
                    layer_info["kept_kv_heads"] = len(kept_kv_heads)

        ffn_specs = ffn_by_layer.get(layer_idx, [])
        if ffn_specs:
            mlp = mlp_module(layer)
            projections = mlp_projections(mlp) if mlp is not None else {}
            down_proj = projections.get("down_proj") or projections.get("w2") or projections.get("c_proj")
            up_proj = projections.get("up_proj") or projections.get("w3") or projections.get("c_fc")
            gate_proj = projections.get("gate_proj") or projections.get("w1")
            keep_channels: List[int] = []
            for spec in ffn_specs:
                if spec.unit_id in selected_unit_ids:
                    start = int(spec.metadata["start_channel"]); size = int(spec.metadata["group_size"])
                    keep_channels.extend(range(start, start + size))
            keep_channels = sorted(set(keep_channels))
            if keep_channels and len(keep_channels) < bundle.intermediate_size:
                if down_proj is not None and up_proj is not None:
                    # Separate gate/up/down (SwiGLU). up/gate produce intermediate channels
                    # (rows); down consumes them (cols).
                    new_up = _sliced_linear(up_proj, keep_channels, None)
                    new_down = _sliced_linear(down_proj, None, keep_channels)
                    for name, mod in (("up_proj", new_up), ("w3", new_up), ("c_fc", new_up)):
                        if hasattr(mlp, name):
                            setattr(mlp, name, mod); break
                    for name, mod in (("down_proj", new_down), ("w2", new_down), ("c_proj", new_down)):
                        if hasattr(mlp, name):
                            setattr(mlp, name, mod); break
                    if gate_proj is not None:
                        new_gate = _sliced_linear(gate_proj, keep_channels, None)
                        for name in ("gate_proj", "w1"):
                            if hasattr(mlp, name):
                                setattr(mlp, name, new_gate); break
                    if hasattr(mlp, "intermediate_size"):
                        mlp.intermediate_size = len(keep_channels)
                    layer_info["kept_ffn_channels"] = len(keep_channels)
                elif down_proj is not None and hasattr(mlp, "gate_up_proj"):
                    # Phi3-style FUSED gate_up_proj: output rows are concatenated
                    #   [ gate: intermediate | up: intermediate ]; down consumes the channels.
                    inter = bundle.intermediate_size
                    keep_rows = keep_channels + [inter + c for c in keep_channels]
                    mlp.gate_up_proj = _sliced_linear(mlp.gate_up_proj, keep_rows, None)
                    mlp.down_proj = _sliced_linear(down_proj, None, keep_channels)
                    if hasattr(mlp, "intermediate_size"):
                        mlp.intermediate_size = len(keep_channels)
                    layer_info["kept_ffn_channels"] = len(keep_channels)
        info["layers"].append(layer_info)

    return info
