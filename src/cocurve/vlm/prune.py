"""True structural removal for the two towers of a VLM.

The VLM experiments select the same residual-writing units as the language-only
pipeline, but historically realized them with forward hooks.  Hooks are the
right implementation for quality evaluation and recovery; they are not a valid
way to claim deployment speed.  This module materializes the identical mask by
shrinking the affected attention and FFN matrices in both towers.

The mutation is intentionally irreversible.  Call it on a freshly loaded model
and validate it against :func:`cocurve.vlm.masks.masked` before timing.
"""
from __future__ import annotations

from typing import Dict, List, Set

import torch
from torch import nn

from .bundle import VLMBundle, _get
from .units import VLMRegistry, VLMUnit


def _sliced_linear(old: nn.Linear, keep_rows: List[int] | None,
                   keep_cols: List[int] | None) -> nn.Linear:
    w = old.weight.detach()
    rows = list(range(w.shape[0])) if keep_rows is None else list(keep_rows)
    cols = list(range(w.shape[1])) if keep_cols is None else list(keep_cols)
    ri = torch.tensor(rows, dtype=torch.long, device=w.device)
    ci = torch.tensor(cols, dtype=torch.long, device=w.device)
    new = nn.Linear(len(cols), len(rows), bias=old.bias is not None).to(
        device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new.weight.copy_(w.index_select(0, ri).index_select(1, ci))
        if old.bias is not None:
            new.bias.copy_(old.bias.detach().index_select(0, ri))
    return new


def _set_first(parent: nn.Module, names: tuple[str, ...], value: nn.Module) -> str:
    for name in names:
        if hasattr(parent, name):
            setattr(parent, name, value)
            return name
    raise AttributeError(f"none of {names} exists on {type(parent).__name__}")


def _slice_norm(norm: nn.Module, keep: List[int]) -> None:
    """Shrink a post-projection Q/K norm used by InternVL's vision tower."""
    if isinstance(norm, nn.Identity):
        return
    if not hasattr(norm, "weight") or norm.weight is None:
        raise TypeError(f"cannot structurally slice {type(norm).__name__}")
    idx = torch.tensor(keep, dtype=torch.long, device=norm.weight.device)
    with torch.no_grad():
        norm.weight = nn.Parameter(norm.weight.detach().index_select(0, idx).clone(),
                                   requires_grad=norm.weight.requires_grad)
        if getattr(norm, "bias", None) is not None:
            norm.bias = nn.Parameter(norm.bias.detach().index_select(0, idx).clone(),
                                     requires_grad=norm.bias.requires_grad)
    for name in ("normalized_shape", "hidden_size", "dim"):
        if hasattr(norm, name):
            old = getattr(norm, name)
            setattr(norm, name, (len(keep),) if isinstance(old, tuple) else len(keep))


def _units_by(reg: VLMRegistry) -> Dict[tuple[str, int, str], List[VLMUnit]]:
    out: Dict[tuple[str, int, str], List[VLMUnit]] = {}
    for unit in reg.units:
        out.setdefault((unit.tower, unit.layer_idx, unit.unit_type), []).append(unit)
    return out


def apply_structural_prune_inplace(bundle: VLMBundle, reg: VLMRegistry,
                                   removed_unit_ids: Set[int]) -> Dict[str, object]:
    """Physically remove ``removed_unit_ids`` from both VLM towers.

    Attention units are KV-aligned groups in the language tower and individual
    heads in the vision tower.  FFN units are contiguous channel groups.  Every
    operation therefore has an exact hook-mask counterpart.
    """
    grouped = _units_by(reg)
    report: Dict[str, object] = {"removed_units": len(removed_unit_ids), "layers": []}

    for tower_name in ("lm", "vis"):
        tower = bundle.towers[tower_name]
        for layer_idx, block in enumerate(tower.blocks):
            row: Dict[str, object] = {"tower": tower_name, "layer": layer_idx}

            # ---- attention
            specs = grouped.get((tower_name, layer_idx, "attn_head"), [])
            kept_specs = [u for u in specs if u.unit_id not in removed_unit_ids]
            kept_q_heads = sorted({int(h) for u in kept_specs
                                   for h in u.meta["query_head_indices"]})
            if specs and not kept_q_heads:
                raise ValueError(f"mask removes every attention head in {tower_name} layer {layer_idx}")
            if specs and len(kept_q_heads) < tower.num_heads:
                attn = _get(block, tower.attn_path)
                hd = tower.head_dim
                q_rows = [h * hd + d for h in kept_q_heads for d in range(hd)]

                if tower_name == "lm":
                    kept_kv = sorted(int(u.index_within_layer) for u in kept_specs)
                    kv_rows = [h * hd + d for h in kept_kv for d in range(hd)]
                    for name, rows in (("q_proj", q_rows), ("k_proj", kv_rows),
                                       ("v_proj", kv_rows)):
                        if not hasattr(attn, name):
                            raise AttributeError(f"{type(attn).__name__} lacks {name}")
                        setattr(attn, name, _sliced_linear(getattr(attn, name), rows, None))
                    out_name = tower.attn_out_name
                    setattr(attn, out_name,
                            _sliced_linear(getattr(attn, out_name), None, q_rows))
                    groups = len(kept_q_heads) // len(kept_kv)
                    for name, value in (("num_heads", len(kept_q_heads)),
                                        ("num_attention_heads", len(kept_q_heads)),
                                        ("num_key_value_heads", len(kept_kv)),
                                        ("num_key_value_groups", groups)):
                        if hasattr(attn, name):
                            setattr(attn, name, value)
                    row.update(kept_query_heads=len(kept_q_heads),
                               kept_kv_heads=len(kept_kv))
                else:
                    out_name = tower.attn_out_name
                    out_proj = getattr(attn, out_name)
                    if hasattr(attn, "qkv"):
                        width = tower.num_heads * hd
                        keep_rows = (q_rows + [width + i for i in q_rows]
                                     + [2 * width + i for i in q_rows])
                        attn.qkv = _sliced_linear(attn.qkv, keep_rows, None)
                    elif all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj")):
                        for name in ("q_proj", "k_proj", "v_proj"):
                            setattr(attn, name,
                                    _sliced_linear(getattr(attn, name), q_rows, None))
                        # InternVL normalizes the concatenated projected heads.
                        if hasattr(attn, "q_norm"):
                            _slice_norm(attn.q_norm, q_rows)
                        if hasattr(attn, "k_norm"):
                            _slice_norm(attn.k_norm, q_rows)
                    else:
                        raise TypeError(f"unsupported vision attention {type(attn).__name__}")
                    setattr(attn, out_name, _sliced_linear(out_proj, None, q_rows))
                    for name, value in (("num_heads", len(kept_q_heads)),
                                        ("num_attention_heads", len(kept_q_heads)),
                                        ("num_key_value_groups", 1),
                                        ("embed_dim", len(q_rows)),
                                        ("dim", len(q_rows))):
                        if hasattr(attn, name):
                            setattr(attn, name, value)
                    row["kept_query_heads"] = len(kept_q_heads)

            # ---- FFN
            specs = grouped.get((tower_name, layer_idx, "ffn_group"), [])
            # Some VLM FFN widths are not divisible by the configured number of
            # groups.  ``build_registry`` intentionally leaves that short tail
            # outside the removable inventory, so it must remain in the physical
            # model.  Constructing the slice as the union of registered *kept*
            # groups silently discarded those unregistered channels even when
            # ``removed_unit_ids`` was empty.  Match runtime masking exactly by
            # starting from the full width and deleting only channels belonging
            # to explicitly removed units.
            removed_channels = {
                c
                for u in specs
                if u.unit_id in removed_unit_ids
                for c in range(
                    int(u.meta["start_channel"]),
                    int(u.meta["start_channel"]) + int(u.meta["group_size"]),
                )
            }
            kept_channels = [
                c for c in range(tower.intermediate_size) if c not in removed_channels
            ]
            if specs and not kept_channels:
                raise ValueError(f"mask removes every FFN channel in {tower_name} layer {layer_idx}")
            if specs and len(kept_channels) < tower.intermediate_size:
                mlp = _get(block, tower.ffn_path)
                down = getattr(mlp, tower.ffn_out_name)
                setattr(mlp, tower.ffn_out_name,
                        _sliced_linear(down, None, kept_channels))
                if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                    mlp.gate_proj = _sliced_linear(mlp.gate_proj, kept_channels, None)
                    mlp.up_proj = _sliced_linear(mlp.up_proj, kept_channels, None)
                elif hasattr(mlp, "linear_fc1"):
                    mlp.linear_fc1 = _sliced_linear(mlp.linear_fc1, kept_channels, None)
                elif hasattr(mlp, "fc1"):
                    mlp.fc1 = _sliced_linear(mlp.fc1, kept_channels, None)
                else:
                    raise TypeError(f"unsupported VLM MLP {type(mlp).__name__}")
                if hasattr(mlp, "intermediate_size"):
                    mlp.intermediate_size = len(kept_channels)
                row["kept_ffn_channels"] = len(kept_channels)

            report["layers"].append(row)
    return report
