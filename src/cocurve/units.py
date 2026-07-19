from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .types import UnitSpec


@dataclass
class UnitRegistry:
    units: List[UnitSpec]
    attn_units_by_layer: Dict[int, List[int]]
    ffn_units_by_layer: Dict[int, List[int]]
    ffn_group_size: int

    @property
    def num_units(self) -> int:
        return len(self.units)


def _head_cost(hidden_size: int, head_dim: int, q_group_size: int, use_params: bool) -> float:
    # Q for each query head + one shared K/V head + O columns for each query head.
    params = hidden_size * head_dim * q_group_size  # q_proj rows
    params += hidden_size * head_dim * 2  # shared k_proj + v_proj
    params += hidden_size * head_dim * q_group_size  # o_proj columns
    if use_params:
        return float(params)
    # FLOPs proxy for projection matmuls, same units as params proxy.
    return float(params * 2)


def _ffn_group_cost(hidden_size: int, group_size: int, use_params: bool) -> float:
    if use_params:
        return float(group_size * hidden_size * 3)
    return float(group_size * hidden_size * 2)


def build_unit_registry(
    num_layers: int,
    num_heads: int,
    hidden_size: int,
    intermediate_size: int,
    ffn_groups_per_layer: int,
    cost_type: str,
    kv_heads: int,
    head_dim: int | None = None,
) -> UnitRegistry:
    units: List[UnitSpec] = []
    attn_by_layer: Dict[int, List[int]] = {}
    ffn_by_layer: Dict[int, List[int]] = {}

    use_params = cost_type == "params"
    # Most models have head_dim == hidden_size // num_heads, but some (e.g. Gemma-2,
    # Mistral-Nemo) decouple it (Gemma-2-9B: head_dim=256, hidden//heads=224). Pass the
    # real bundle.head_dim so attention-unit cost is correct; fall back when unknown.
    if head_dim is None:
        head_dim = hidden_size // num_heads
    ffn_group_size = intermediate_size // ffn_groups_per_layer
    unit_id = 0

    # GQA mode: default KV-aligned grouping. query heads sharing one KV head are grouped.
    kv_group = max(1, num_heads // max(1, kv_heads))

    for layer_idx in range(num_layers):
        attn_ids: List[int] = []
        for head_group_idx in range(kv_heads):
            q_start = head_group_idx * kv_group
            q_indices = list(range(q_start, min(num_heads, q_start + kv_group)))
            cost = _head_cost(hidden_size, head_dim, len(q_indices), use_params)
            spec = UnitSpec(
                unit_id=unit_id,
                layer_idx=layer_idx,
                unit_type="attn_head",
                index_within_layer=head_group_idx,
                cost=cost,
                metadata={
                    "kv_group_size": kv_group,
                    "num_heads": num_heads,
                    "num_kv_heads": kv_heads,
                    "kv_head_index": head_group_idx,
                    "query_head_indices": q_indices,
                },
            )
            units.append(spec)
            attn_ids.append(unit_id)
            unit_id += 1
        attn_by_layer[layer_idx] = attn_ids

        ffn_ids: List[int] = []
        for group_idx in range(ffn_groups_per_layer):
            cost = _ffn_group_cost(hidden_size, ffn_group_size, use_params)
            spec = UnitSpec(
                unit_id=unit_id,
                layer_idx=layer_idx,
                unit_type="ffn_group",
                index_within_layer=group_idx,
                cost=cost,
                metadata={"start_channel": group_idx * ffn_group_size, "group_size": ffn_group_size},
            )
            units.append(spec)
            ffn_ids.append(unit_id)
            unit_id += 1
        ffn_by_layer[layer_idx] = ffn_ids

    return UnitRegistry(units=units, attn_units_by_layer=attn_by_layer, ffn_units_by_layer=ffn_by_layer, ffn_group_size=ffn_group_size)


def unit_cost_vector(registry: UnitRegistry) -> torch.Tensor:
    return torch.tensor([u.cost for u in registry.units], dtype=torch.float32)


def unit_map_by_id(registry: UnitRegistry) -> Dict[int, UnitSpec]:
    return {u.unit_id: u for u in registry.units}
