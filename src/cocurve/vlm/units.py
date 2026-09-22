"""Removable units of a VLM, over both towers, in one flat inventory.

The point of running the method on a VLM is that the inventory is no longer one stack: attention
and FFN units of the language tower and of the vision tower all draw on a single budget, and the
off-diagonal of the risk Hessian is defined between any two of them -- including a vision unit and
a language unit, which no per-tower or per-module scheme can score against each other at all.
Costs are removable parameter counts, so they are comparable across towers by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from .bundle import VLMBundle


@dataclass
class VLMUnit:
    unit_id: int
    tower: str            # "lm" | "vis"
    layer_idx: int
    unit_type: str        # "attn_head" | "ffn_group"
    index_within_layer: int
    cost: float
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.tower}.{self.unit_type}.L{self.layer_idx}.{self.index_within_layer}"


@dataclass
class VLMRegistry:
    units: List[VLMUnit]

    @property
    def n(self) -> int:
        return len(self.units)

    def by(self, tower: str = None, unit_type: str = None) -> List[VLMUnit]:
        return [u for u in self.units
                if (tower is None or u.tower == tower) and (unit_type is None or u.unit_type == unit_type)]

    def cost_vector(self) -> torch.Tensor:
        return torch.tensor([u.cost for u in self.units], dtype=torch.float32)

    def total_cost(self) -> float:
        return float(sum(u.cost for u in self.units))


def build_registry(bundle: VLMBundle, ffn_groups_per_layer: int = 16) -> VLMRegistry:
    units: List[VLMUnit] = []
    uid = 0
    for tname in ("lm", "vis"):
        t = bundle.towers[tname]
        kv_group = max(1, t.num_heads // max(1, t.kv_heads))
        gsize = t.intermediate_size // ffn_groups_per_layer
        for L in range(t.num_layers):
            for g in range(t.kv_heads):
                q0 = g * kv_group
                q_idx = list(range(q0, min(t.num_heads, q0 + kv_group)))
                # q rows + shared k,v rows + o columns, in removable parameters
                cost = t.hidden_size * t.head_dim * (2 * len(q_idx) + 2)
                units.append(VLMUnit(uid, tname, L, "attn_head", g, float(cost),
                                     {"query_head_indices": q_idx, "kv_group_size": kv_group}))
                uid += 1
            for g in range(ffn_groups_per_layer):
                # two matrices for a plain MLP (fc1/fc2), three for a gated one
                t_gated = t.ffn_out_name == "down_proj"
                cost = gsize * t.hidden_size * (3 if t_gated else 2)
                units.append(VLMUnit(uid, tname, L, "ffn_group", g, float(cost),
                                     {"start_channel": g * gsize, "group_size": gsize}))
                uid += 1
    return VLMRegistry(units=units)


def summarise(reg: VLMRegistry) -> str:
    lines = []
    tot = reg.total_cost()
    for tname in ("lm", "vis"):
        for ut in ("attn_head", "ffn_group"):
            sel = reg.by(tname, ut)
            if not sel:
                continue
            c = sum(u.cost for u in sel)
            lines.append(f"  {tname:3s} {ut:10s} n={len(sel):5d}  cost={c/1e6:9.1f}M  ({100*c/tot:5.1f}%)")
    lines.append(f"  total units={reg.n}  removable cost={tot/1e6:.1f}M")
    return "\n".join(lines)
