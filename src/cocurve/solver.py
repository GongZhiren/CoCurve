from __future__ import annotations

from typing import Dict, List, Sequence, Set

import numpy as np

from .types import PruneResult


def greedy_budget_prune(
    h: np.ndarray,
    costs: np.ndarray,
    prune_ratio: float,
    allow_budget_overshoot: bool = True,
    normalize_by_cost: bool = True,
    unit_layers: Sequence[int] | None = None,
    max_pruned_cost_fraction_per_layer: float | None = None,
    protected_layers: Set[int] | None = None,
    interaction_strength: float = 1.0,
) -> PruneResult:
    # interaction_strength (lambda) scales the signed off-diagonal term relative
    # to each unit's diagonal curvature.  The publication path selects lambda by
    # held-out teacher KL; lambda=0 is the diagonal endpoint and lambda=1 uses the
    # unscaled Fisher--Gram cross term.
    m = h.shape[0]
    protected: Set[int] = set(int(x) for x in protected_layers) if protected_layers else set()
    total_cost = float(costs.sum())
    target_pruned_cost = total_cost * prune_ratio
    layer_costs: Dict[int, float] = {}
    layer_pruned_costs: Dict[int, float] = {}
    if unit_layers is not None:
        if len(unit_layers) != m:
            raise ValueError(f"unit_layers length {len(unit_layers)} does not match H dimension {m}")
        for unit_id, layer_idx in enumerate(unit_layers):
            layer = int(layer_idx)
            layer_costs[layer] = layer_costs.get(layer, 0.0) + float(costs[unit_id])
            layer_pruned_costs.setdefault(layer, 0.0)

    selected: Set[int] = set()
    current_pruned_cost = 0.0
    score_trace: List[Dict[str, float]] = []

    interaction_sum = np.zeros((m,), dtype=np.float64)
    while current_pruned_cost < target_pruned_cost and len(selected) < m:
        candidates_list = []
        for unit_id in range(m):
            if unit_id in selected:
                continue
            if protected and unit_layers is not None and int(unit_layers[unit_id]) in protected:
                continue
            if unit_layers is not None and max_pruned_cost_fraction_per_layer is not None:
                layer = int(unit_layers[unit_id])
                layer_budget = layer_costs[layer] * float(max_pruned_cost_fraction_per_layer)
                next_layer_cost = layer_pruned_costs[layer] + float(costs[unit_id])
                if next_layer_cost > layer_budget:
                    continue
            candidates_list.append(unit_id)
        if not candidates_list:
            break
        candidates = np.array(candidates_list, dtype=np.int32)
        deltas = 0.5 * h[candidates, candidates] + float(interaction_strength) * interaction_sum[candidates]
        if normalize_by_cost:
            scores = deltas / costs[candidates].clip(min=1e-12)
        else:
            scores = deltas

        best_idx = int(np.argmin(scores))
        best_u = int(candidates[best_idx])
        next_cost = current_pruned_cost + float(costs[best_u])
        if (not allow_budget_overshoot) and next_cost > target_pruned_cost:
            break

        selected.add(best_u)
        interaction_sum += h[:, best_u]
        current_pruned_cost = next_cost
        if unit_layers is not None:
            layer = int(unit_layers[best_u])
            layer_pruned_costs[layer] += float(costs[best_u])
        score_trace.append(
            {
                "step": float(len(selected)),
                "unit_id": float(best_u),
                "delta": float(deltas[best_idx]),
                "score": float(scores[best_idx]),
                "pruned_cost": current_pruned_cost,
            }
        )

    pruned_units = sorted(selected)
    kept_units = sorted(set(range(m)) - selected)
    actual_ratio = current_pruned_cost / max(total_cost, 1e-12)
    overshoot = max(0.0, current_pruned_cost - target_pruned_cost)
    return PruneResult(
        selected_units=kept_units,
        pruned_units=pruned_units,
        total_cost=total_cost,
        pruned_cost=current_pruned_cost,
        target_pruned_cost=target_pruned_cost,
        target_prune_ratio=prune_ratio,
        actual_prune_ratio=actual_ratio,
        overshoot_cost=overshoot,
        score_trace=score_trace,
    )
