"""Edge-derived coupling-aware compensation (one-shot, training-free).

Removing a unit set S perturbs the logits by ~= sum_{u in S} dz_u, where dz_u is the
single-unit ablation logit-perturbation whose Fisher-Gram is the edge matrix H
(H_uv = <dz_u, dz_v>_F). We cancel this by *rescaling* the surviving FFN units: scaling a
kept unit v's output by (1+c_v) adds c_v * dz_v. Least-squares in the Fisher metric gives

    (H_KK + gamma I) c = H_KS 1_S        (K = kept FFN compensators, S = removed units)

so the SAME off-diagonal edges that decide what to co-prune also prescribe, in closed
form, how the survivors compensate. No extra calibration (reuses H), no weight-space
reconstruction, no iteration. Apply the returned gains via register_runtime_masks(...,
ffn_gains=gains). Pair with an apply-if-helps gate (keep only if held-out PPL drops).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def compute_ffn_compensation_gains(
    H: np.ndarray,
    units: Sequence,
    selected_unit_ids: set,
    gamma: float = 1.0,
    clip: float = 0.35,
) -> Dict[int, float]:
    """Return {kept FFN unit_id -> multiplicative gain (1+c_v)}.

    H rows/cols are aligned with `units` order (registry order). Compensators are kept
    FFN units; targets are all removed units (cross-module coupling flows through H_KS).
    """
    H = np.asarray(H, dtype=np.float64)
    id2idx = {u.unit_id: i for i, u in enumerate(units)}
    kept_ffn = [i for i, u in enumerate(units)
                if u.unit_id in selected_unit_ids and u.unit_type != "attn_head"]
    removed = [i for i, u in enumerate(units) if u.unit_id not in selected_unit_ids]
    if not kept_ffn or not removed:
        return {}
    HKK = H[np.ix_(kept_ffn, kept_ffn)]
    rhs = H[np.ix_(kept_ffn, removed)].sum(axis=1)
    reg = gamma * float(np.mean(np.diag(HKK)))
    c = np.linalg.solve(HKK + reg * np.eye(len(kept_ffn)), rhs)
    c = np.clip(c, -clip, clip)
    return {units[kept_ffn[j]].unit_id: 1.0 + float(c[j]) for j in range(len(kept_ffn))}
