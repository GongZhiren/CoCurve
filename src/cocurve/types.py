from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

import numpy as np


UnitType = Literal["attn_head", "ffn_group"]


@dataclass(frozen=True)
class UnitSpec:
    unit_id: int
    layer_idx: int
    unit_type: UnitType
    index_within_layer: int
    cost: float
    metadata: Dict[str, Any]


@dataclass
class RunContext:
    model_key: str
    run_name: str
    run_dir: str
    seed: int


@dataclass
class CalibrationBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray


@dataclass
class FullPassCache:
    token_positions: np.ndarray
    top_indices: np.ndarray
    top_probs: np.ndarray
    logits_shape: List[int]


@dataclass
class PruneResult:
    selected_units: List[int]
    pruned_units: List[int]
    total_cost: float
    pruned_cost: float
    target_pruned_cost: float
    target_prune_ratio: float
    actual_prune_ratio: float
    overshoot_cost: float
    score_trace: List[Dict[str, float]]


@dataclass
class QualityGateReport:
    surrogate_vs_real_spearman: float
    surrogate_vs_real_pearson: float
    diagonal_damage_spearman: float
    mask_vs_physical_max_abs_err: float
    passed: bool
