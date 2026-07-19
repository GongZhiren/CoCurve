from __future__ import annotations

import json
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .calibration import load_calibration_texts, tokenize_batch
from .io import ArtifactStore
from .model import (
    ModelBundle,
    bundle_device,
    capture_single_layer_input,
    forward_from_layer,
    forward_logits,
    kl_from_top_r,
    supports_layer_replay,
)
from .prune import clear_runtime_masks, register_runtime_masks
from .types import UnitSpec


def _layer_maps(units: List[UnitSpec]) -> Tuple[Dict[int, List[UnitSpec]], Dict[int, List[UnitSpec]]]:
    attn: Dict[int, List[UnitSpec]] = {}
    ffn: Dict[int, List[UnitSpec]] = {}
    for spec in units:
        if spec.unit_type == "attn_head":
            attn.setdefault(spec.layer_idx, []).append(spec)
        else:
            ffn.setdefault(spec.layer_idx, []).append(spec)
    return attn, ffn


def _token_level_delta_feature(
    teacher_top_logits: torch.Tensor,
    teacher_top_probs: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_top_indices: torch.Tensor,
) -> torch.Tensor:
    student_top_logits = torch.gather(student_logits[:, :-1, :], dim=-1, index=teacher_top_indices)
    delta = teacher_top_logits - student_top_logits
    centered = delta - (teacher_top_probs * delta).sum(dim=-1, keepdim=True)
    weighted = torch.sqrt(teacher_top_probs.clamp_min(1e-12)) * centered
    return weighted


def run_single_unit_ablations(
    bundle: ModelBundle,
    cfg: Dict[str, object],
    store: ArtifactStore,
    units: List[UnitSpec],
) -> Tuple[np.ndarray, np.ndarray]:
    texts = load_calibration_texts(cfg["calibration"])
    batch_size = int(cfg["calibration"]["calibration"]["batch_size"])
    max_seq_len = int(cfg["calibration"]["calibration"]["max_seq_len"])

    device = bundle_device(bundle)
    teacher_top_indices = torch.from_numpy(np.load(store.run_dir / "cache/top_indices.npy")).to(device)
    teacher_top_probs = torch.from_numpy(np.load(store.run_dir / "cache/top_probs.npy")).to(device)
    teacher_top_logits = torch.from_numpy(np.load(store.run_dir / "cache/top_logits.npy")).to(device)
    token_positions = torch.from_numpy(np.load(store.run_dir / "cache/token_positions.npy")).to(device)

    num_samples = int(teacher_top_indices.shape[0])
    top_r = teacher_top_indices.shape[-1]
    p_total = int(token_positions.sum().item())

    replay_cfg = cfg["pruning"].get("ablation_acceleration", {})
    enable_layer_replay = bool(replay_cfg.get("enable_layer_replay", True))
    use_layer_replay = enable_layer_replay and supports_layer_replay(bundle)

    feature_path = store.run_dir / "cache/unit_features.memmap"
    progress_path = store.run_dir / "cache/ablation_progress.json"
    partial_kl_path = store.run_dir / "cache/single_unit_kl_partial.npy"
    expected_shape = [len(units), p_total, int(top_r)]

    completed_units: Set[int] = set()
    unit_stats: List[Dict[str, object]] = []
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            progress = json.load(f)
        if progress.get("shape") == expected_shape:
            completed_units = {int(u) for u in progress.get("completed_units", [])}
            unit_stats = list(progress.get("unit_stats", []))

    mode = "r+" if completed_units and feature_path.exists() else "w+"
    features = np.memmap(feature_path, dtype=np.float32, mode=mode, shape=(len(units), p_total, top_r))
    if completed_units and partial_kl_path.exists():
        single_unit_kl = np.load(partial_kl_path).astype(np.float32)
    else:
        single_unit_kl = np.zeros((len(units),), dtype=np.float32)

    attn_by_layer, ffn_by_layer = _layer_maps(units)
    all_unit_ids: Set[int] = {u.unit_id for u in units}

    # Pre-tokenize calibration batches once; this removes repeated tokenizer overhead.
    prepared_batches: List[Dict[str, torch.Tensor]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        prepared_batches.append(tokenize_batch(bundle, batch_texts, max_seq_len=max_seq_len))
    total_started = time.perf_counter()

    units_by_layer: Dict[int, List[UnitSpec]] = {}
    for spec in units:
        units_by_layer.setdefault(spec.layer_idx, []).append(spec)

    def save_progress(stage: str) -> None:
        features.flush()
        store.save_numpy("cache/single_unit_kl_partial.npy", single_unit_kl)
        store.save_json(
            "cache/ablation_progress.json",
            {
                "stage": stage,
                "shape": expected_shape,
                "completed_units": sorted(int(u) for u in completed_units),
                "num_completed_units": len(completed_units),
                "num_total_units": len(units),
                "num_selected_tokens": p_total,
                "layer_replay_enabled": bool(use_layer_replay),
                "unit_stats": unit_stats,
                "updated_at_unix": time.time(),
            },
        )

    if not use_layer_replay:
        for spec in tqdm(units, desc="single-unit-ablation"):
            if spec.unit_id in completed_units:
                continue
            unit_started = time.perf_counter()
            kept = set(all_unit_ids)
            kept.remove(spec.unit_id)
            mask_state = register_runtime_masks(
                bundle=bundle,
                units_by_layer=attn_by_layer,
                ffn_by_layer=ffn_by_layer,
                selected_unit_ids=kept,
            )
            row_offset = 0
            token_offset = 0
            kl_values: List[torch.Tensor] = []
            try:
                for batch in prepared_batches:
                    logits = forward_logits(bundle, batch["input_ids"], batch["attention_mask"])
                    bsz = int(logits.shape[0])
                    t_idx = teacher_top_indices[row_offset : row_offset + bsz]
                    t_prob = teacher_top_probs[row_offset : row_offset + bsz]
                    t_logit = teacher_top_logits[row_offset : row_offset + bsz]
                    t_mask = token_positions[row_offset : row_offset + bsz].float()

                    feat = _token_level_delta_feature(t_logit, t_prob, logits, t_idx)
                    selected = feat[t_mask > 0]
                    selected_np = selected.detach().float().cpu().numpy()
                    next_token_offset = token_offset + selected_np.shape[0]
                    features[spec.unit_id, token_offset:next_token_offset, :] = selected_np
                    token_offset = next_token_offset

                    kl = kl_from_top_r(t_idx, t_prob, logits[:, :-1, :]) * t_mask
                    kl_values.append(kl.sum(dim=1) / t_mask.sum(dim=1).clamp_min(1.0))
                    row_offset += bsz

                    del logits, feat, selected
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            finally:
                clear_runtime_masks(mask_state)

            if token_offset != p_total:
                raise RuntimeError(f"Token feature count mismatch for unit {spec.unit_id}: {token_offset} != {p_total}")
            single_unit_kl[spec.unit_id] = torch.cat(kl_values).mean().item()
            unit_stats.append(
                {
                    "unit_id": int(spec.unit_id),
                    "layer_idx": int(spec.layer_idx),
                    "unit_type": spec.unit_type,
                    "kl_mean": float(single_unit_kl[spec.unit_id]),
                    "selected_tokens_written": int(token_offset),
                    "replay_batches": 0,
                    "full_forward_batches": int(len(prepared_batches)),
                    "elapsed_sec": round(time.perf_counter() - unit_started, 6),
                }
            )
            completed_units.add(spec.unit_id)
            save_progress(stage="running")

        save_progress(stage="completed")
        store.save_json(
            "cache/unit_features_meta.json",
            {
                "path": str(feature_path),
                "shape": expected_shape,
                "dtype": "float32",
                "num_samples": num_samples,
                "num_selected_tokens": p_total,
                "layer_replay_enabled": bool(use_layer_replay),
            },
        )
        store.save_numpy("cache/single_unit_kl.npy", single_unit_kl)
        store.save_json(
            "logs/ablation_summary.json",
            {
                "num_units": len(units),
                "num_samples": num_samples,
                "num_selected_tokens": p_total,
                "layer_replay_enabled": bool(use_layer_replay),
                "completed_units": len(completed_units),
                "elapsed_sec": round(time.perf_counter() - total_started, 6),
            },
        )
        store.save_json("logs/ablation_unit_stats.json", {"units": unit_stats})
        return np.asarray(features), single_unit_kl

    token_offsets: Dict[int, int] = {spec.unit_id: 0 for spec in units}
    kl_values_by_unit: Dict[int, List[torch.Tensor]] = {spec.unit_id: [] for spec in units}
    replay_batches_by_unit: Dict[int, int] = {spec.unit_id: 0 for spec in units}
    full_forward_by_unit: Dict[int, int] = {spec.unit_id: 0 for spec in units}
    per_unit_started: Dict[int, float] = {spec.unit_id: time.perf_counter() for spec in units}

    for layer_idx in tqdm(sorted(units_by_layer.keys()), desc="ablation-layer-groups"):
        layer_units = units_by_layer[layer_idx]
        row_offset = 0
        for batch in prepared_batches:
            bsz = int(batch["input_ids"].shape[0])
            t_idx = teacher_top_indices[row_offset : row_offset + bsz]
            t_prob = teacher_top_probs[row_offset : row_offset + bsz]
            t_logit = teacher_top_logits[row_offset : row_offset + bsz]
            t_mask = token_positions[row_offset : row_offset + bsz].float()
            replay_hidden = None
            if use_layer_replay:
                replay_hidden = capture_single_layer_input(bundle, batch["input_ids"], batch["attention_mask"], layer_idx)

            for spec in layer_units:
                kept = set(all_unit_ids)
                kept.remove(spec.unit_id)
                mask_state = register_runtime_masks(
                    bundle=bundle,
                    units_by_layer=attn_by_layer,
                    ffn_by_layer=ffn_by_layer,
                    selected_unit_ids=kept,
                )
                if replay_hidden is not None:
                    logits = forward_from_layer(
                        bundle=bundle,
                        start_layer_idx=spec.layer_idx,
                        hidden_states=replay_hidden,
                        attention_mask=batch["attention_mask"],
                    )
                    replay_batches_by_unit[spec.unit_id] += 1
                else:
                    logits = forward_logits(bundle, batch["input_ids"], batch["attention_mask"])
                    full_forward_by_unit[spec.unit_id] += 1
                clear_runtime_masks(mask_state)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                feat = _token_level_delta_feature(t_logit, t_prob, logits, t_idx)
                selected = feat[t_mask > 0]  # [num_selected_tokens, top_r]
                selected_np = selected.detach().float().cpu().numpy()
                token_offset = token_offsets[spec.unit_id]
                next_token_offset = token_offset + selected_np.shape[0]
                features[spec.unit_id, token_offset:next_token_offset, :] = selected_np
                token_offsets[spec.unit_id] = next_token_offset

                kl = kl_from_top_r(t_idx, t_prob, logits[:, :-1, :]) * t_mask
                kl_values_by_unit[spec.unit_id].append(kl.sum(dim=1) / t_mask.sum(dim=1).clamp_min(1.0))
            row_offset += bsz

    for spec in units:
        token_offset = token_offsets[spec.unit_id]
        if token_offset != p_total:
            raise RuntimeError(f"Token feature count mismatch for unit {spec.unit_id}: {token_offset} != {p_total}")
        single_unit_kl[spec.unit_id] = torch.cat(kl_values_by_unit[spec.unit_id]).mean().item()
        unit_stats.append(
            {
                "unit_id": int(spec.unit_id),
                "layer_idx": int(spec.layer_idx),
                "unit_type": spec.unit_type,
                "kl_mean": float(single_unit_kl[spec.unit_id]),
                "selected_tokens_written": int(token_offset),
                "replay_batches": int(replay_batches_by_unit[spec.unit_id]),
                "full_forward_batches": int(full_forward_by_unit[spec.unit_id]),
                "elapsed_sec": round(time.perf_counter() - per_unit_started[spec.unit_id], 6),
            }
        )

    store.save_json(
        "cache/unit_features_meta.json",
        {
            "path": str(feature_path),
            "shape": [len(units), p_total, int(top_r)],
            "dtype": "float32",
            "num_samples": num_samples,
            "num_selected_tokens": p_total,
            "layer_replay_enabled": bool(use_layer_replay),
        },
    )
    store.save_numpy("cache/single_unit_kl.npy", single_unit_kl)
    store.save_json(
        "logs/ablation_summary.json",
        {
            "num_units": len(units),
            "num_samples": num_samples,
            "num_selected_tokens": p_total,
            "layer_replay_enabled": bool(use_layer_replay),
            "elapsed_sec": round(time.perf_counter() - total_started, 6),
        },
    )
    store.save_json("logs/ablation_unit_stats.json", {"units": unit_stats})
    features.flush()
    return np.asarray(features), single_unit_kl
