from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .io import ArtifactStore
from .model import ModelBundle, bundle_device, forward_logits, top_r_distribution


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_multiple_choice_prompt(text: str) -> bool:
    upper = text.upper()
    markers = (" A.", " A)", "\nA.", "\nA)", "CHOICES:", "OPTION A")
    return any(marker in upper for marker in markers)


def _load_manifest(calibration_cfg: Dict[str, object]) -> Dict[str, Any]:
    cfg = calibration_cfg["calibration"]
    manifest_path = cfg.get("manifest_path")
    require_manifest = bool(cfg.get("require_manifest", False))
    if not manifest_path:
        if require_manifest:
            raise FileNotFoundError("Calibration manifest is required but manifest_path is not set.")
        return {}
    path = Path(str(manifest_path))
    if not path.exists():
        if require_manifest:
            raise FileNotFoundError(f"Calibration manifest is required but missing: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    allowed_sources = set(cfg.get("allowed_sources", []))
    selected = manifest.get("selected_source", {})
    source = selected.get("source")
    if allowed_sources and source not in allowed_sources:
        raise ValueError(f"Unsupported calibration source={source!r}; expected one of {sorted(allowed_sources)}")
    if manifest.get("purpose") != "LLM pruning calibration, not evaluation":
        raise ValueError("Calibration manifest purpose is missing or unexpected.")
    return manifest


def calibration_signature(calibration_cfg: Dict[str, object]) -> Dict[str, Any]:
    cfg = calibration_cfg["calibration"]
    dataset_mix = cfg["dataset_mix"]
    datasets: List[Dict[str, Any]] = []
    for entry in dataset_mix:
        path = Path(str(entry["path"]))
        datasets.append(
            {
                "source": entry.get("source"),
                "path": str(path),
                "weight": entry.get("weight"),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": _sha256_file(path) if path.exists() else None,
            }
        )
    manifest_path = cfg.get("manifest_path")
    manifest = None
    if manifest_path:
        path = Path(str(manifest_path))
        manifest = {
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha256_file(path) if path.exists() else None,
        }
    holdout_path = cfg.get("holdout_path")
    holdout = None
    if holdout_path:
        path = Path(str(holdout_path))
        holdout = {
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha256_file(path) if path.exists() else None,
        }
    return {
        "max_samples": int(cfg["max_samples"]),
        "max_seq_len": int(cfg["max_seq_len"]),
        "top_r_logits": int(cfg["top_r_logits"]),
        "tokens_per_sequence": int(cfg["tokens_per_sequence"]),
        "dataset_mix": datasets,
        "manifest": manifest,
        "holdout": holdout,
    }


def load_calibration_texts(calibration_cfg: Dict[str, object]) -> List[str]:
    manifest = _load_manifest(calibration_cfg)
    dataset_mix = calibration_cfg["calibration"]["dataset_mix"]
    texts: List[str] = []
    missing: List[str] = []
    for entry in dataset_mix:
        source = entry["source"]
        path = Path(entry["path"])
        if source != "jsonl":
            raise ValueError(f"Unsupported calibration source: {source}")
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = row.get("text", "")
                if text:
                    texts.append(text)
    if missing:
        raise FileNotFoundError(f"Missing calibration sources: {missing}")
    max_samples = int(calibration_cfg["calibration"]["max_samples"])
    result = texts[:max_samples]
    if len(result) < max(8, min(max_samples, 32)):
        raise ValueError(
            f"Calibration text is too small ({len(result)}). "
            "Please provide a non-placeholder mixed-domain calibration dataset."
        )
    cfg = calibration_cfg["calibration"]
    max_mc_fraction = float(cfg.get("max_multiple_choice_fraction", 1.0))
    if bool(cfg.get("reject_multiple_choice_prompts", False)):
        mc_count = sum(1 for text in result if _looks_like_multiple_choice_prompt(text))
        mc_fraction = mc_count / max(1, len(result))
        if mc_fraction > max_mc_fraction:
            raise ValueError(
                f"Calibration set appears prompt/MC-biased: {mc_count}/{len(result)} "
                f"({mc_fraction:.1%}) look like multiple-choice prompts. "
                "Rebuild calibration from standard LM text such as C4 or WikiText train."
            )
    manifest_samples = int(manifest.get("num_samples", len(result))) if manifest else len(result)
    if manifest_samples < max_samples:
        raise ValueError(f"Calibration manifest declares only {manifest_samples} samples; need {max_samples}.")
    return result


def tokenize_batch(bundle: ModelBundle, texts: List[str], max_seq_len: int) -> Dict[str, torch.Tensor]:
    encoded = bundle.tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )
    device = bundle_device(bundle)
    return {k: v.to(device) for k, v in encoded.items()}


def _subsample_token_positions(
    shifted_mask: torch.Tensor,
    tokens_per_sequence: int,
    seed: int,
) -> torch.Tensor:
    # shifted_mask shape: [B, T-1], positions already aligned for next-token KL.
    out = torch.zeros_like(shifted_mask, dtype=torch.int32)
    rng = np.random.default_rng(seed)
    for b in range(shifted_mask.shape[0]):
        valid = torch.where(shifted_mask[b] > 0)[0].cpu().numpy()
        if valid.size == 0:
            continue
        k = min(tokens_per_sequence, int(valid.size))
        chosen = np.sort(rng.choice(valid, size=k, replace=False))
        out[b, chosen] = 1
    return out


def run_full_model_collection(
    bundle: ModelBundle,
    cfg: Dict[str, object],
    store: ArtifactStore,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    texts = load_calibration_texts(cfg["calibration"])
    if not texts:
        raise ValueError("No calibration texts found in data/calibration/calibration_mix.jsonl")

    batch_size = int(cfg["calibration"]["calibration"]["batch_size"])
    max_seq_len = int(cfg["calibration"]["calibration"]["max_seq_len"])
    top_r = int(cfg["calibration"]["calibration"]["top_r_logits"])
    tokens_per_sequence = int(cfg["calibration"]["calibration"]["tokens_per_sequence"])
    seed = int(cfg["project"]["seed"])

    all_top_idx: List[np.ndarray] = []
    all_top_prob: List[np.ndarray] = []
    all_positions: List[np.ndarray] = []
    all_top_logits: List[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="collect-full"):
        batch_texts = texts[start : start + batch_size]
        batch = tokenize_batch(bundle, batch_texts, max_seq_len=max_seq_len)
        logits = forward_logits(bundle, batch["input_ids"], batch["attention_mask"])
        shifted_logits = logits[:, :-1, :]
        shifted_mask = batch["attention_mask"][:, 1:]
        sampled_mask = _subsample_token_positions(shifted_mask, tokens_per_sequence=tokens_per_sequence, seed=seed + start)
        top_idx, top_prob = top_r_distribution(shifted_logits, top_r=top_r)

        all_top_idx.append(top_idx.detach().cpu().numpy())
        all_top_prob.append(top_prob.detach().float().cpu().numpy())
        selected_logits = torch.gather(shifted_logits, dim=-1, index=top_idx)
        all_positions.append(sampled_mask.detach().cpu().numpy())
        all_top_logits.append(selected_logits.detach().float().cpu().numpy())

    top_idx_np = np.concatenate(all_top_idx, axis=0)
    top_prob_np = np.concatenate(all_top_prob, axis=0)
    positions_np = np.concatenate(all_positions, axis=0).astype(np.int32)
    top_logits_np = np.concatenate(all_top_logits, axis=0).astype(np.float32)

    store.save_numpy("cache/top_indices.npy", top_idx_np)
    store.save_numpy("cache/top_probs.npy", top_prob_np)
    store.save_numpy("cache/top_logits.npy", top_logits_np)
    store.save_numpy("cache/token_positions.npy", positions_np)
    store.save_json(
        "cache/calibration_meta.json",
        {
            "num_samples": int(top_idx_np.shape[0]),
            "seq_len_minus_one": int(top_idx_np.shape[1]),
            "top_r": int(top_idx_np.shape[2]),
            "selected_token_count": int(positions_np.sum()),
            "tokens_per_sequence": tokens_per_sequence,
            "calibration_signature": calibration_signature(cfg["calibration"]),
        },
    )

    return top_idx_np, top_prob_np, positions_np
