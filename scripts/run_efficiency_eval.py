#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset

from cocurve.config import load_config, model_config
from cocurve.model import ModelBundle, bundle_device, load_model_bundle
from cocurve.prune import (
    apply_physical_prune_inplace,
    apply_structural_prune_inplace,
    clear_runtime_masks,
    register_runtime_masks,
)
from cocurve.units import build_unit_registry, unit_cost_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure inference efficiency for full, masked, or physically-zeroed models.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mask-run-dir", default=None)
    parser.add_argument(
        "--mode",
        choices=["full", "runtime_mask", "physical_zero", "physical_slice"],
        default="full",
        help="physical_slice = true structural removal (smaller matrices, real speedup).",
    )
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--generate-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    return parser.parse_args()


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_layer_maps(bundle: ModelBundle, cfg: Dict[str, Any]) -> Tuple[Any, Dict[int, List[Any]], Dict[int, List[Any]]]:
    pruning_cfg = cfg["pruning"]
    registry = build_unit_registry(
        num_layers=bundle.num_layers,
        num_heads=bundle.num_heads,
        hidden_size=bundle.hidden_size,
        intermediate_size=bundle.intermediate_size,
        ffn_groups_per_layer=int(pruning_cfg["units"]["ffn_groups_per_layer"]),
        cost_type=str(pruning_cfg["units"]["cost_type"]),
        kv_heads=bundle.kv_heads,
        head_dim=bundle.head_dim,
    )
    attn_by_layer: Dict[int, List[Any]] = {}
    ffn_by_layer: Dict[int, List[Any]] = {}
    for spec in registry.units:
        if spec.unit_type == "attn_head":
            attn_by_layer.setdefault(spec.layer_idx, []).append(spec)
        else:
            ffn_by_layer.setdefault(spec.layer_idx, []).append(spec)
    return registry, attn_by_layer, ffn_by_layer


def load_selected_units(mask_run_dir: Optional[str]) -> Optional[set[int]]:
    if not mask_run_dir:
        return None
    mask_path = Path(mask_run_dir) / "masks/prune_solution.json"
    with mask_path.open("r", encoding="utf-8") as f:
        return {int(x) for x in json.load(f)["selected_units"]}


def load_prompts(cache_dir: str, num_prompts: int) -> List[str]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir)
    prompts = [row["text"] for row in ds if isinstance(row.get("text"), str) and row["text"].strip()]
    return prompts[:num_prompts]


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def memory_report() -> Dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def measure_forward(bundle: ModelBundle, prompts: List[str], max_seq_len: int, warmup: int, repeat: int) -> Dict[str, float]:
    tokenizer = bundle.tokenizer
    device = bundle_device(bundle)
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(device)
    batch.pop("token_type_ids", None)  # Falcon3/Phi tokenizers emit it; forward/generate reject it
    input_tokens = int(batch["attention_mask"].sum().item())
    for _ in range(warmup):
        with torch.no_grad():
            _ = bundle.model(**batch, use_cache=False)
    synchronize()
    reset_peak_memory()
    started = time.perf_counter()
    for _ in range(repeat):
        with torch.no_grad():
            _ = bundle.model(**batch, use_cache=False)
    synchronize()
    elapsed = time.perf_counter() - started
    total_tokens = input_tokens * repeat
    return {
        "batch_size": float(len(prompts)),
        "input_tokens_per_repeat": float(input_tokens),
        "repeat": float(repeat),
        "elapsed_sec": elapsed,
        "tokens_per_sec": total_tokens / max(elapsed, 1e-12),
        **memory_report(),
    }


def measure_generate(
    bundle: ModelBundle,
    prompts: List[str],
    max_seq_len: int,
    max_new_tokens: int,
    warmup: int,
    repeat: int,
) -> Dict[str, float]:
    tokenizer = bundle.tokenizer
    device = bundle_device(bundle)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    prompts = prompts[: max(1, len(prompts))]
    try:
        batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(device)
        batch.pop("token_type_ids", None)  # Falcon3/Phi tokenizers emit it; generate() rejects it
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}
        for _ in range(warmup):
            with torch.no_grad():
                _ = bundle.model.generate(**batch, **kwargs)
        synchronize()
        reset_peak_memory()
        started = time.perf_counter()
        new_tokens = 0
        for _ in range(repeat):
            with torch.no_grad():
                out = bundle.model.generate(**batch, **kwargs)
            new_tokens += int((out.shape[1] - batch["input_ids"].shape[1]) * len(prompts))
        synchronize()
        elapsed = time.perf_counter() - started
        return {
            "batch_size": float(len(prompts)),
            "max_new_tokens": float(max_new_tokens),
            "repeat": float(repeat),
            "elapsed_sec": elapsed,
            "new_tokens_per_sec": new_tokens / max(elapsed, 1e-12),
            **memory_report(),
        }
    finally:
        tokenizer.padding_side = old_padding_side


def param_report(bundle: ModelBundle) -> Dict[str, Any]:
    total = int(sum(p.numel() for p in bundle.model.parameters()))
    embed = 0
    try:
        embed += int(bundle.model.get_input_embeddings().weight.numel())
    except Exception:
        pass
    lm_head = getattr(bundle.model, "lm_head", None)
    tied = getattr(getattr(bundle.model, "config", None), "tie_word_embeddings", False)
    if lm_head is not None and not tied:
        try:
            embed += int(lm_head.weight.numel())
        except Exception:
            pass
    return {"total_params": total, "embedding_params": embed, "non_embedding_params": total - embed}


def mask_cost_report(bundle: ModelBundle, cfg: Dict[str, Any], selected_units: Optional[set[int]]) -> Dict[str, Any]:
    if selected_units is None:
        return {"mask_enabled": False}
    registry, _attn, _ffn = build_layer_maps(bundle, cfg)
    costs = unit_cost_vector(registry).numpy()
    kept_cost = float(sum(float(costs[i]) for i in selected_units))
    total_cost = float(costs.sum())
    pruned_units = [spec for spec in registry.units if spec.unit_id not in selected_units]
    return {
        "mask_enabled": True,
        "num_units": len(registry.units),
        "num_kept_units": len(selected_units),
        "num_pruned_units": len(pruned_units),
        "total_cost": total_cost,
        "kept_cost": kept_cost,
        "pruned_cost": total_cost - kept_cost,
        "nominal_prune_ratio": (total_cost - kept_cost) / max(total_cost, 1e-12),
        "pruned_attn_units": sum(1 for spec in pruned_units if spec.unit_type == "attn_head"),
        "pruned_ffn_units": sum(1 for spec in pruned_units if spec.unit_type == "ffn_group"),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.model_key is not None:
        cfg["run"] = dict(cfg["run"])
        cfg["run"]["model_key"] = args.model_key

    out_dir = Path(args.run_dir) / "efficiency"
    selected_units = load_selected_units(args.mask_run_dir)
    bundle = load_model_bundle(model_config(cfg), cfg["model"].get("tokenizer", {}))
    mask_state = None
    backups = None
    slice_info = None
    try:
        if args.mode in {"runtime_mask", "physical_zero", "physical_slice"}:
            if selected_units is None:
                raise ValueError(f"--mask-run-dir is required for mode={args.mode}")
            _registry, attn_by_layer, ffn_by_layer = build_layer_maps(bundle, cfg)
            if args.mode == "runtime_mask":
                mask_state = register_runtime_masks(bundle, attn_by_layer, ffn_by_layer, selected_units)
            elif args.mode == "physical_zero":
                backups = apply_physical_prune_inplace(bundle, attn_by_layer, ffn_by_layer, selected_units)
            else:  # physical_slice: true structural removal (irreversible)
                slice_info = apply_structural_prune_inplace(bundle, attn_by_layer, ffn_by_layer, selected_units)

        prompts = load_prompts(cfg["eval"]["eval"]["standard_benchmarks"]["cache_dir"], args.num_prompts)
        report = {
            "mode": args.mode,
            "model_key": cfg["run"]["model_key"],
            "device": str(bundle_device(bundle)),
            "mask_run_dir": args.mask_run_dir,
            "cost": mask_cost_report(bundle, cfg, selected_units),
            "params": param_report(bundle),
            "forward": measure_forward(bundle, prompts, args.max_seq_len, args.warmup, args.repeat),
            "generate": measure_generate(
                bundle,
                prompts[: args.generate_prompts],
                args.max_seq_len,
                args.max_new_tokens,
                args.warmup,
                args.repeat,
            ),
        }
        if slice_info is not None:
            report["structural_slice"] = slice_info
        save_json(out_dir / f"efficiency_{args.mode}.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if mask_state is not None:
            clear_runtime_masks(mask_state)
        if backups is not None:
            from cocurve.prune import restore_physical_prune

            restore_physical_prune(backups)


if __name__ == "__main__":
    main()
