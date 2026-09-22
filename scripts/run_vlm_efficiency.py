#!/usr/bin/env python3
"""Measure a released VLM mask after true two-tower structural slicing."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from cocurve.vlm.bundle import load_vlm
from cocurve.vlm.calib import build_batch, load_pairs
from cocurve.vlm.evalvlm import MAX_PIXELS, cap_pixels
from cocurve.vlm.masks import masked
from cocurve.vlm.prune import apply_structural_prune_inplace
from cocurve.vlm.units import build_registry


def _forward(bundle, batch):
    inputs = {key: value for key, value in batch.items() if not key.startswith("_")}
    return bundle.model(**inputs, use_cache=False).logits


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mode", choices=("dense", "physical_slice"), required=True)
    parser.add_argument("--mask", help="released prune_solution.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffn-groups", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    args = parser.parse_args()
    if args.mode == "physical_slice" and not args.mask:
        parser.error("physical_slice requires --mask")

    torch.manual_seed(0)
    bundle = load_vlm(args.model_id, dtype=torch.bfloat16)
    bundle.processor.tokenizer.padding_side = "left"
    cap_pixels(bundle, args.max_pixels)
    registry = build_registry(bundle, ffn_groups_per_layer=args.ffn_groups)
    batch = build_batch(bundle, load_pairs(args.batch_size, seed=17))

    dense_parameters = sum(parameter.numel() for parameter in bundle.model.parameters())
    equivalence = None
    realized_ratio = 0.0
    if args.mode == "physical_slice":
        payload = json.loads(Path(args.mask).read_text(encoding="utf-8"))
        if payload.get("model_id") not in (None, args.model_id):
            raise ValueError(f"mask model {payload.get('model_id')} != {args.model_id}")
        removed = set(map(int, payload["pruned_units"]))
        realized_ratio = float(payload["actual_prune_ratio"])
        with torch.inference_mode(), masked(bundle, registry, removed):
            masked_logits = _forward(bundle, batch).float().cpu()
        apply_structural_prune_inplace(bundle, registry, removed)
        with torch.inference_mode():
            sliced_logits = _forward(bundle, batch).float().cpu()
        difference = sliced_logits - masked_logits
        rms = float(difference.square().mean().sqrt())
        denominator = float(masked_logits.square().mean().sqrt().clamp_min(1e-8))
        equivalence = {
            "max_abs": float(difference.abs().max()),
            "rms": rms,
            "relative_rms": rms / denominator,
        }
        if (not torch.isfinite(difference).all()
                or equivalence["relative_rms"] > 0.10
                or equivalence["max_abs"] > 8.0):
            raise RuntimeError(f"mask/slice validation failed: {equivalence}")

    torch.cuda.empty_cache()
    with torch.inference_mode():
        for _ in range(args.warmup):
            _forward(bundle, batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(args.repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _forward(bundle, batch)
            end.record()
            torch.cuda.synchronize()
            durations.append(float(start.elapsed_time(end)))
        peak = int(torch.cuda.max_memory_allocated())

    tokens = int(batch["attention_mask"].sum())
    median_ms = statistics.median(durations)
    report = {
        "model_id": args.model_id,
        "mode": args.mode,
        "realized_ratio": realized_ratio,
        "hardware": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "input_tokens": tokens,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "params_dense": dense_parameters,
        "params_measured": sum(parameter.numel() for parameter in bundle.model.parameters()),
        "prefill_ms_median": median_ms,
        "prefill_tokens_s": 1000.0 * tokens / median_ms,
        "peak_allocated_gib": peak / (1 << 30),
        "equivalence": equivalence,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
