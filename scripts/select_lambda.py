#!/usr/bin/env python3
"""Select CoCurve's interaction strength on held-out calibration text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cocurve.config import load_config, model_config
from cocurve.model import bundle_device, load_model_bundle
from cocurve.path import enumerate_path
from cocurve.prune import clear_runtime_masks, register_runtime_masks
from cocurve.units import build_unit_registry, unit_cost_vector


def pack_holdout(tokenizer, path: Path, count: int, length: int) -> list[torch.Tensor]:
    sequences, buffer = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            ids = tokenizer(json.loads(line)["text"], return_tensors="pt")["input_ids"][0]
            buffer.append(ids)
            if sum(int(part.numel()) for part in buffer) >= length:
                sequences.append(torch.cat(buffer)[:length].unsqueeze(0))
                buffer = []
                if len(sequences) == count:
                    break
    if len(sequences) != count:
        raise ValueError(f"packed {len(sequences)} holdout sequences, expected {count}")
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--h", required=True, help="CoCurve H.npy")
    parser.add_argument("--holdout", required=True, help="held-out calibration JSONL")
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["run"] = dict(cfg["run"])
    cfg["run"]["model_key"] = args.model_key
    bundle = load_model_bundle(model_config(cfg), cfg["model"].get("tokenizer", {}))
    device = bundle_device(bundle)
    registry = build_unit_registry(
        num_layers=bundle.num_layers,
        num_heads=bundle.num_heads,
        hidden_size=bundle.hidden_size,
        intermediate_size=bundle.intermediate_size,
        ffn_groups_per_layer=int(cfg["pruning"]["units"]["ffn_groups_per_layer"]),
        cost_type=str(cfg["pruning"]["units"]["cost_type"]),
        kv_heads=bundle.kv_heads,
        head_dim=bundle.head_dim,
    )
    costs = unit_cost_vector(registry).numpy().astype(np.float64)
    layers = [int(unit.layer_idx) for unit in registry.units]
    protect_first = int(cfg["pruning"]["solver"].get("protect_first_layers", 2))
    protect_last = int(cfg["pruning"]["solver"].get("protect_last_layers", 2))
    protected = set(range(protect_first)) | set(
        range(bundle.num_layers - protect_last, bundle.num_layers)
    )
    layer_cap = 1.5 * float(args.ratio)
    family = enumerate_path(
        np.load(args.h), costs, args.ratio, layers, layer_cap, protected, workers=args.workers
    )
    print(f"enumerated {len(family)} distinct masks", flush=True)

    sequences = pack_holdout(
        bundle.tokenizer, Path(args.holdout), args.sequences, args.sequence_length
    )
    teacher = []
    with torch.no_grad():
        for sequence in sequences:
            logits = bundle.model(input_ids=sequence.to(device), use_cache=False).logits[0]
            log_prob = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            probability = log_prob.exp()
            teacher.append((probability.to(torch.float16).cpu(),
                            (probability * log_prob).sum(-1).cpu()))
    torch.cuda.empty_cache()

    attention, ffn = {}, {}
    for unit in registry.units:
        target = attention if unit.unit_type == "attn_head" else ffn
        target.setdefault(unit.layer_idx, []).append(unit)

    for index, member in enumerate(family, start=1):
        kept = set(range(registry.num_units)) - set(member["pruned_units"])
        state = register_runtime_masks(bundle, attention, ffn, kept)
        risks = []
        try:
            with torch.no_grad():
                for sequence, (probability, entropy) in zip(sequences, teacher):
                    logits = bundle.model(
                        input_ids=sequence.to(device), use_cache=False
                    ).logits[0]
                    log_prob = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                    q = probability.to(device, dtype=torch.float32)
                    risks.append(float((entropy.to(device) - (q * log_prob).sum(-1)).mean()))
        finally:
            clear_runtime_masks(state)
        member["risk_per_sequence"] = risks
        member["heldout_kl"] = float(np.mean(risks))
        print(f"[{index}/{len(family)}] lambda={member['lambda_lo']:.6g} "
              f"KL={member['heldout_kl']:.6f}", flush=True)

    best_index = int(np.argmin([member["heldout_kl"] for member in family]))
    best = family[best_index]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path_payload = {
        "schema_version": 1,
        "model_key": args.model_key,
        "target_prune_ratio": args.ratio,
        "best_index": best_index,
        "interaction_strength": float(best["lambda_lo"]),
        "heldout_sequences": args.sequences,
        "sequence_length": args.sequence_length,
        "members": [{k: v for k, v in member.items() if k != "pruned_units"}
                    for member in family],
    }
    mask = {
        "schema_version": 1,
        "method": "cocurve",
        "model_key": args.model_key,
        "target_prune_ratio": args.ratio,
        "actual_prune_ratio": float(best["actual_prune_ratio"]),
        "interaction_strength": float(best["lambda_lo"]),
        "selected_units": sorted(set(range(registry.num_units)) - set(best["pruned_units"])),
        "pruned_units": sorted(map(int, best["pruned_units"])),
    }
    (output / "lambda_path.json").write_text(
        json.dumps(path_payload, indent=2) + "\n", encoding="utf-8"
    )
    (output / "prune_solution.json").write_text(
        json.dumps(mask, indent=2) + "\n", encoding="utf-8"
    )
    print(f"selected lambda={best['lambda_lo']:.8g}; wrote {output}")


if __name__ == "__main__":
    main()
