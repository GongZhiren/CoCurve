#!/usr/bin/env python3
"""Reconstruct a fixed CoCurve mask, optionally refine FFNs, and save a checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from cocurve.checkpoints import save_checkpoint
from cocurve.config import load_config, model_config
from cocurve.model import load_model_bundle
from cocurve.reconstruction import reconstruct


def packed_inputs(tokenizer, path, count, length):
    windows, buffer = [], []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            buffer.extend(tokenizer(json.loads(line)["text"], add_special_tokens=False)["input_ids"])
            while len(buffer) >= length:
                windows.append(torch.tensor([buffer[:length]], dtype=torch.long))
                del buffer[:length]
                if len(windows) == count:
                    return windows
    raise ValueError(f"Calibration contains only {len(windows)} complete windows; need {count}")


def main():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--reconstruction-config", default="configs/reconstruction.yaml")
    known, _ = preliminary.parse_known_args()
    protocol = yaml.safe_load(Path(known.reconstruction_config).read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-config", default=known.reconstruction_config)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--mask", required=True, help="CoCurve prune_solution.json")
    parser.add_argument("--method", choices=("obs", "ffn_refinement"),
                        default=protocol["reconstruction"]["method"])
    parser.add_argument("--calibration", required=True, help="Independent calibration JSONL (text field)")
    parser.add_argument("--samples", type=int, default=protocol["calibration"]["sequences"])
    parser.add_argument("--seq-len", type=int, default=protocol["calibration"]["sequence_length"])
    parser.add_argument("--damping", type=float, default=protocol["reconstruction"]["damping"])
    parser.add_argument("--ffn-groups", type=int, default=None,
                        help="infer from mask metadata, otherwise use 16")
    parser.add_argument("--device", default=protocol["reconstruction"]["device"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.seq_len < 1:
        parser.error("samples and seq-len must be positive")
    source = json.loads(Path(args.mask).read_text())
    if source.get("model_key", args.model_key) != args.model_key:
        parser.error("the mask belongs to a different model")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error("output must be empty")
    cfg = load_config(args.config)
    cfg["run"]["model_key"] = args.model_key
    mc = dict(model_config(cfg))
    mc.update(device_map={"": "cpu"}, load_in_4bit=False, load_in_8bit=False,
              attn_implementation="eager")
    bundle = load_model_bundle(mc, cfg["model"].get("tokenizer", {}))
    inputs = packed_inputs(bundle.tokenizer, args.calibration, args.samples, args.seq_len)
    report = reconstruct(bundle, args.mask, inputs, method=args.method, damping=args.damping,
                         device=args.device,
                         ffn_groups=args.ffn_groups or int(source.get("ffn_groups_per_layer", 16)))
    report.update(model_key=args.model_key, calibration_sequences=args.samples,
                  calibration_length=args.seq_len, damping=args.damping)
    save_checkpoint(bundle.model, out, tokenizer=bundle.tokenizer)
    (out / "reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "prune_solution.json").write_text(json.dumps(
        {k: report[k] for k in ("selected_units", "pruned_units", "ffn_groups_per_layer",
                               "actual_prune_ratio", "model_key")}, indent=2) + "\n")
    print(json.dumps({"output": str(out), "method": args.method,
                      "parameters": report["parameters_after"]}))


if __name__ == "__main__":
    main()
