#!/usr/bin/env python3
"""Measure the paper's sliced bf16/INT8/NF4 deployment configurations.

Irregular per-layer widths cannot be reconstructed by a stock Hugging Face
config.  This entry point therefore rebuilds the released structure first,
optionally merges its LoRA adapter, and only then quantizes the remaining
linear layers.  Dense low-bit references use the normal Transformers loader.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from cocurve.config import load_config, model_config
from cocurve.model import bundle_device, load_model_bundle
from cocurve.prune import apply_structural_prune_inplace
from cocurve.units import build_unit_registry


def _logical_parameter_count(model: torch.nn.Module) -> int:
    """Count logical scalar weights rather than packed low-bit storage."""
    total = 0
    for parameter in model.parameters():
        quant_state = getattr(parameter, "quant_state", None)
        original_shape = getattr(quant_state, "shape", None)
        if original_shape is None:
            total += parameter.numel()
            continue
        count = 1
        for width in original_shape:
            count *= int(width)
        total += count
    return int(total)


def _load_script(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_cocurve_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _slice(bundle, cfg: dict, mask_path: Path) -> tuple[int, int]:
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
    attention, ffn = {}, {}
    for unit in registry.units:
        target = attention if unit.unit_type == "attn_head" else ffn
        target.setdefault(unit.layer_idx, []).append(unit)
    payload = json.loads(mask_path.read_text(encoding="utf-8"))
    kept = set(range(registry.num_units)) - set(map(int, payload["pruned_units"]))
    before = _logical_parameter_count(bundle.model)
    apply_structural_prune_inplace(bundle, attention, ffn, kept)
    after = _logical_parameter_count(bundle.model)
    return before, after


def _quantize_linears(model: torch.nn.Module, precision: str, device: torch.device) -> int:
    import bitsandbytes as bnb

    converted = 0

    def visit(parent: torch.nn.Module, prefix: str = "") -> None:
        nonlocal converted
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear) and qualified != "lm_head":
                weight = child.weight.detach().cpu()
                bias = child.bias.detach().to(device) if child.bias is not None else None
                if precision == "nf4":
                    replacement = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.bfloat16,
                        compress_statistics=True,
                        quant_type="nf4",
                    )
                    replacement.source_cls = type(child)
                    replacement.weight = bnb.nn.Params4bit(
                        weight,
                        requires_grad=False,
                        compress_statistics=True,
                        quant_type="nf4",
                        module=replacement,
                    ).to(device)
                else:
                    replacement = bnb.nn.Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0,
                    )
                    replacement.source_cls = type(child)
                    replacement.weight = bnb.nn.Int8Params(
                        weight, requires_grad=False, has_fp16_weights=False
                    ).to(device)
                if bias is not None:
                    replacement.bias = torch.nn.Parameter(bias, requires_grad=False)
                parent._modules[name] = replacement
                converted += 1
            else:
                visit(child, qualified)

    visit(model)
    if converted == 0:
        raise RuntimeError("no linear modules were quantized")
    model.eval()
    torch.cuda.empty_cache()
    return converted


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--model-key", default="llama-3.1-8b-instruct")
    parser.add_argument("--structure", choices=("dense", "pruned", "recovered"), required=True)
    parser.add_argument("--precision", choices=("bf16", "int8", "nf4"), required=True)
    parser.add_argument("--mask", help="released prune_solution.json")
    parser.add_argument("--adapter", help="LoRA adapter directory for structure=recovered")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["run"] = dict(cfg["run"])
    cfg["run"]["model_key"] = args.model_key
    model_cfg = dict(model_config(cfg))
    if args.structure == "dense" and args.precision == "nf4":
        model_cfg["load_in_4bit"] = True
    if args.structure == "dense" and args.precision == "int8":
        model_cfg["load_in_8bit"] = True
    bundle = load_model_bundle(model_cfg, cfg["model"].get("tokenizer", {}))

    if not torch.cuda.is_available():
        raise RuntimeError("deployment timing requires a CUDA GPU")

    before = after = _logical_parameter_count(bundle.model)
    if args.structure != "dense":
        if not args.mask:
            parser.error("pruned/recovered structures require --mask")
        before, after = _slice(bundle, cfg, Path(args.mask))
        if args.structure == "recovered":
            if not args.adapter:
                parser.error("structure=recovered requires --adapter")
            from peft import PeftModel

            bundle.model = PeftModel.from_pretrained(bundle.model, args.adapter).merge_and_unload()
        converted = 0 if args.precision == "bf16" else _quantize_linears(
            bundle.model, args.precision, bundle_device(bundle)
        )
    else:
        converted = sum(
            module.__class__.__name__ in {"Linear4bit", "Linear8bitLt"}
            for module in bundle.model.modules()
        )
        if args.precision != "bf16" and converted == 0:
            raise RuntimeError(f"{args.precision} was requested but no low-bit modules were loaded")

    timing = _load_script("run_efficiency_eval.py")
    prompts = timing.load_prompts(
        cfg["eval"]["eval"]["standard_benchmarks"]["cache_dir"], 16
    )
    report = {
        "model_key": args.model_key,
        "structure": args.structure,
        "precision": args.precision,
        "hardware": torch.cuda.get_device_name(0),
        "logical_params_before": before,
        "logical_params_after": after,
        "quantized_linear_modules": converted,
        "memory_metric": "maximum allocated CUDA bytes",
        "forward": timing.measure_forward(bundle, prompts, 2048, 2, 5),
        "generate": timing.measure_generate(bundle, prompts[:8], 2048, 128, 2, 5),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
