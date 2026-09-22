#!/usr/bin/env python3
"""Apply the paper's lightweight LoRA recipe to a fixed CoCurve mask.

Defaults reproduce the paper: cleaned Alpaca, one epoch, 1024-token sequences,
rank 16, alpha 32, batch 2 with 8-way accumulation, and peak learning rate 1e-4.
The compact adapter is saved by default; use ``--save-merged`` only when needed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cocurve.config import load_config, model_config  # noqa: E402
from cocurve.model import bundle_device, load_model_bundle  # noqa: E402
from cocurve.prune import (apply_structural_prune_inplace,
                               register_runtime_masks)  # noqa: E402
from cocurve.units import build_unit_registry  # noqa: E402


def alpaca_tokens(tok, seq_len: int):
    """Token-pack the cleaned Alpaca recovery corpus with its standard prompt format."""
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train", cache_dir="data/eval/hf_cache")
    buf, seqs = [], []
    for rec in ds:
        inp = str(rec.get("input") or "").strip()
        head = ("Below is an instruction that describes a task, paired with an input that "
                "provides further context. Write a response that appropriately completes "
                "the request.") if inp else ("Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.")
        body = f"{head}\n\n### Instruction:\n{rec['instruction']}\n"
        if inp:
            body += f"\n### Input:\n{inp}\n"
        body += f"\n### Response:\n{rec['output']}"
        buf.extend(tok(body)["input_ids"])
        while len(buf) >= seq_len:
            seqs.append(torch.tensor(buf[:seq_len]))
            buf = buf[seq_len:]
    return torch.stack(seqs)


def stream_tokens(tok, n_tokens: int, seq_len: int, skip_docs: int):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    buf, seqs, seen = [], [], 0
    for rec in ds:
        seen += 1
        if seen <= skip_docs:
            continue
        buf.extend(tok(rec["text"])["input_ids"])
        while len(buf) >= seq_len:
            seqs.append(torch.tensor(buf[:seq_len]))
            buf = buf[seq_len:]
            if len(seqs) * seq_len >= n_tokens:
                return torch.stack(seqs)
    return torch.stack(seqs)


FULL_EVAL_TASKS = [
    "wikitext", "ptb", "c4", "arc_challenge", "arc_easy", "hellaswag",
    "winogrande", "piqa", "openbookqa", "boolq", "mmlu",
    "commonsense_qa", "race", "race_high", "quail",
]


def _evaluate_in_process(merged, tok, cfg, out: Path, tasks: list[str]) -> None:
    """Run the standard suite on the merged model without a save/load round trip.

    Imports the evaluators from run_standard_eval rather than reimplementing them, so a
    recovered model is scored by exactly the code that scored every other row in the
    paper -- the same accuracy convention, the same perplexity stride, the same
    full-test-set policy.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_rse", str(Path(__file__).with_name("run_standard_eval.py")))
    rse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rse)

    eval_dir = out / "standard_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    # A ModelBundle is the model plus shape metadata; the evaluators use only the model
    # and the tokenizer, so the metadata is left at zero rather than recomputed.
    from cocurve.model import ModelBundle
    bundle = ModelBundle(model=merged, tokenizer=tok,
                         device="cuda" if torch.cuda.is_available() else "cpu",
                         num_layers=0, num_heads=0, hidden_size=0, head_dim=0,
                         intermediate_size=0, kv_heads=0, attn_pattern="gqa",
                         family="recovered", model_path=str(out))
    cache_dir = cfg["eval"]["eval"]["standard_benchmarks"]["cache_dir"]
    report_path = eval_dir / "standard_eval_report.json"
    report = {}
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                report = loaded
        except (json.JSONDecodeError, OSError):
            report = {}
    for task in tasks:
        ds = rse.dataset_for_task(task, cache_dir)
        previous = report.get(task)
        if task in rse.LM_TASKS:
            complete = (
                isinstance(previous, dict)
                and previous.get("metric") == "perplexity"
                and int(previous.get("num_tokens", 0)) > 0
                and (eval_dir / f"{task}_windows.json").is_file()
            )
        else:
            complete = (
                isinstance(previous, dict)
                and previous.get("metric") == "multiple_choice_loglik"
                and int(previous.get("num_samples", -1)) == len(ds)
                and "accuracy_norm" in previous
                and (eval_dir / f"{task}_predictions.json").is_file()
            )
        if complete:
            print(f"  [eval] {task} already complete; skip", flush=True)
            continue
        if task in rse.LM_TASKS:
            report[task] = rse.evaluate_wikitext_ppl(bundle, ds, eval_dir, 2048, 512, -1, task=task)
        else:
            report[task] = rse.evaluate_mc(bundle, task, ds, eval_dir, 2048, 100)
        report_path.write_text(json.dumps(report, indent=1))
        print(f"  [eval] {task} done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--mask-run-dir", required=True,
                    help="run dir holding masks/prune_solution.json to realise before training")
    ap.add_argument("--out", required=True, help="destination for the adapter, metadata, and evaluation")
    ap.add_argument("--tokens", type=int, default=8_000_000)
    ap.add_argument("--epochs", type=int, default=1,
                    help="passes over the fixed recovery corpus; default 1 preserves the common recipe")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--skip-docs", type=int, default=200_000,
                    help="skip well past the documents the calibration set consumed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", choices=["c4", "alpaca"], default="alpaca",
                    help="recovery corpus; the paper uses cleaned Alpaca")
    ap.add_argument("--eval-only", action="store_true",
                    help="score an already-trained checkpoint in --out; skip training")
    ap.add_argument("--save-merged", action="store_true",
                    help="also store the full merged checkpoint (adapter-only is the default)")
    ap.add_argument("--runtime-mask", action="store_true",
                    help=("keep the fixed deletion mask as forward hooks instead of slicing "
                          "packed low-bit matrices; required for the 70B NF4 QLoRA route"))
    ap.add_argument("--max-train-sequences", type=int, default=-1,
                    help="pilot-only cap; -1 retains the common full recovery corpus")
    ap.add_argument("--skip-eval", action="store_true",
                    help="pilot-only: stop after training and adapter serialization")
    ap.add_argument(
        "--eval-tasks",
        default=",".join(FULL_EVAL_TASKS),
        help=("comma-separated in-process evaluation tasks; use wikitext,ptb,c4 "
              "for the low-cost cross-model recovery sweep"),
    )
    args = ap.parse_args()
    if args.epochs < 1:
        raise SystemExit("ABORT: --epochs must be positive")
    eval_tasks = [task.strip() for task in args.eval_tasks.split(",") if task.strip()]
    unknown = sorted(set(eval_tasks) - set(FULL_EVAL_TASKS))
    if unknown:
        raise SystemExit(f"ABORT: unsupported recovery evaluation tasks: {unknown}")

    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    cfg["run"] = dict(cfg["run"]); cfg["run"]["model_key"] = args.model_key
    bundle = load_model_bundle(model_config(cfg), cfg["model"].get("tokenizer", {}))
    device = bundle_device(bundle)
    tok = bundle.tokenizer

    registry = build_unit_registry(
        num_layers=bundle.num_layers, num_heads=bundle.num_heads,
        hidden_size=bundle.hidden_size, intermediate_size=bundle.intermediate_size,
        ffn_groups_per_layer=int(cfg["pruning"]["units"]["ffn_groups_per_layer"]),
        cost_type=str(cfg["pruning"]["units"]["cost_type"]), kv_heads=bundle.kv_heads, head_dim=bundle.head_dim)
    attn_by_layer, ffn_by_layer = {}, {}
    for s in registry.units:
        (attn_by_layer if s.unit_type == "attn_head" else ffn_by_layer) \
            .setdefault(s.layer_idx, []).append(s)
    solution = json.loads((Path(args.mask_run_dir) / "masks/prune_solution.json").read_text())
    pruned = set(solution["pruned_units"])
    kept = set(range(len(registry.units))) - pruned
    n0 = sum(p.numel() for p in bundle.model.parameters())
    if args.runtime_mask:
        # bitsandbytes stores NF4 matrices in a packed representation that must
        # not be sliced as if it were an ordinary dense tensor.  Runtime hooks
        # implement the identical fixed unit mask while preserving the packed
        # base model; LoRA is still the only trainable component.
        register_runtime_masks(bundle, attn_by_layer, ffn_by_layer, kept)
        removed_fraction = float(solution.get("actual_prune_ratio", 0.0))
        n1 = round(n0 * (1.0 - removed_fraction))
        print(f"installed fixed runtime mask for {100 * removed_fraction:.2f}% logical "
              f"parameter pruning", flush=True)
    else:
        apply_structural_prune_inplace(bundle, attn_by_layer, ffn_by_layer, kept)
        n1 = sum(p.numel() for p in bundle.model.parameters())
        removed_fraction = 1 - n1 / n0
        print(f"physically removed {100 * removed_fraction:.2f}% of parameters "
              f"({n0/1e9:.2f}B -> {n1/1e9:.2f}B)", flush=True)

    # --eval-only recovers an already-trained checkpoint without retraining it.  The
    # saved weights are correct; only the config beside them is not, and the pruned
    # module shapes have just been rebuilt above by exactly the code that produced them,
    # so the state dict loads into this model even though from_pretrained cannot.
    if args.eval_only:
        out = Path(args.out)
        adapter_dir = out / "adapter"
        if (adapter_dir / "adapter_model.safetensors").is_file():
            from peft import PeftModel
            recovered = PeftModel.from_pretrained(bundle.model, adapter_dir)
            # Merging into packed 4-bit weights changes the numerical contract;
            # evaluate the adapter composition directly for QLoRA.
            merged = recovered if args.runtime_mask else recovered.merge_and_unload()
            print(f"loaded compact adapter from {adapter_dir}", flush=True)
        else:
            from safetensors.torch import load_file
            sd = {}
            for shard in sorted(out.glob("model*.safetensors")):
                sd.update(load_file(str(shard)))
            if not sd:
                raise SystemExit(f"ABORT: no merged weights or compact adapter found in {out}")
            missing, unexpected = bundle.model.load_state_dict(sd, strict=False)
            tied = [k for k in missing if "lm_head" in k]  # tied embeddings are not stored
            if [k for k in missing if k not in tied] or unexpected:
                raise SystemExit(f"ABORT: recovered checkpoint does not match the pruned model "
                                 f"({len(missing)} missing, {len(unexpected)} unexpected)")
            merged = bundle.model
            print(f"loaded recovered weights from {out} ({len(sd)} tensors)", flush=True)
        meta_path = out / "recovery_meta.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps(dict(
                model_key=args.model_key, mask_run_dir=args.mask_run_dir,
                tokens=args.tokens, seq_len=args.seq_len, rank=args.rank,
                alpha=args.alpha, lr=args.lr, corpus=args.corpus,
                params_before=n0, params_after=n1,
                removed_fraction=removed_fraction,
                mask_realization="runtime" if args.runtime_mask else "physical",
                eval_tasks=eval_tasks, resumed_evaluation=True,
                status="evaluation_pending"), indent=1))
        _evaluate_in_process(merged, tok, cfg, out, eval_tasks)
        meta = json.loads(meta_path.read_text())
        # Keep provenance synchronized when an existing PPL-only checkpoint is
        # later extended to the full task suite.
        completed = set(meta.get("eval_tasks", [])) | set(eval_tasks)
        meta["eval_tasks"] = [task for task in FULL_EVAL_TASKS if task in completed]
        meta["status"] = "complete"
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"evaluated {out}")
        return

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    is_kbit = bool(getattr(bundle.model, "is_loaded_in_4bit", False)
                   or getattr(bundle.model, "is_loaded_in_8bit", False))
    if is_kbit:
        if not args.runtime_mask:
            raise SystemExit("ABORT: packed low-bit LoRA requires --runtime-mask")
        bundle.model = prepare_model_for_kbit_training(
            bundle.model, use_gradient_checkpointing=True)
    targets = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")
               if any(n in mn for mn, _ in bundle.model.named_modules())]
    model = get_peft_model(bundle.model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    print("streaming recovery corpus ...", flush=True)
    if args.corpus == "alpaca":
        data = alpaca_tokens(tok, args.seq_len)
    else:
        data = stream_tokens(tok, args.tokens, args.seq_len, args.skip_docs)
    if args.max_train_sequences > 0:
        data = data[:args.max_train_sequences]
    print(f"recovery corpus: {args.corpus}, {data.shape[0]} sequences of {args.seq_len}", flush=True)
    print(f"  {data.shape[0]} sequences x {args.seq_len} tokens", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_kbit:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW32bit(trainable, lr=args.lr,
                                        weight_decay=0.0, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr,
                                weight_decay=0.0, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(data.shape[0] / (args.batch * args.accum))
    steps = steps_per_epoch * args.epochs
    if steps < 3:
        # A one-step memory pilot has no meaningful LR trajectory, and some
        # PyTorch releases make OneCycleLR's warm-up interval degenerate here.
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=steps,
            pct_start=0.03, anneal_strategy="cos")
    t0, step, running = time.time(), 0, 0.0
    for epoch in range(args.epochs):
        i = 0
        print(f"epoch {epoch + 1}/{args.epochs}", flush=True)
        while i < data.shape[0]:
            opt.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                if i >= data.shape[0]:
                    break
                b = data[i:i + args.batch].to(device); i += b.shape[0]
                out = model(input_ids=b, labels=b)
                (out.loss / args.accum).backward()
                running += float(out.loss.detach())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); step += 1
            if step % 20 == 0:
                print(f"  step {step}/{steps}  loss {running / max(1, step * args.accum):.4f}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if not args.save_merged:
        adapter_dir = out / "adapter"
        model.save_pretrained(adapter_dir)
        tok.save_pretrained(out)
        print(f"saved compact adapter to {adapter_dir}", flush=True)
    print("merging adapters ...", flush=True)
    merged = model if is_kbit else model.merge_and_unload()
    if args.save_merged:
        merged.save_pretrained(out); tok.save_pretrained(out)
    meta_path = out / "recovery_meta.json"
    meta = dict(
        model_key=args.model_key, mask_run_dir=args.mask_run_dir, tokens=args.tokens,
        epochs=args.epochs, effective_training_tokens=int(data.numel()) * args.epochs,
        seq_len=args.seq_len, rank=args.rank, alpha=args.alpha, lr=args.lr,
        steps=step, corpus=args.corpus, params_before=n0, params_after=n1,
        removed_fraction=removed_fraction,
        mask_realization="runtime" if args.runtime_mask else "physical",
        base_precision="NF4" if is_kbit else "bf16",
        checkpoint_format="merged" if args.save_merged else "adapter_only",
        eval_tasks=eval_tasks,
        training_minutes=(time.time() - t0) / 60, status="evaluation_pending")
    meta_path.write_text(json.dumps(meta, indent=1))
    if args.skip_eval:
        meta["minutes"] = (time.time() - t0) / 60
        meta["peak_cuda_gib"] = (torch.cuda.max_memory_allocated() / 2**30
                                 if torch.cuda.is_available() else None)
        meta["status"] = "pilot_complete"
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"pilot complete: peak CUDA allocation {meta['peak_cuda_gib']}", flush=True)
        return
    # Evaluate in-process because heterogeneous per-layer sliced shapes cannot be
    # reconstructed from a stock Hugging Face config alone. Compact adapters remain
    # reloadable after rebuilding the same released structural mask.
    _evaluate_in_process(merged, tok, cfg, out, eval_tasks)
    meta["minutes"] = (time.time() - t0) / 60
    meta["status"] = "complete"
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
