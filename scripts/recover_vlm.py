#!/usr/bin/env python3
"""Run the paper's common VLM recovery recipe on a released CoCurve mask."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cocurve.vlm.bundle import load_vlm
from cocurve.vlm.calib import build_batch, load_pairs_split
from cocurve.vlm import evalvlm as evaluation
from cocurve.vlm.masks import masked
from cocurve.vlm import recovery
from cocurve.vlm.units import build_registry


PAPER_TASKS = "mmbench,seedbench,scienceqa,ai2d,mmstar,pope,mme"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("mask", help="released prune_solution.json")
    parser.add_argument("output")
    parser.add_argument("--tasks", default=PAPER_TASKS)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--projector-learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    torch.manual_seed(args.seed)
    mask_payload = json.loads(Path(args.mask).read_text(encoding="utf-8"))
    if mask_payload.get("model_id") not in (None, args.model_id):
        raise ValueError(f"mask model {mask_payload.get('model_id')} != {args.model_id}")
    removed = set(map(int, mask_payload["pruned_units"]))

    bundle = load_vlm(args.model_id)
    bundle.processor.tokenizer.padding_side = "left"
    evaluation.cap_pixels(bundle, evaluation.MAX_PIXELS)
    registry = build_registry(bundle, ffn_groups_per_layer=16)

    from peft import LoraConfig, get_peft_model
    targets = recovery.lora_targets(bundle)
    model = get_peft_model(bundle.model, LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
    ))
    bundle.model = model
    recovery.freeze_tower_lora(model, bundle, "vis")
    projector_parameters, projector_names = recovery.projector_parameters(bundle)
    for parameter in projector_parameters:
        parameter.requires_grad = True

    projector_ids = {id(parameter) for parameter in projector_parameters}
    adapter_parameters = [parameter for parameter in model.parameters()
                          if parameter.requires_grad and id(parameter) not in projector_ids]
    groups = [{"params": adapter_parameters, "lr": args.learning_rate}]
    if projector_parameters:
        groups.append({"params": projector_parameters,
                       "lr": args.projector_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)

    examples = recovery.load_mixed(args.samples, seed=args.seed, short_frac=0.5)
    steps = math.ceil(len(examples) / (args.batch_size * args.gradient_accumulation))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in groups],
        total_steps=steps,
        pct_start=0.03,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    completed, micro_step = 0, 0
    optimizer.zero_grad(set_to_none=True)
    with masked(bundle, registry, removed):
        while completed < len(examples):
            chunk = examples[completed:completed + args.batch_size]
            completed += len(chunk)
            encoded = recovery.build_train_batch(bundle, chunk)
            loss_mask = encoded.pop("_loss_mask")
            forward = {key: value for key, value in encoded.items() if not key.startswith("_")}
            target = encoded["input_ids"][:, 1:]
            positions = loss_mask[:, :-1]
            output = model(**forward, use_cache=False)
            logits = output.logits[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                logits[positions].float(), target[positions]
            ) / args.gradient_accumulation
            loss.backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation == 0 or completed == len(examples):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        items = {task: evaluation.load_task(task, args.limit, seed=0) for task in tasks}
        accuracy = {task: evaluation.run_task(bundle, items[task], batch_size=4)
                    for task in tasks}
        _, holdout_pairs = load_pairs_split(128, 48, seed=0)
        holdout = [build_batch(bundle, holdout_pairs[i:i + 4])
                   for i in range(0, len(holdout_pairs), 4)]
        ppl = evaluation.caption_ppl(bundle, holdout)

    destination = Path(args.output)
    adapter_dir = destination / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    bundle.processor.save_pretrained(destination / "processor")
    result = {
        "schema_version": 1,
        "model_id": args.model_id,
        "mask": str(Path(args.mask)),
        "recipe": {
            "rank": args.rank,
            "alpha": args.alpha,
            "learning_rate": args.learning_rate,
            "projector_learning_rate": args.projector_learning_rate,
            "projector_modules": projector_names,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "samples": len(examples),
            "steps": steps,
            "vision_tower_frozen": True,
        },
        "caption_ppl": float(ppl),
        "accuracy": {key: 100 * float(value) for key, value in accuracy.items()},
        "mean_accuracy": 100 * float(np.mean(list(accuracy.values()))),
        "minutes": (time.time() - started) / 60,
    }
    (destination / "recovery_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote adapter and report to {destination}")


if __name__ == "__main__":
    main()
