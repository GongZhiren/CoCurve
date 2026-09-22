#!/usr/bin/env python3
"""Select and evaluate CoCurve jointly over a VLM's language and vision towers."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cocurve.vlm.bundle import load_vlm
from cocurve.vlm.calib import build_batch, load_pairs_split, teacher_cache
from cocurve.vlm.evalvlm import MAX_PIXELS, cap_pixels, caption_ppl, load_task, run_task
from cocurve.vlm.masks import masked
from cocurve.vlm.solve import enumerate_family, heldout_risk, select_lambda, split_by
from cocurve.vlm.units import build_registry, summarise


PAPER_TASKS = "mmbench,seedbench,scienceqa,ai2d,mmstar,pope,mme"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("h_dir")
    parser.add_argument("output")
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--mask", default=None,
                        help="released prune_solution.json; skips path selection")
    parser.add_argument("--tasks", default=PAPER_TASKS)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--calibration-pairs", type=int, default=128)
    parser.add_argument("--holdout-pairs", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--ffn-groups", type=int, default=16)
    parser.add_argument("--cap-multiplier", type=float, default=1.5)
    parser.add_argument("--protect-layers", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--include-dense", action="store_true")
    args = parser.parse_args()

    started = time.time()
    bundle = load_vlm(args.model_id)
    bundle.processor.tokenizer.padding_side = "left"
    cap_pixels(bundle, args.max_pixels)
    registry = build_registry(bundle, ffn_groups_per_layer=args.ffn_groups)
    h = np.load(Path(args.h_dir) / "H.npy")
    if h.shape != (registry.n, registry.n):
        raise ValueError(f"H shape {h.shape} does not match {registry.n} registered units")
    print(f"{args.model_id}\n{summarise(registry)}", flush=True)

    _, holdout_pairs = load_pairs_split(
        args.calibration_pairs, args.holdout_pairs, seed=0
    )
    holdout = [build_batch(bundle, holdout_pairs[i:i + 4])
               for i in range(0, len(holdout_pairs), 4)]
    teacher_index, teacher_probability, _ = teacher_cache(bundle, holdout, top_r=64)

    lambda_summary = None
    if args.mask:
        mask = json.loads(Path(args.mask).read_text(encoding="utf-8"))
        if mask.get("model_id") not in (None, args.model_id):
            raise ValueError(f"mask model {mask.get('model_id')} != {args.model_id}")
        removed = sorted(map(int, mask["pruned_units"]))
        strength = float(mask["interaction_strength"])
    else:
        family = enumerate_family(
            h, registry, bundle, args.ratio, args.cap_multiplier, args.protect_layers
        )
        best, scored = select_lambda(
            bundle, registry, family, holdout, teacher_index, teacher_probability
        )
        removed = sorted(map(int, best["pruned"]))
        strength = float(best["lam_lo"])
        lambda_summary = {"members": len(family), "selected": strength, "scores": scored}

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    items = {task: load_task(task, args.limit, seed=0) for task in tasks}

    def evaluate(pruned: list[int]) -> dict:
        with masked(bundle, registry, set(pruned)):
            accuracy = {task: run_task(bundle, items[task], batch_size=args.eval_batch_size)
                        for task in tasks}
            ppl = caption_ppl(bundle, holdout)
            kl = heldout_risk(
                bundle, registry, pruned, holdout, teacher_index, teacher_probability
            ) if pruned else 0.0
        costs = registry.cost_vector().numpy()
        actual = float(costs[pruned].sum() / costs.sum()) if pruned else 0.0
        return {
            "actual_prune_ratio": actual,
            "pruned_units": pruned,
            "caption_ppl": float(ppl),
            "heldout_kl": float(kl),
            "accuracy": {key: 100 * float(value) for key, value in accuracy.items()},
            "mean_accuracy": 100 * float(np.mean(list(accuracy.values()))),
            "split": split_by(registry, pruned),
        }

    payload = {
        "schema_version": 1,
        "model_id": args.model_id,
        "target_prune_ratio": args.ratio,
        "interaction_strength": strength,
        "tasks": tasks,
        "evaluation_items_per_task": args.limit,
        "cocurve": evaluate(removed),
        "minutes": (time.time() - started) / 60,
    }
    if args.include_dense:
        payload["dense"] = evaluate([])
    if lambda_summary:
        payload["lambda_path"] = lambda_summary
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
