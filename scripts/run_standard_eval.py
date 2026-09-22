#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from cocurve.benchmark_scoring import normalize_number, score_continuation
from cocurve.config import load_config, model_config
from cocurve.model import ModelBundle, bundle_device, forward_logits, load_model_bundle
from cocurve.prune import clear_runtime_masks, register_runtime_masks
from cocurve.units import build_unit_registry


MC_TASKS = {"arc_challenge", "arc_easy", "hellaswag", "winogrande", "mmlu",
            "piqa", "openbookqa", "boolq", "commonsense_qa", "race",
            "race_high", "quail", "mmlu_5shot"}
GEN_TASKS = {"gsm8k", "humaneval", "mbpp"}
LM_TASKS = {"wikitext", "ptb", "c4"}

# Project-local jsonl materialized from the corresponding HF datasets.
LOCAL_JSONL_TASKS = {
    "piqa": "data/datasets/reasoning/piqa/validation.jsonl",
    "openbookqa": "data/datasets/reasoning/openbookqa/test.jsonl",
    "boolq": "data/datasets/reasoning/boolq/validation.jsonl",
    "ptb": "data/datasets/language_modeling/ptb/test.jsonl",
    "c4": "data/datasets/language_modeling/c4/validation.jsonl",
}


def _load_local_jsonl(rel_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(rel_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense or CoCurve-masked language models.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--load-in-4bit", action="store_true", help="Evaluate the fixed mask with NF4 weights.")
    parser.add_argument("--load-in-8bit", action="store_true", help="Evaluate the fixed mask with LLM.int8 weights.")
    parser.add_argument("--bf16", action="store_true", help="Force bf16 even if the model registry defaults to NF4.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks or all.")
    parser.add_argument("--mask-run-dir", default=None, help="Optional pruning run dir containing masks/prune_solution.json.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit per non-generation task; -1 evaluates all rows.")
    parser.add_argument("--max-generation-samples", type=int, default=-1, help="Limit per generation task; -1 evaluates all rows.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--ppl-stride", type=int, default=512)
    parser.add_argument("--max-ppl-tokens", type=int, default=-1, help="Limit WikiText tokens for quick diagnostics; -1 uses all tokens.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_for_task(task: str, cache_dir: str):
    if task == "wikitext":
        return load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir)
    if task == "arc_challenge":
        return load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir=cache_dir)
    if task == "arc_easy":
        return load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", cache_dir=cache_dir)
    if task == "hellaswag":
        return load_dataset("hellaswag", split="validation", cache_dir=cache_dir)
    if task == "winogrande":
        return load_dataset("winogrande", "winogrande_xl", split="validation", cache_dir=cache_dir)
    if task == "mmlu":
        return load_dataset("cais/mmlu", "all", split="test", cache_dir=cache_dir)
    if task == "mmlu_5shot":
        return load_dataset("cais/mmlu", "all", split="test", cache_dir=cache_dir)
    if task == "commonsense_qa":
        return load_dataset("tau/commonsense_qa", split="validation", cache_dir=cache_dir)
    if task == "race":
        return load_dataset("ehovy/race", "middle", split="test", cache_dir=cache_dir)
    if task == "race_high":
        return load_dataset("ehovy/race", "high", split="test", cache_dir=cache_dir)
    if task == "quail":
        return load_dataset("textmachinelab/quail", split="validation",
                            revision="refs/convert/parquet", cache_dir=cache_dir)
    if task == "gsm8k":
        return load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
    if task == "humaneval":
        return load_dataset("openai_humaneval", split="test", cache_dir=cache_dir)
    if task == "mbpp":
        return load_dataset("mbpp", split="test", cache_dir=cache_dir)
    if task in LOCAL_JSONL_TASKS:
        return _load_local_jsonl(LOCAL_JSONL_TASKS[task])
    raise ValueError(f"Unsupported task: {task}")


def maybe_limit(ds: Sequence[Any], limit: int) -> Sequence[Any]:
    if limit is None or limit < 0:
        return ds
    return ds.select(range(min(limit, len(ds)))) if hasattr(ds, "select") else ds[:limit]


def arc_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    labels = [str(x) for x in row["choices"]["label"]]
    texts = [str(x) for x in row["choices"]["text"]]
    prompt = f"Question: {row['question']}\nAnswer:"
    gold = str(row["answerKey"]).strip().upper()
    return prompt, labels, texts, gold


def mmlu_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    labels = [chr(ord("A") + i) for i in range(len(row["choices"]))]
    prompt = f"Subject: {row.get('subject', 'unknown')}\nQuestion: {row['question']}\nAnswer:"
    gold = labels[int(row["answer"])]
    return prompt, labels, [str(x) for x in row["choices"]], gold


_MMLU_DEV_CACHE: Dict[str, str] = {}


def mmlu_5shot_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    subject = str(row.get("subject", "unknown"))
    if subject not in _MMLU_DEV_CACHE:
        dev = load_dataset("cais/mmlu", "all", split="dev")
        examples = []
        for item in dev:
            if item.get("subject") != subject:
                continue
            labels = [chr(ord("A") + i) for i in range(len(item["choices"]))]
            options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, item["choices"]))
            examples.append(f"Question: {item['question']}\n{options}\nAnswer: {labels[int(item['answer'])]}")
            if len(examples) == 5:
                break
        _MMLU_DEV_CACHE[subject] = "\n\n".join(examples) + "\n\n"
    labels = [chr(ord("A") + i) for i in range(len(row["choices"]))]
    prompt = (f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"
              f"{_MMLU_DEV_CACHE[subject]}Question: {row['question']}\nAnswer:")
    return prompt, labels, [str(x) for x in row["choices"]], labels[int(row["answer"])]


def hellaswag_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    labels = [str(i) for i in range(len(row["endings"]))]
    prompt = str(row["ctx"])
    gold = str(row["label"])
    return prompt, labels, [str(x) for x in row["endings"]], gold


def winogrande_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    sentence = str(row["sentence"])
    prefix, suffix = sentence.split("_", 1)
    labels = ["1", "2"]
    continuations = [str(row["option1"]) + suffix, str(row["option2"]) + suffix]
    return prefix, labels, continuations, str(row["answer"])


def piqa_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    prompt = f"Question: {row['goal']}\nAnswer:"
    labels = ["0", "1"]
    continuations = [str(row["sol1"]), str(row["sol2"])]
    return prompt, labels, continuations, str(int(row["label"]))


def openbookqa_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    choices = row["choices"]
    labels = [str(x) for x in choices["label"]]
    texts = [str(x) for x in choices["text"]]
    prompt = f"Question: {row['question_stem']}\nAnswer:"
    return prompt, labels, texts, str(row["answerKey"]).strip().upper()


def boolq_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    prompt = f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:"
    labels = ["no", "yes"]
    continuations = ["no", "yes"]
    gold = "yes" if bool(row["answer"]) else "no"
    return prompt, labels, continuations, gold


def commonsenseqa_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    choices = row["choices"]
    return (f"Question: {row['question']}\nAnswer:",
            [str(x) for x in choices["label"]],
            [str(x) for x in choices["text"]],
            str(row["answerKey"]).strip().upper())


def race_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    return (f"Article: {row['article']}\nQuestion: {row['question']}\nAnswer:",
            ["A", "B", "C", "D"], [str(x) for x in row["options"]],
            str(row["answer"]).strip().upper())


def quail_choices(row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    answers = [str(x) for x in row["answers"]]
    labels = ["A", "B", "C", "D"][:len(answers)]
    return (f"{row['context']}\nQuestion: {row['question']}\nAnswer:", labels,
            answers, labels[int(row["correct_answer_id"])])


def row_to_mc(task: str, row: Dict[str, Any]) -> Tuple[str, List[str], List[str], str]:
    if task.startswith("arc_"):
        return arc_choices(row)
    if task == "mmlu_5shot":
        return mmlu_5shot_choices(row)
    if task == "mmlu":
        return mmlu_choices(row)
    if task == "hellaswag":
        return hellaswag_choices(row)
    if task == "winogrande":
        return winogrande_choices(row)
    if task == "piqa":
        return piqa_choices(row)
    if task == "openbookqa":
        return openbookqa_choices(row)
    if task == "boolq":
        return boolq_choices(row)
    if task == "commonsense_qa":
        return commonsenseqa_choices(row)
    if task in {"race", "race_high"}:
        return race_choices(row)
    if task == "quail":
        return quail_choices(row)
    raise ValueError(task)


def evaluate_mc(bundle: ModelBundle, task: str, ds: Sequence[Any], out_dir: Path, max_seq_len: int, save_every: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    correct_raw = 0
    correct_norm = 0
    started = time.time()
    for idx, row in enumerate(tqdm(ds, desc=f"mc-{task}")):
        prompt, labels, continuations, gold = row_to_mc(task, row)
        scored = []
        for label, continuation in zip(labels, continuations):
            total, avg, n_tokens = score_continuation(bundle, prompt, " " + continuation, max_seq_len=max_seq_len)
            scored.append({"label": label, "text": continuation, "loglik": total, "avg_loglik": avg, "tokens": n_tokens})
        pred_raw = max(scored, key=lambda x: x["loglik"])["label"]
        pred_norm = max(scored, key=lambda x: x["avg_loglik"])["label"]
        correct_raw += int(str(pred_raw).upper() == str(gold).upper())
        correct_norm += int(str(pred_norm).upper() == str(gold).upper())
        rows.append({"idx": idx, "gold": gold, "pred_raw": pred_raw, "pred_norm": pred_norm, "scores": scored})
        if save_every > 0 and (idx + 1) % save_every == 0:
            save_json(out_dir / f"{task}_predictions.partial.json", {"task": task, "predictions": rows})
    total = len(rows)
    report = {
        "task": task,
        "metric": "multiple_choice_loglik",
        "num_samples": total,
        "accuracy_raw": correct_raw / max(1, total),
        "accuracy_norm": correct_norm / max(1, total),
        "elapsed_sec": time.time() - started,
    }
    save_json(out_dir / f"{task}_predictions.json", {"task": task, "predictions": rows})
    return report


def evaluate_wikitext_ppl(
    bundle: ModelBundle,
    ds: Sequence[Any],
    out_dir: Path,
    max_seq_len: int,
    stride: int,
    max_ppl_tokens: int,
    task: str = "wikitext",
) -> Dict[str, Any]:
    text = "\n\n".join(row["text"] for row in ds if isinstance(row.get("text"), str) and row["text"].strip())
    tokenizer = bundle.tokenizer
    device = bundle_device(bundle)
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if max_ppl_tokens is not None and max_ppl_tokens > 0:
        input_ids = input_ids[:, :max_ppl_tokens]
    nll_sum = 0.0
    token_count = 0
    started = time.time()
    prev_end = 0
    windows = []
    for begin in tqdm(range(0, input_ids.shape[1], stride), desc="ppl-wikitext"):
        end = min(begin + max_seq_len, input_ids.shape[1])
        trg_len = end - prev_end
        if trg_len <= 0:
            break
        ids = input_ids[:, begin:end]
        labels = ids.clone()
        labels[:, :-trg_len] = -100
        with torch.no_grad():
            out = bundle.model(input_ids=ids, labels=labels, use_cache=False)
        valid = int((labels[:, 1:] != -100).sum().item())
        nll_sum += float(out.loss.item()) * valid
        token_count += valid
        windows.append({"begin": begin, "end": end, "target_tokens": valid, "loss": float(out.loss.item())})
        prev_end = end
        if end == input_ids.shape[1]:
            break
    ppl = math.exp(nll_sum / max(1, token_count))
    save_json(out_dir / f"{task}_windows.json", {"windows": windows})
    return {
        "task": task,
        "metric": "perplexity",
        "num_tokens": token_count,
        "token_limit": int(max_ppl_tokens) if max_ppl_tokens and max_ppl_tokens > 0 else None,
        "nll": nll_sum,
        "ppl": ppl,
        "elapsed_sec": time.time() - started,
    }


def generate_text(bundle: ModelBundle, prompt: str, max_new_tokens: int) -> str:
    tokenizer = bundle.tokenizer
    device = bundle_device(bundle)
    batch = tokenizer(prompt, return_tensors="pt").to(device)
    # Some tokenizers (e.g. Falcon3) emit token_type_ids, which decoder-only generate() rejects
    # ("model_kwargs are not used: ['token_type_ids']"). Drop it; it is never used for generation.
    batch.pop("token_type_ids", None)
    with torch.no_grad():
        out = bundle.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][batch["input_ids"].shape[1] :], skip_special_tokens=True)


def evaluate_gsm8k(bundle: ModelBundle, ds: Sequence[Any], out_dir: Path, max_new_tokens: int, save_every: int) -> Dict[str, Any]:
    rows = []
    correct = 0
    started = time.time()
    for idx, row in enumerate(tqdm(ds, desc="gen-gsm8k")):
        prompt = f"Question: {row['question']}\nLet's solve this step by step.\n"
        pred = generate_text(bundle, prompt, max_new_tokens)
        gold_num = normalize_number(str(row["answer"]).split("####")[-1])
        pred_num = normalize_number(pred)
        ok = bool(gold_num) and pred_num == gold_num
        correct += int(ok)
        rows.append({"idx": idx, "question": row["question"], "gold": row["answer"], "prediction": pred, "gold_number": gold_num, "pred_number": pred_num, "is_correct": ok})
        if save_every > 0 and (idx + 1) % save_every == 0:
            save_json(out_dir / "gsm8k_predictions.partial.json", {"task": "gsm8k", "predictions": rows})
    total = len(rows)
    save_json(out_dir / "gsm8k_predictions.json", {"task": "gsm8k", "predictions": rows})
    return {"task": "gsm8k", "metric": "final_number_exact_match", "num_samples": total, "accuracy": correct / max(1, total), "elapsed_sec": time.time() - started}


# Canonical MBPP 3-shot demonstrations (Austin et al. 2021 / lm-eval-harness).
# Base models do NOT write a function from a 0-shot description+tests prompt (they
# just continue emitting asserts); the few-shot [BEGIN]...[DONE] pattern is what
# elicits an actual definition, so this is the standard MBPP protocol.
_MBPP_FEWSHOT = [
    ("Write a function to find the shared elements from the given two lists.",
     ["assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))"],
     "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res)"),
    ("Write a python function to identify non-prime numbers.",
     ["assert is_not_prime(2) == False"],
     "import math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result"),
    ("Write a function to find the n largest integers from a given list of numbers, returned in descending order.",
     ["assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]"],
     "import heapq as hq\ndef heap_queue_largest(nums,n):\n  largest_nums = hq.nlargest(n, nums)\n  return largest_nums"),
]


def _mbpp_block(text: str, tests: List[str], code: str | None) -> str:
    body = f"You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n" + "\n".join(tests) + "\n[BEGIN]\n"
    return body + (f"{code}\n[DONE]\n\n" if code is not None else "")


def evaluate_code_generation(bundle: ModelBundle, task: str, ds: Sequence[Any], out_dir: Path, max_new_tokens: int, save_every: int) -> Dict[str, Any]:
    rows = []
    started = time.time()
    mbpp_prefix = "".join(_mbpp_block(t, ts, c) for (t, ts, c) in _MBPP_FEWSHOT)
    for idx, row in enumerate(tqdm(ds, desc=f"gen-{task}")):
        if task == "humaneval":
            prompt = row["prompt"]
            meta = {"task_id": row["task_id"], "entry_point": row["entry_point"], "test": row["test"]}
        else:
            test_list = list(row["test_list"])
            prompt = mbpp_prefix + _mbpp_block(row["text"], test_list, None)
            meta = {"task_id": row["task_id"], "tests": test_list, "challenge_tests": row.get("challenge_test_list", [])}
        pred = generate_text(bundle, prompt, max_new_tokens)
        rows.append({"idx": idx, "prompt": prompt, "prediction": pred, **meta})
        if save_every > 0 and (idx + 1) % save_every == 0:
            save_json(out_dir / f"{task}_generations.partial.json", {"task": task, "generations": rows})
    save_json(out_dir / f"{task}_generations.json", {"task": task, "generations": rows})
    gen_elapsed = time.time() - started
    # NB: pass@1 execution is run AFTER this process finishes generating, in a
    # separate non-CUDA subprocess (see main()). Executing candidate code via
    # multiprocessing from inside this CUDA-heavy process gets the workers
    # SIGKILLed, so generation and execution are deliberately decoupled.
    return {
        "task": task,
        "metric": "generation_saved_pending_pass@1",
        "num_samples": len(rows),
        "generation_elapsed_sec": gen_elapsed,
    }


def maybe_register_mask(bundle: ModelBundle, cfg: Dict[str, Any], mask_run_dir: Optional[str]):
    if not mask_run_dir:
        return None
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
    mask_path = Path(mask_run_dir) / "masks/prune_solution.json"
    with mask_path.open("r", encoding="utf-8") as f:
        selected_units = set(int(x) for x in json.load(f)["selected_units"])
    attn_by_layer: Dict[int, List[Any]] = {}
    ffn_by_layer: Dict[int, List[Any]] = {}
    for spec in registry.units:
        if spec.unit_type == "attn_head":
            attn_by_layer.setdefault(spec.layer_idx, []).append(spec)
        else:
            ffn_by_layer.setdefault(spec.layer_idx, []).append(spec)
    return register_runtime_masks(bundle, attn_by_layer, ffn_by_layer, selected_units)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.model_key is not None:
        cfg["run"] = dict(cfg["run"])
        cfg["run"]["model_key"] = args.model_key
    out_dir = ensure_dir(Path(args.run_dir) / "standard_eval")
    save_json(out_dir / "eval_args.json", vars(args))

    mcfg = dict(model_config(cfg))
    if sum((args.load_in_4bit, args.load_in_8bit, args.bf16)) > 1:
        raise SystemExit("choose only one of --load-in-4bit, --load-in-8bit, and --bf16")
    if args.bf16:
        mcfg.pop("load_in_4bit", None)
        mcfg.pop("load_in_8bit", None)
    if args.load_in_4bit:
        mcfg["load_in_4bit"] = True
        mcfg.pop("load_in_8bit", None)
    if args.load_in_8bit:
        mcfg["load_in_8bit"] = True
        mcfg.pop("load_in_4bit", None)
    bundle = load_model_bundle(mcfg, cfg["model"].get("tokenizer", {}))
    device = bundle_device(bundle)
    run_env = {
        "model_key": cfg["run"]["model_key"],
        "model_path": bundle.model_path,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mask_run_dir": args.mask_run_dir,
        "max_seq_len": args.max_seq_len,
        "ppl_stride": args.ppl_stride,
        "max_ppl_tokens": args.max_ppl_tokens,
    }
    if torch.cuda.is_available():
        dev_idx = device.index if device.type == "cuda" and device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_idx)
        run_env.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_bytes": int(props.total_memory),
            }
        )
    save_json(out_dir / "run_env.json", run_env)
    tasks = list(cfg["eval"]["eval"]["standard_benchmarks"]["tasks"]) if args.tasks == "all" else [x.strip() for x in args.tasks.split(",") if x.strip()]
    cache_dir = cfg["eval"]["eval"]["standard_benchmarks"]["cache_dir"]

    mask_state = maybe_register_mask(bundle, cfg, args.mask_run_dir)
    # Merge into any existing report so a subset re-run (e.g. adding new datasets
    # or refreshing a fixed task) augments rather than clobbers prior results.
    report_path = out_dir / "standard_eval_report.json"
    reports: Dict[str, Any] = {}
    if report_path.exists():
        try:
            reports = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            reports = {}
    try:
        for task in tasks:
            ds = dataset_for_task(task, cache_dir)
            if task in LM_TASKS:
                report = evaluate_wikitext_ppl(bundle, ds, out_dir, args.max_seq_len, args.ppl_stride, args.max_ppl_tokens, task=task)
            elif task in MC_TASKS:
                report = evaluate_mc(bundle, task, maybe_limit(ds, args.max_samples), out_dir, args.max_seq_len, args.save_every)
            elif task == "gsm8k":
                report = evaluate_gsm8k(bundle, maybe_limit(ds, args.max_generation_samples), out_dir, args.max_new_tokens, args.save_every)
            elif task in {"humaneval", "mbpp"}:
                report = evaluate_code_generation(bundle, task, maybe_limit(ds, args.max_generation_samples), out_dir, args.max_new_tokens, args.save_every)
            else:
                continue
            reports[task] = report
            save_json(out_dir / "standard_eval_report.partial.json", reports)
    finally:
        if mask_state is not None:
            clear_runtime_masks(mask_state)

    save_json(out_dir / "standard_eval_report.json", reports)

    # Score coding pass@1 now that the GPU/model work is done, in a separate
    # non-CUDA process (executing candidate code from this process gets SIGKILLed).
    if any(t in reports for t in ("humaneval", "mbpp")):
        import subprocess
        import sys as _sys

        try:
            subprocess.run(
                [_sys.executable, "scripts/score_coding_pass_at_1.py", "--run-dir", str(args.run_dir),
                 "--timeout", "8", "--workers", "8"],
                check=False,
            )
            reports = json.loads((out_dir / "standard_eval_report.json").read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] coding pass@1 scoring failed: {exc}")

    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
