from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr

from .benchmark_scoring import (
    pick_mc_loglik_prediction,
    prediction_artifact_relpath,
    score_gsm8k_prediction,
)
from .calibration import load_calibration_texts, tokenize_batch
from .io import ArtifactStore
from .model import ModelBundle, bundle_device, forward_logits, kl_from_top_r, top_r_distribution
from .prompting import detect_task_type, render_prompt
from .prune import apply_physical_prune_inplace, clear_runtime_masks, register_runtime_masks, restore_physical_prune
from .scoring import evaluate_prediction
from .types import QualityGateReport, UnitSpec


def _correlation_pair(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return 0.0, 0.0
    s = float(spearmanr(x, y).correlation)
    p = float(pearsonr(x, y).statistic)
    if np.isnan(s):
        s = 0.0
    if np.isnan(p):
        p = 0.0
    return s, p


def _layer_maps(units: List[UnitSpec]) -> Tuple[Dict[int, List[UnitSpec]], Dict[int, List[UnitSpec]]]:
    attn: Dict[int, List[UnitSpec]] = {}
    ffn: Dict[int, List[UnitSpec]] = {}
    for spec in units:
        if spec.unit_type == "attn_head":
            attn.setdefault(spec.layer_idx, []).append(spec)
        else:
            ffn.setdefault(spec.layer_idx, []).append(spec)
    return attn, ffn


def _read_text_jsonl(path: Path, limit: int) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= limit:
                break
            row = json.loads(line)
            text = row.get("text", "")
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return texts


def _quality_gate_texts(cfg: Dict[str, object], gate_cfg: Dict[str, object]) -> Tuple[List[str], str]:
    eval_samples = int(gate_cfg["quality_eval_samples"])
    cal_cfg = cfg["calibration"]["calibration"]
    if bool(gate_cfg.get("use_calibration_holdout", False)):
        holdout_path = cal_cfg.get("holdout_path")
        if holdout_path and Path(str(holdout_path)).exists():
            texts = _read_text_jsonl(Path(str(holdout_path)), eval_samples)
            if texts:
                return texts, str(holdout_path)
    calibration_texts = load_calibration_texts(cfg["calibration"])
    return calibration_texts[: min(eval_samples, len(calibration_texts))], "calibration_train_fallback"


def _prepare_quality_batches(
    bundle: ModelBundle,
    texts: List[str],
    max_seq_len: int,
    top_r: int,
    batch_size: int,
) -> List[Dict[str, torch.Tensor]]:
    prepared: List[Dict[str, torch.Tensor]] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenize_batch(bundle, texts[start : start + batch_size], max_seq_len)
        logits_base = forward_logits(bundle, batch["input_ids"], batch["attention_mask"])
        teacher_top_indices, teacher_top_probs = top_r_distribution(logits_base[:, :-1, :], top_r=top_r)
        prepared.append(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "teacher_top_indices": teacher_top_indices.detach().cpu(),
                "teacher_top_probs": teacher_top_probs.detach().float().cpu(),
                "token_positions": batch["attention_mask"][:, 1:].detach().cpu(),
            }
        )
        del logits_base, teacher_top_indices, teacher_top_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return prepared


def run_quality_gates(
    bundle: ModelBundle,
    cfg: Dict[str, object],
    store: ArtifactStore,
    units: List[UnitSpec],
    selected_units: Sequence[int],
) -> QualityGateReport:
    h = np.load(store.run_dir / "matrices/H.npy")
    diag = np.diag(h)
    single_unit_kl = np.load(store.run_dir / "cache/single_unit_kl.npy")
    device = bundle_device(bundle)
    top_r = int(np.load(store.run_dir / "cache/top_indices.npy", mmap_mode="r").shape[-1])

    gate_cfg = cfg["eval"]["eval"]["quality_gates"]
    subset_count = int(gate_cfg["surrogate_subsets"])
    subset_fraction = float(gate_cfg["surrogate_subset_fraction"])
    # Vary subset SIZE across a range so the surrogate 0.5 s^T H s spans a wide
    # magnitude band. With a single fixed fraction, random subsets of a roughly
    # homogeneous pruned set have ~2% CV, so the surrogate-vs-real correlation
    # measures noise rather than fidelity. A size range is the standard way to
    # validate a second-order (Taylor) surrogate across perturbation magnitudes.
    subset_fraction_min = float(gate_cfg.get("surrogate_subset_fraction_min", 0.05))
    subset_fraction_max = float(gate_cfg.get("surrogate_subset_fraction_max", subset_fraction))
    q_threshold = float(gate_cfg["min_corr_threshold"])
    max_err_threshold = float(gate_cfg["max_mask_physical_abs_err"])
    quality_batch_size = int(gate_cfg.get("quality_batch_size", 1))

    surrogates: List[float] = []
    reals: List[float] = []

    rng = np.random.default_rng(123)
    all_pruned = list(set(range(len(units))) - set(selected_units))
    attn_by_layer, ffn_by_layer = _layer_maps(units)
    texts, quality_source = _quality_gate_texts(cfg, gate_cfg)
    quality_batches = _prepare_quality_batches(
        bundle=bundle,
        texts=texts,
        max_seq_len=int(cfg["calibration"]["calibration"]["max_seq_len"]),
        top_r=top_r,
        batch_size=quality_batch_size,
    )

    lo, hi = sorted((subset_fraction_min, subset_fraction_max))
    for _ in range(min(subset_count, max(1, len(all_pruned)))):
        frac = float(rng.uniform(lo, hi)) if hi > lo else lo
        sample_size = max(1, min(len(all_pruned), int(round(len(all_pruned) * frac))))
        subset = sorted(rng.choice(all_pruned, size=sample_size, replace=False).tolist())
        s = np.zeros((len(units),), dtype=np.float32)
        s[subset] = 1.0
        surrogates.append(float(0.5 * s @ h @ s))

        kept = set(range(len(units))) - set(subset)
        mask_state = register_runtime_masks(bundle, attn_by_layer, ffn_by_layer, kept)
        real_values: List[torch.Tensor] = []
        try:
            for quality_batch in quality_batches:
                logits = forward_logits(bundle, quality_batch["input_ids"], quality_batch["attention_mask"])
                t_idx = quality_batch["teacher_top_indices"].to(device)
                t_prob = quality_batch["teacher_top_probs"].to(device)
                pos = quality_batch["token_positions"].to(device).float()
                kl = kl_from_top_r(t_idx, t_prob, logits[:, :-1, :])
                real_values.append((kl * pos).sum(dim=1) / pos.sum(dim=1).clamp_min(1.0))
                del logits, t_idx, t_prob, pos, kl
            reals.append(float(torch.cat(real_values).mean().item()))
        finally:
            clear_runtime_masks(mask_state)

    surrogate_s, surrogate_p = _correlation_pair(np.array(surrogates), np.array(reals))
    diagonal_s, _ = _correlation_pair(diag * 0.5, single_unit_kl)

    # mask-vs-physical equivalence without persistent model mutation
    selected_set: Set[int] = set(selected_units)
    equivalence_batch = quality_batches[0]
    mask_state = register_runtime_masks(bundle, attn_by_layer, ffn_by_layer, selected_set)
    masked_logits = forward_logits(bundle, equivalence_batch["input_ids"], equivalence_batch["attention_mask"])
    clear_runtime_masks(mask_state)

    backups = apply_physical_prune_inplace(bundle, attn_by_layer, ffn_by_layer, selected_set)
    physical_logits = forward_logits(bundle, equivalence_batch["input_ids"], equivalence_batch["attention_mask"])
    restore_physical_prune(backups)
    max_abs_err = float((masked_logits - physical_logits).abs().max().item())

    passed = (
        surrogate_s > q_threshold
        and surrogate_p > q_threshold
        and diagonal_s > q_threshold
        and max_abs_err < max_err_threshold
    )
    report = QualityGateReport(
        surrogate_vs_real_spearman=surrogate_s,
        surrogate_vs_real_pearson=surrogate_p,
        diagonal_damage_spearman=diagonal_s,
        mask_vs_physical_max_abs_err=max_abs_err,
        passed=passed,
    )
    store.save_json(
        "eval/quality_gate_report.json",
        {
            "surrogate_vs_real_spearman": report.surrogate_vs_real_spearman,
            "surrogate_vs_real_pearson": report.surrogate_vs_real_pearson,
            "diagonal_damage_spearman": report.diagonal_damage_spearman,
            "mask_vs_physical_max_abs_err": report.mask_vs_physical_max_abs_err,
            "thresholds": {
                "min_corr_threshold": q_threshold,
                "max_mask_physical_abs_err": max_err_threshold,
            },
            "quality_source": quality_source,
            "quality_eval_samples": len(texts),
            "quality_batch_size": quality_batch_size,
            "passed": report.passed,
        },
    )
    return report


def _iter_json_records(path: Path) -> Iterable[Dict[str, object]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    else:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
        elif isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                for item in payload["data"]:
                    if isinstance(item, dict):
                        yield item


def _choice_label(idx: int) -> str:
    return chr(ord("A") + idx)


def _choices_from_record(record: Dict[str, object]) -> List[Tuple[str, str]]:
    choices = record.get("choices")
    if isinstance(choices, list):
        return [(_choice_label(i), str(choice)) for i, choice in enumerate(choices)]
    raw = record.get("raw")
    if isinstance(raw, dict):
        raw_choices = raw.get("choices")
        if isinstance(raw_choices, dict):
            texts = raw_choices.get("text")
            labels = raw_choices.get("label")
            if isinstance(texts, list):
                if not isinstance(labels, list) or len(labels) != len(texts):
                    labels = [_choice_label(i) for i in range(len(texts))]
                return [(str(label), str(text)) for label, text in zip(labels, texts)]
        endings = raw.get("endings")
        if isinstance(endings, list):
            return [(_choice_label(i), str(choice)) for i, choice in enumerate(endings)]
        if "option1" in raw or "option2" in raw:
            out = []
            for i in range(1, 6):
                key = f"option{i}"
                if key in raw:
                    out.append((_choice_label(i - 1), str(raw[key])))
            return out
    return []


def _normalize_answer_label(answer: object, choices: List[Tuple[str, str]], record: Dict[str, object]) -> str:
    raw = str(answer).strip()
    if not choices:
        return raw
    labels = [label.upper() for label, _ in choices]
    if raw.upper() in labels:
        return raw.upper()
    if raw.isdigit():
        numeric = int(raw)
        raw_record = record.get("raw")
        dataset_source = str(record.get("dataset_source", "")).lower()
        if isinstance(raw_record, dict) and isinstance(raw_record.get("endings"), list):
            # HellaSwag stores labels as 0-based indices.
            idx = numeric
        elif "hellaswag" in dataset_source:
            idx = numeric
        elif isinstance(raw_record, dict) and any(f"option{i}" in raw_record for i in range(1, 6)):
            # WinoGrande-style option labels are 1-based.
            idx = numeric - 1
        else:
            # Default for common MC datasets that expose labels as 1..N.
            idx = numeric - 1
        if 0 <= idx < len(choices):
            return choices[idx][0].upper()
    for label, text in choices:
        if raw.lower() == text.strip().lower():
            return label.upper()
    return raw


def _prompt_has_options(prompt: str) -> bool:
    upper = prompt.upper()
    return any(marker in upper for marker in ("A.", "A)", "OPTION A", "CHOICES:"))


def _extract_base_prompt(record: Dict[str, object]) -> str:
    prompt_keys = ["prompt", "question", "query", "instruction", "text", "input"]
    for key in prompt_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_prompt_and_answer(record: Dict[str, object]) -> Tuple[str, str]:
    prompt = _extract_base_prompt(record)
    answer = ""
    choices = _choices_from_record(record)
    if choices and not _prompt_has_options(prompt):
        option_text = "\n".join(f"{label}. {text}" for label, text in choices)
        prompt = f"{prompt}\n\nChoices:\n{option_text}"
    answer_keys = ["answer", "target", "gold", "label", "output"]
    for key in answer_keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            answer = _normalize_answer_label(value, choices, record)
            break
    return prompt, answer


def _is_gsm8k_answer(answer: str) -> bool:
    return "####" in answer


def _mc_loglik_inputs(record: Dict[str, object]) -> Tuple[str, List[Tuple[str, str]], str] | None:
    raw = record.get("raw")
    answer_keys = ["answer", "target", "gold", "label", "output"]
    answer = ""
    for key in answer_keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            answer = str(value).strip()
            break
    if not answer:
        return None

    if isinstance(raw, dict) and "sentence" in raw and "option1" in raw:
        sentence = str(raw["sentence"])
        if "_" not in sentence:
            return None
        prefix, suffix = sentence.split("_", 1)
        choices = [("1", str(raw["option1"]) + suffix), ("2", str(raw["option2"]) + suffix)]
        gold = _normalize_answer_label(answer, choices, record)
        return prefix, choices, gold

    if isinstance(raw, dict) and isinstance(raw.get("endings"), list):
        prompt = str(raw.get("ctx") or _extract_base_prompt(record))
        endings = [str(x) for x in raw["endings"]]
        choices = [(_choice_label(i), ending) for i, ending in enumerate(endings)]
        gold = _normalize_answer_label(answer, choices, record)
        return prompt, choices, gold

    choices = _choices_from_record(record)
    if len(choices) < 2:
        return None
    base = _extract_base_prompt(record)
    if not base:
        return None
    prompt = f"Question: {base}\nAnswer:" if "?" in base else f"{base}\nAnswer:"
    gold = _normalize_answer_label(answer, choices, record)
    return prompt, choices, gold


def _path_matches_globs(path: Path, globs: Sequence[str]) -> bool:
    path_str = str(path).replace("\\", "/")
    for pattern in globs:
        if path.match(pattern):
            return True
        # Also allow matching against full relative path strings.
        normalized = pattern.replace("**/", "").replace("**", "")
        if normalized and normalized in path_str:
            return True
    return False


def _discover_benchmark_files(root: Path, pattern: str, bench_cfg: Dict[str, object]) -> List[Path]:
    exclude_globs = [str(x) for x in bench_cfg.get("exclude_globs", [])]
    exclude_paths = {str(x) for x in bench_cfg.get("exclude_paths", [])}
    files: List[Path] = []
    for file in sorted(root.rglob(pattern)):
        if str(file) in exclude_paths:
            continue
        if exclude_globs and _path_matches_globs(file, exclude_globs):
            continue
        files.append(file)
    return files


def _extract_records(record: Dict[str, object]) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    prompt, answer = _extract_prompt_and_answer(record)
    if prompt:
        rows.append((prompt, answer, detect_task_type(record)))
    turns = record.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            p, a = _extract_prompt_and_answer(turn)
            if p:
                rows.append((p, a, detect_task_type(turn)))
    return rows


def _evaluate_json_record(
    bundle: ModelBundle,
    record: Dict[str, object],
    prompt: str,
    answer: str,
    task_type: str,
    eval_limits: Dict[str, object],
    use_chat_template: bool,
) -> Tuple[float, Dict[str, object], float]:
    device = bundle_device(bundle)
    max_seq_len = int(eval_limits.get("max_seq_len", 2048))
    mc_use_loglik = bool(eval_limits.get("mc_use_loglik", True))

    if mc_use_loglik and task_type == "multiple_choice":
        mc_inputs = _mc_loglik_inputs(record)
        if mc_inputs is not None:
            mc_prompt, choices, gold = mc_inputs
            start = time.time()
            pred_raw, pred_norm, scored = pick_mc_loglik_prediction(bundle, mc_prompt, choices, max_seq_len=max_seq_len)
            elapsed = time.time() - start
            pred = pred_norm
            ok = str(pred).upper() == str(gold).upper()
            score = 1.0 if ok else 0.0
            payload = {
                "prompt": prompt,
                "gold": gold,
                "prediction": pred,
                "prediction_raw": pred_raw,
                "task_type": task_type,
                "score": score,
                "metric": "multiple_choice_loglik",
                "choice_scores": scored,
            }
            return score, payload, elapsed

    if task_type == "reasoning" and _is_gsm8k_answer(answer):
        rendered = f"Question: {prompt}\nLet's solve this step by step.\n"
        if use_chat_template:
            rendered, _ = render_prompt(bundle, rendered, task_type="reasoning", use_chat_template=True)
        batch = bundle.tokenizer(rendered, return_tensors="pt").to(device)
        batch.pop("token_type_ids", None)  # decoder-only generate() rejects it (e.g. Falcon3)
        gen_kwargs: Dict[str, object] = {
            "max_new_tokens": int(eval_limits["max_new_tokens"]),
            "do_sample": False,
            "pad_token_id": bundle.tokenizer.eos_token_id,
        }
        rep_penalty = float(eval_limits.get("repetition_penalty", 1.0))
        if rep_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = rep_penalty
        start = time.time()
        with torch.no_grad():
            out = bundle.model.generate(**batch, **gen_kwargs)
        elapsed = time.time() - start
        generated = bundle.tokenizer.decode(out[0][batch["input_ids"].shape[1] :], skip_special_tokens=True)
        score_payload = score_gsm8k_prediction(generated, answer)
        payload = {
            "prompt": prompt,
            "gold": answer,
            "prediction": generated,
            "task_type": task_type,
            "score": float(score_payload["score"]),
            "metric": score_payload["metric"],
            "gold_number": score_payload.get("gold_number"),
            "pred_number": score_payload.get("pred_number"),
        }
        return float(score_payload["score"]), payload, elapsed

    rendered_prompt, _ = render_prompt(bundle, prompt, task_type, use_chat_template=use_chat_template)
    batch = bundle.tokenizer(rendered_prompt, return_tensors="pt").to(device)
    batch.pop("token_type_ids", None)  # decoder-only generate() rejects it (e.g. Falcon3)
    gen_kwargs = {
        "max_new_tokens": int(eval_limits["max_new_tokens"]),
        "do_sample": False,
        "pad_token_id": bundle.tokenizer.eos_token_id,
    }
    rep_penalty = float(eval_limits.get("repetition_penalty", 1.0))
    if rep_penalty > 1.0:
        gen_kwargs["repetition_penalty"] = rep_penalty
    start = time.time()
    with torch.no_grad():
        out = bundle.model.generate(**batch, **gen_kwargs)
    elapsed = time.time() - start
    generated = bundle.tokenizer.decode(out[0][batch["input_ids"].shape[1] :], skip_special_tokens=True)
    score_payload = evaluate_prediction(generated, answer, task_type)
    payload = {
        "prompt": prompt,
        "gold": answer,
        "prediction": generated,
        "task_type": task_type,
        "score": float(score_payload["score"]),
        "metric": score_payload.get("metric"),
    }
    return float(score_payload["score"]), payload, elapsed


def evaluate_json_benchmarks(
    bundle: ModelBundle,
    cfg: Dict[str, object],
    store: ArtifactStore,
    selected_units: Sequence[int],
    units: List[UnitSpec],
) -> Dict[str, object]:
    eval_cfg = cfg["eval"]["eval"]["json_benchmarks"]
    eval_limits = cfg["eval"]["eval"]["json_eval_limits"]
    use_chat_template = bool(eval_limits.get("use_chat_template", False))
    attn_by_layer, ffn_by_layer = _layer_maps(units)
    selected_set = set(selected_units)
    results: Dict[str, object] = {}
    seen_files: Set[str] = set()

    mask_state = register_runtime_masks(bundle, attn_by_layer, ffn_by_layer, selected_set)
    try:
        for bench_name, bench_cfg in eval_cfg.items():
            if not bench_cfg.get("enabled", False):
                continue
            root = Path(str(bench_cfg["path"]))
            pattern = str(bench_cfg.get("pattern", "*.jsonl"))
            files = _discover_benchmark_files(root, pattern, bench_cfg)
            if int(eval_limits["max_files_per_benchmark"]) > 0:
                files = files[: int(eval_limits["max_files_per_benchmark"])]
            bench_scores: List[float] = []
            file_stats: Dict[str, object] = {}

            for file in files:
                file_key = str(file.resolve())
                if file_key in seen_files:
                    continue
                seen_files.add(file_key)

                records: List[Tuple[str, str, str, Dict[str, object]]] = []
                for row in _iter_json_records(file):
                    for prompt, answer, task_type in _extract_records(row):
                        records.append((prompt, answer, task_type, row))
                        if len(records) >= int(eval_limits["max_samples_per_file"]):
                            break
                    if len(records) >= int(eval_limits["max_samples_per_file"]):
                        break
                if not records:
                    continue

                file_scores: List[float] = []
                prediction_rows: List[Dict[str, object]] = []
                latency_values: List[float] = []
                for prompt, answer, task_type, row in records:
                    score, payload, elapsed = _evaluate_json_record(
                        bundle=bundle,
                        record=row,
                        prompt=prompt,
                        answer=answer,
                        task_type=task_type,
                        eval_limits=eval_limits,
                        use_chat_template=use_chat_template,
                    )
                    file_scores.append(score)
                    latency_values.append(elapsed)
                    if bool(cfg["eval"]["eval"].get("reporting", {}).get("save_predictions", True)):
                        prediction_rows.append(payload)

                acc = float(np.mean(file_scores)) if file_scores else 0.0
                bench_scores.append(acc)
                file_stats[str(file)] = {
                    "num_samples": len(records),
                    "score": acc,
                    "avg_latency_sec": float(np.mean(latency_values)) if latency_values else 0.0,
                }
                if prediction_rows:
                    pred_name = prediction_artifact_relpath(file, root)
                    store.save_json(
                        f"eval/predictions/{bench_name}/{pred_name}.json",
                        {"file": str(file), "predictions": prediction_rows},
                    )

            results[bench_name] = {
                "mean_score": float(np.mean(bench_scores)) if bench_scores else 0.0,
                "files_evaluated": len(file_stats),
                "files": file_stats,
            }
    finally:
        clear_runtime_masks(mask_state)

    store.save_json("eval/json_benchmark_report.json", results)
    return results


def prepare_standard_benchmarks(cfg: Dict[str, object], store: ArtifactStore) -> Dict[str, object]:
    ds_cfg = cfg["datasets"]["datasets"]["standard"]
    enabled = cfg["eval"]["eval"]["standard_benchmarks"]["enabled"]
    prepared: Dict[str, object] = {}
    if not enabled:
        return prepared

    cache_dir = Path(cfg["eval"]["eval"]["standard_benchmarks"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    for task in cfg["eval"]["eval"]["standard_benchmarks"]["tasks"]:
        if task not in ds_cfg:
            continue
        task_cfg = ds_cfg[task]
        hf_name = task_cfg["hf_name"]
        split = task_cfg.get("split", "test")
        config = task_cfg.get("config")
        kwargs = {"split": split, "cache_dir": str(cache_dir)}
        if config is not None:
            dataset = load_dataset(hf_name, config, **kwargs)
        else:
            dataset = load_dataset(hf_name, **kwargs)
        prepared[task] = {"rows": int(len(dataset)), "hf_name": hf_name, "split": split}
    store.save_json("eval/standard_benchmark_prepared.json", prepared)
    return prepared
