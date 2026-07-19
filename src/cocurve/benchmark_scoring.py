from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from .model import ModelBundle, bundle_device


def normalize_number(text: str) -> str:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else ""


def score_continuation(
    bundle: ModelBundle,
    prompt: str,
    continuation: str,
    max_seq_len: int,
) -> Tuple[float, float, int]:
    tokenizer = bundle.tokenizer
    device = bundle_device(bundle)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not continuation_ids:
        return -1e30, -1e30, 0
    input_ids = prompt_ids + continuation_ids
    continuation_start = len(prompt_ids)
    if len(input_ids) > max_seq_len:
        overflow = len(input_ids) - max_seq_len
        input_ids = input_ids[overflow:]
        continuation_start = max(0, continuation_start - overflow)
    if continuation_start >= len(input_ids):
        return -1e30, -1e30, 0

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    with torch.no_grad():
        logits = bundle.model(input_ids=ids, attention_mask=attention, use_cache=False).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = ids[:, 1:]
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    start = max(0, continuation_start - 1)
    selected = token_log_probs[0, start:]
    total = float(selected.sum().item())
    count = int(selected.numel())
    avg = total / max(1, count)
    return total, avg, count


def pick_mc_loglik_prediction(
    bundle: ModelBundle,
    prompt: str,
    choices: Sequence[Tuple[str, str]],
    max_seq_len: int,
) -> Tuple[str, str, List[Dict[str, object]]]:
    scored: List[Dict[str, object]] = []
    for label, continuation in choices:
        cont = continuation if continuation.startswith(" ") else f" {continuation}"
        total, avg, n_tokens = score_continuation(bundle, prompt, cont, max_seq_len=max_seq_len)
        scored.append(
            {
                "label": label,
                "text": continuation,
                "loglik": total,
                "avg_loglik": avg,
                "tokens": n_tokens,
            }
        )
    pred_raw = str(max(scored, key=lambda x: x["loglik"])["label"])
    pred_norm = str(max(scored, key=lambda x: x["avg_loglik"])["label"])
    return pred_raw, pred_norm, scored


def score_gsm8k_prediction(prediction: str, answer: str) -> Dict[str, float | bool | str]:
    gold_num = normalize_number(str(answer).split("####")[-1])
    pred_num = normalize_number(prediction)
    ok = bool(gold_num) and pred_num == gold_num
    return {
        "is_correct": ok,
        "score": 1.0 if ok else 0.0,
        "metric": "final_number_exact_match",
        "gold_number": gold_num,
        "pred_number": pred_num,
    }


def prediction_artifact_relpath(file_path: Path, root: Path) -> str:
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        rel = file_path
    parts = [part.replace(" ", "_") for part in rel.parts]
    stem = parts[-1]
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    elif stem.endswith(".json"):
        stem = stem[: -len(".json")]
    prefix = "__".join(parts[:-1])
    return f"{prefix}__{stem}" if prefix else stem
