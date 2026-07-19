from __future__ import annotations

import re
from typing import Dict


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_mc(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    head = re.match(r"^\s*\(?\s*([A-E])\s*\)?(?:[\s.:,\-]|$)", stripped, flags=re.IGNORECASE)
    if head:
        return head.group(1).upper()
    match = re.search(r"\b([A-E])\b", stripped.upper())
    return match.group(1) if match else ""


def evaluate_prediction(prediction: str, answer: str, task_type: str) -> Dict[str, float | bool | str]:
    pred = prediction or ""
    gold = answer or ""
    if not gold:
        return {"is_correct": False, "score": 0.0, "metric": "missing_gold"}

    if task_type == "multiple_choice":
        p = _extract_mc(pred)
        g = _extract_mc(gold)
        ok = p != "" and p == g
        return {"is_correct": ok, "score": 1.0 if ok else 0.0, "metric": "mc_letter_match"}

    pred_n = _normalize(pred)
    gold_n = _normalize(gold)
    exact = pred_n == gold_n
    contain = gold_n in pred_n if gold_n else False
    score = 1.0 if exact else (0.5 if contain else 0.0)
    return {"is_correct": score > 0.0, "score": score, "metric": "exact_or_contains"}
