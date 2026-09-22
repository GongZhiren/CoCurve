#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


SOURCE_SPECS: Dict[str, Dict[str, Optional[str]]] = {
    "c4": {"dataset_id": "allenai/c4", "config": "en", "split": "train", "text_key": "text"},
    "wikitext2": {"dataset_id": "wikitext", "config": "wikitext-2-raw-v1", "split": "train", "text_key": "text"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pruning calibration set from standard language-modeling corpora. "
            "The default follows common LLM pruning practice: 128 token-packed C4 samples."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="data/calibration/calibration_mix.jsonl")
    parser.add_argument("--holdout-output", default="data/calibration/calibration_holdout.jsonl")
    parser.add_argument("--manifest", default="data/calibration/calibration_manifest.json")
    parser.add_argument("--source", default="c4", choices=sorted(SOURCE_SPECS))
    parser.add_argument("--fallback-source", default="c4", choices=sorted(SOURCE_SPECS))
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--holdout-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--min-words", type=int, default=32)
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--local-files-only", action="store_true", help="Only load the tokenizer from local files.")
    parser.add_argument("--no-fallback", action="store_true", help="Fail instead of falling back if the primary source is unavailable.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_multiple_choice_prompt(text: str) -> bool:
    upper = text.upper()
    markers = (" A.", " A)", "\nA.", "\nA)", "CHOICES:", "OPTION A")
    return any(marker in upper for marker in markers)


def load_tokenizer(tokenizer_path: Path, local_files_only: bool):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError(f"transformers import failed: {exc}") from exc

    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=local_files_only, use_fast=True)
    except Exception:
        if local_files_only:
            raise
        return AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)


def load_stream(source: str, seed: int, shuffle_buffer: int) -> Tuple[Iterable[Dict[str, Any]], Dict[str, Optional[str]]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError(f"datasets import failed: {exc}") from exc

    spec = SOURCE_SPECS[source]
    kwargs: Dict[str, Any] = {"split": spec["split"], "streaming": True}
    if spec["config"] is None:
        ds = load_dataset(spec["dataset_id"], **kwargs)
    else:
        ds = load_dataset(spec["dataset_id"], spec["config"], **kwargs)
    if hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return ds, spec


def iter_clean_documents(dataset: Iterable[Dict[str, Any]], text_key: str, min_words: int) -> Iterator[str]:
    for item in dataset:
        raw = item.get(text_key, "")
        if not isinstance(raw, str):
            continue
        text = normalize_text(raw)
        if len(text.split()) < min_words:
            continue
        if looks_like_multiple_choice_prompt(text):
            continue
        yield text


def build_token_packed_samples(
    docs: Iterable[str],
    tokenizer: Any,
    num_samples: int,
    seq_len: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    eos_id = tokenizer.eos_token_id
    token_buffer: List[int] = []
    rows: List[Dict[str, Any]] = []
    consumed_docs = 0
    consumed_tokens = 0

    for doc in docs:
        consumed_docs += 1
        token_ids = tokenizer.encode(doc, add_special_tokens=False)
        if not token_ids:
            continue
        token_buffer.extend(token_ids)
        if eos_id is not None:
            token_buffer.append(int(eos_id))
        consumed_tokens += len(token_ids)

        while len(token_buffer) >= seq_len and len(rows) < num_samples:
            chunk = token_buffer[:seq_len]
            del token_buffer[:seq_len]
            text = tokenizer.decode(chunk, skip_special_tokens=True).strip()
            if text:
                rows.append(
                    {
                        "text": text,
                        "source": "standard_lm_calibration",
                        "token_length": seq_len,
                    }
                )
        if len(rows) >= num_samples:
            break

    return rows, {"consumed_docs": consumed_docs, "consumed_tokens": consumed_tokens}


def build_char_window_samples(
    docs: Iterable[str],
    num_samples: int,
    seq_len: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    consumed_docs = 0
    target_chars = max(4096, seq_len * 4)
    for doc in docs:
        consumed_docs += 1
        if len(doc) <= target_chars:
            text = doc
        else:
            start = rng.randint(0, len(doc) - target_chars)
            text = doc[start : start + target_chars]
        rows.append({"text": text.strip(), "source": "standard_lm_calibration", "token_length": None})
        if len(rows) >= num_samples:
            break
    return rows, {"consumed_docs": consumed_docs, "consumed_tokens": 0}


def try_build(
    source: str,
    args: argparse.Namespace,
    tokenizer: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dataset, spec = load_stream(source, seed=int(args.seed), shuffle_buffer=int(args.shuffle_buffer))
    docs = iter_clean_documents(dataset, text_key=str(spec["text_key"]), min_words=int(args.min_words))
    total_samples = int(args.num_samples) + int(args.holdout_samples)
    if tokenizer is not None:
        rows, stats = build_token_packed_samples(docs, tokenizer, total_samples, int(args.seq_len))
        method = "token_packed"
    else:
        rows, stats = build_char_window_samples(docs, total_samples, int(args.seq_len), int(args.seed))
        method = "char_window"
    if len(rows) < total_samples:
        raise RuntimeError(f"{source} yielded only {len(rows)} usable samples; need {total_samples}")
    meta = {
        "source": source,
        "dataset_id": spec["dataset_id"],
        "config": spec["config"],
        "split": spec["split"],
        "text_key": spec["text_key"],
        "method": method,
        **stats,
    }
    return rows, meta


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output = root / args.output
    holdout_output = root / args.holdout_output
    manifest_path = root / args.manifest
    local_tokenizer = root / args.tokenizer
    tokenizer_source = str(local_tokenizer) if local_tokenizer.exists() else args.tokenizer

    tokenizer = None
    tokenizer_error = None
    try:
        tokenizer = load_tokenizer(Path(tokenizer_source), local_files_only=bool(args.local_files_only))
    except Exception as exc:
        tokenizer_error = str(exc)
        raise RuntimeError(f"tokenizer is required for exact token-packed calibration: {exc}") from exc

    attempted: List[Dict[str, Any]] = []
    build_error: Optional[Exception] = None
    for source in [args.source] + ([] if args.no_fallback or args.fallback_source == args.source else [args.fallback_source]):
        try:
            rows, source_meta = try_build(source, args, tokenizer)
            source_meta["status"] = "ok"
            attempted.append(source_meta)
            break
        except Exception as exc:
            attempted.append({"source": source, "status": "error", "error": str(exc)})
            build_error = exc
    else:
        raise RuntimeError(f"Could not build calibration set. Attempts: {attempted}") from build_error

    train_rows = rows[: int(args.num_samples)]
    holdout_rows = rows[int(args.num_samples) :]
    count = write_jsonl(output, train_rows)
    holdout_count = write_jsonl(holdout_output, holdout_rows)
    manifest = {
        "status": "ok",
        "purpose": "LLM pruning calibration, not evaluation",
        "output": str(output.relative_to(root)),
        "holdout_output": str(holdout_output.relative_to(root)),
        "num_samples": count,
        "holdout_samples": holdout_count,
        "seq_len": int(args.seq_len),
        "seed": int(args.seed),
        "tokenizer": tokenizer_source,
        "tokenizer_loaded": tokenizer is not None,
        "tokenizer_error": tokenizer_error,
        "policy": {
            "primary_source": args.source,
            "fallback_source": None if args.no_fallback else args.fallback_source,
            "allowed_sources": sorted(SOURCE_SPECS),
            "reject_multiple_choice_prompts": True,
            "min_words": int(args.min_words),
        },
        "selected_source": attempted[-1],
        "attempted_sources": attempted,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "holdout_output": str(holdout_output),
                "manifest": str(manifest_path),
                "num_samples": count,
                "holdout_samples": holdout_count,
                "source": attempted[-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
