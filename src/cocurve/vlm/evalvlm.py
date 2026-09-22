"""Evaluation for pruned VLMs: multiple-choice accuracy and caption perplexity.

Scoring mirrors the text-only harness as closely as the modality allows. Every task is
multiple-choice and is scored by comparing the model's logit for each option *letter* at the answer
position -- one forward pass per item, deterministic, no generation and no judge. The letter
protocol is the one these benchmarks are built for, so the numbers are readable against published
ones. Caption perplexity on held-out images is the language-modelling axis, the closest analogue of
WikiText perplexity for a model whose inputs include pixels.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

LETTERS = "ABCDEFGH"

# Qwen's processor accepts up to 12.8M pixels per image by default, which is ~16k visual tokens; a
# batch of eight such images asks the language head to materialise a 152k-wide logit row for every one
# of ~15k positions, and it OOMs a 48 GB card. Capping the pixel budget bounds the visual token count
# instead. The cap applies to every arm identically, dense included, so it moves the level of every
# number by the same amount and none of the comparisons.
MAX_PIXELS = 768 * 28 * 28


def cap_pixels(bundle, max_pixels: int = MAX_PIXELS) -> None:
    ip = getattr(bundle.processor, "image_processor", None)
    if ip is None:
        return
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = int(max_pixels)
    sz = getattr(ip, "size", None)
    if isinstance(sz, dict) and "longest_edge" in sz:
        sz["longest_edge"] = int(max_pixels)

TASKS: Dict[str, Dict] = {
    "mmbench": dict(repo="lmms-lab/MMBench_EN", split="dev", kind="abcd"),
    "seedbench": dict(repo="lmms-lab/SEED-Bench", split="test", kind="choice_x"),
    "scienceqa": dict(repo="lmms-lab/ScienceQA", cfg="ScienceQA-IMG", split="test", kind="choices"),
    "ai2d": dict(repo="lmms-lab/ai2d", split="test", kind="options"),
    "mmstar": dict(repo="Lin-Chen/MMStar", cfg="val", split="val", kind="inline"),
    # POPE and MME are the yes/no half of the suite this literature reports. They are scored the same
    # way as the multiple-choice tasks -- compare the model's logit for each admissible answer at the
    # answer position -- except the admissible answers are the words rather than option letters.
    "pope": dict(repo="lmms-lab/POPE", split="test", kind="yesno"),
    "mme": dict(repo="lmms-lab/MME", split="test", kind="yesno"),
}
YESNO = ("Yes", "No")


def _norm_image(v):
    if isinstance(v, list):
        v = v[0] if v else None
    return v.convert("RGB") if v is not None else None


def _item(task: str, r: Dict) -> Optional[Tuple[object, str, List[str], int]]:
    """(image, question, options, gold_index) or None when the row is unusable."""
    kind = TASKS[task]["kind"]
    img = _norm_image(r.get("image"))
    if img is None:
        return None
    q = str(r.get("question", "")).strip()

    if kind == "abcd":
        opts = [str(r.get(L, "")).strip() for L in "ABCD"]
        opts = [o for o in opts if o and o.lower() != "nan"]
        hint = str(r.get("hint", "")).strip()
        if hint and hint.lower() != "nan":
            q = f"{hint}\n{q}"
        gold = LETTERS.find(str(r["answer"]).strip().upper()[:1])
    elif kind == "choice_x":
        opts = [str(r.get(f"choice_{c}", "")).strip() for c in "abcd"]
        opts = [o for o in opts if o and o.lower() != "nan"]
        gold = LETTERS.find(str(r["answer"]).strip().upper()[:1])
    elif kind == "choices":
        opts = [str(x).strip() for x in r["choices"]]
        hint = str(r.get("hint", "")).strip()
        if hint and hint.lower() != "nan":
            q = f"{hint}\n{q}"
        gold = int(r["answer"])
    elif kind == "options":
        opts = [str(x).strip() for x in r["options"]]
        gold = int(r["answer"])
    elif kind == "yesno":
        opts = list(YESNO)
        ans = str(r["answer"]).strip().lower()
        if ans.startswith("yes"):
            gold = 0
        elif ans.startswith("no"):
            gold = 1
        else:
            return None
    else:                                    # options already inside the question text
        opts = None
        gold = LETTERS.find(str(r["answer"]).strip().upper()[:1])

    if gold is None or gold < 0:
        return None
    if opts is not None and (len(opts) < 2 or gold >= len(opts)):
        return None
    return img, q, opts, gold


def load_task(task: str, limit: int, seed: int = 0) -> List[Tuple[object, str, Optional[List[str]], int]]:
    from datasets import load_dataset
    spec = TASKS[task]
    kw = dict(streaming=True, split=spec["split"])
    ds = load_dataset(spec["repo"], spec.get("cfg"), **kw) if spec.get("cfg") \
        else load_dataset(spec["repo"], **kw)
    # A large shuffle buffer is a trap on the streamed multi-gigabyte sets: the iterator has to
    # decode buffer_size rows, images included, before it yields anything. 512 is enough to break
    # up the file ordering and returns the first item in seconds rather than minutes.
    ds = ds.shuffle(seed=seed, buffer_size=512)
    out = []
    for r in ds:
        it = _item(task, r)
        if it is None:
            continue
        # skip rows whose image is degenerate
        if min(it[0].size) < 8:
            continue
        out.append(it)
        if len(out) >= limit:
            break
    return out


def _prompt(bundle, question: str, opts: Optional[Sequence[str]]) -> str:
    if opts is not None and tuple(opts) == YESNO:
        instr = f"{question}\nAnswer the question using a single word, Yes or No."
    else:
        body = question if opts is None else question + "\n" + "\n".join(
            f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        instr = f"{body}\nAnswer with the option's letter from the given choices directly."
    if bundle.family == "llava":
        return f"USER: <image>\n{instr} ASSISTANT:"
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
    return bundle.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _first_ids(bundle, words) -> List[List[int]]:
    """First-token ids for each candidate answer, in both the bare and space-prefixed forms.

    Scoring the first token is what makes this one forward pass per item. It is unambiguous here
    because the candidates differ in their first token: option letters do, and so do Yes/No.
    """
    tok = bundle.processor.tokenizer
    out = []
    for w in words:
        ids = []
        for form in (w, " " + w, w.lower(), " " + w.lower()):
            t = tok(form, add_special_tokens=False)["input_ids"]
            if t:
                ids.append(t[0])
        out.append(sorted(set(ids)))
    return out


def _letter_ids(bundle, n: int) -> List[List[int]]:
    return _first_ids(bundle, [LETTERS[i] for i in range(n)])


@torch.no_grad()
def run_task(bundle, items, batch_size: int = 4) -> float:
    """Accuracy under letter scoring."""
    proc = bundle.processor
    correct = tot = 0
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        texts = [_prompt(bundle, q, o) for _, q, o, _ in chunk]
        imgs = [im for im, _, _, _ in chunk]
        enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True, **getattr(bundle, "proc_kwargs", {})).to(bundle.device)
        logits = bundle.model(**enc, use_cache=False).logits[:, -1, :].float()
        for b, (_, _, opts, gold) in enumerate(chunk):
            if opts is not None and tuple(opts) == YESNO:
                cand_ids = _first_ids(bundle, YESNO)
            else:
                cand_ids = _letter_ids(bundle, len(opts) if opts is not None else 4)
            best, best_lp = -1, -1e30
            for i, cands in enumerate(cand_ids):
                lp = max(float(logits[b, c]) for c in cands)
                if lp > best_lp:
                    best, best_lp = i, lp
            correct += int(best == gold)
            tot += 1
    return correct / max(1, tot)


@torch.no_grad()
def caption_ppl(bundle, batches) -> float:
    """Perplexity of held-out captions given their images."""
    nll = 0.0
    n = 0
    for enc in batches:
        sm = enc["_score_mask"]
        fwd = {k: v for k, v in enc.items() if not k.startswith("_")}
        logits = bundle.model(**fwd, use_cache=False).logits
        tgt = enc["input_ids"][:, 1:]
        m = sm[:, :-1]
        # Only the scored positions matter, and they are a few dozen out of a sequence that is mostly
        # image. Gathering them before the softmax turns a full-sequence log_softmax over a 152k
        # vocabulary into one over a handful of rows.
        sel = logits[:, :-1, :][m].float()
        lp = torch.log_softmax(sel, dim=-1)
        nll += float(-torch.gather(lp, -1, tgt[m].unsqueeze(-1)).sum())
        n += int(m.sum())
    import math
    return math.exp(nll / max(1, n))
