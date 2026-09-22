"""Image-conditioned calibration for VLM pruning risk.

The risk is the same object as in the text-only setting -- token-level KL between the frozen model
and its masked copy -- with the conditioning extended to an image. Positions are the text tokens of
an image-grounded sequence, so the quantity being preserved is the model's next-token distribution
*given what it sees*.

Images come from Flickr30k rather than COCO on purpose. Several of the evaluation suites
(SEED-Bench, A-OKVQA) are built on COCO images; Flickr is a disjoint source, so no calibration
image can appear in an evaluation.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import torch

CALIB_REPO = "lmms-lab/flickr30k"
CALIB_SPLIT = "test"
PROMPT = "Describe this image."


def load_pairs(n: int, seed: int = 0, skip: int = 0) -> List[Tuple[object, str]]:
    """n (image, caption) pairs, streamed so nothing large is materialised."""
    from datasets import load_dataset
    ds = load_dataset(CALIB_REPO, split=CALIB_SPLIT, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2048)
    out: List[Tuple[object, str]] = []
    for i, r in enumerate(ds):
        if i < skip:
            continue
        cap = r["caption"]
        if isinstance(cap, list):
            # Use all five references, not the longest one. The risk is an average over scored
            # positions, so positions are the sample size that decides how well H is estimated, and
            # a single caption gives only ~16 of them per image. Joining the five costs almost
            # nothing per forward pass -- the image tokens dominate the sequence -- and multiplies
            # the scored positions by about four.
            caps, seen, uniq = [str(c).strip() for c in cap if str(c).strip()], set(), []
            for c in caps:
                if c.lower() not in seen:
                    seen.add(c.lower())
                    uniq.append(c if c.endswith(".") else c + ".")
            cap = " ".join(uniq)
        cap = str(cap).strip()
        if len(cap) < 40:
            continue
        out.append((r["image"].convert("RGB"), cap))
        if len(out) >= n:
            break
    return out


def build_batch(bundle, pairs: List[Tuple[object, str]]) -> Dict[str, torch.Tensor]:
    """One padded batch, plus a boolean mask over the positions the risk is measured on.

    Only answer tokens are scored. The prompt and the image placeholders are context: including
    them would let a unit look important merely for reproducing text the model was handed.
    """
    proc = bundle.processor
    texts, images, ans_lens = [], [], []
    for img, cap in pairs:
        msgs = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(prompt + cap)
        images.append(img)
        ans_lens.append(len(proc.tokenizer(cap, add_special_tokens=False)["input_ids"]))

    enc = proc(text=texts, images=images, return_tensors="pt", padding=True, **getattr(bundle, "proc_kwargs", {}))
    ids = enc["input_ids"]
    attn = enc["attention_mask"]
    score = torch.zeros_like(ids, dtype=torch.bool)
    for b in range(ids.shape[0]):
        last = int(attn[b].nonzero()[-1]) if attn[b].any() else ids.shape[1] - 1
        a = ans_lens[b]
        # positions whose PREDICTION is an answer token: shifted by one
        lo = max(0, last - a)
        score[b, lo:last] = True
    enc = {k: (v.to(bundle.device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    enc["_score_mask"] = score.to(bundle.device)
    return enc


def teacher_cache(bundle, batches: List[Dict[str, torch.Tensor]], top_r: int = 64):
    """Dense-model top-r support, probabilities and logits at every scored position."""
    idx_all, prob_all, logit_all = [], [], []
    with torch.no_grad():
        for enc in batches:
            sm = enc["_score_mask"]
            fwd = {k: v for k, v in enc.items() if not k.startswith("_")}
            # Index the scored positions BEFORE widening precision: the full logit tensor is
            # (batch, seq, 152k) and casting it costs hundreds of megabytes and most of the runtime,
            # while only a few dozen rows per batch are ever used.
            sel = bundle.model(**fwd, use_cache=False).logits[sm].float()   # (P_b, V)
            probs = torch.softmax(sel, dim=-1)
            p, i = torch.topk(probs, k=min(top_r, probs.shape[-1]), dim=-1)
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
            idx_all.append(i.cpu())
            prob_all.append(p.cpu())
            logit_all.append(torch.gather(sel, -1, i).cpu())
    return (torch.cat(idx_all), torch.cat(prob_all), torch.cat(logit_all))


def fingerprint(pairs: List[Tuple[object, str]]) -> str:
    h = hashlib.sha1()
    for _, cap in pairs:
        h.update(cap.encode())
    return h.hexdigest()[:12]


def load_pairs_split(n_calib: int, n_holdout: int, seed: int = 0):
    """One streamed pass, split into two disjoint draws.

    The held-out draw has to be images the estimator never saw, and getting there by skipping
    forward costs a full decode of everything skipped. Taking both draws from one pass makes
    disjointness exact and the cost the same as loading the calibration set alone.
    """
    both = load_pairs(n_calib + n_holdout, seed=seed)
    return both[:n_calib], both[n_calib:]
