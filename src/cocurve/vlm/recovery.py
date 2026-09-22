"""The recovery corpus for a pruned vision--language model.

The text-only half of this paper recovers on cleaned Alpaca because that is the corpus LLM-Pruner
introduced and SlimGPT and SlimLLM adopted; recovering on something else would be a gratuitous
difference from the methods it compares against. The vision--language analogue of that corpus is
LLaVA's visual instruction data, which is what this literature tunes on, so that is what is used
here -- unchanged, and identical for every arm.

Its images come from COCO, which is also where POPE's images come from. That is worth stating and it
is not leakage introduced by us: LLaVA-1.5 and LLaVA-NeXT were themselves instruction-tuned on this
data before we pruned them, so the stage restores supervision the dense model already had rather than
adding supervision it never saw. More to the point for a comparison between selectors, every arm gets
the identical corpus, so nothing about the ordering depends on it.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

# LLaVA-NeXT's own supervised mixture, first, because the corpus has to speak the language the
# evaluation scores. Measured: recovering on a conversation-only set (llava-instruct-mix-vsft) drove
# the seven-benchmark mean from 56.29 down to 46.71 on Qwen3-VL-8B at rho=30% while *improving*
# caption perplexity from 25.4 to 17.7 -- the stage worked on its own objective and moved the model
# away from short discriminative answers, which is what these benchmarks ask for. The mixture below
# contains the academic-VQA portion with its "answer the question with a single word or phrase"
# instructions, which is what LLaVA-1.5 and LLaVA-NeXT were tuned on and what this literature
# recovers a pruned vision-language model with.
CANDIDATES = ["lmms-lab/LLaVA-NeXT-Data",
              "HuggingFaceH4/llava-instruct-mix-vsft",
              "lmms-lab/LLaVA-Instruct-150K"]


def _first_text(msgs) -> Tuple[str, str]:
    """The (user, assistant) pair of one conversation, flattened to plain strings."""
    user, asst = "", ""
    for m in msgs if isinstance(msgs, list) else []:
        role = m.get("role") or m.get("from") or ""
        c = m.get("content") or m.get("value") or ""
        if isinstance(c, list):
            # a content list interleaves {"type":"image"} blocks whose "text" is None with the
            # real text; str(None) would put the word "None" into the instruction
            c = " ".join(str(x["text"]).strip() for x in c
                         if isinstance(x, dict) and x.get("text"))
        c = str(c).replace("<image>", "").strip()
        if not c:
            continue
        if role in ("user", "human") and not user:
            user = c
        elif role in ("assistant", "gpt") and not asst:
            asst = c
        if user and asst:
            break
    return user, asst


SHORT_REPO = "lmms-lab/textvqa"
# The three suffixes are copied from evalvlm._prompt, which is what makes the recovery corpus
# format-matched to the scoring: same instruction, same answer shape, different data.
WORD_SUFFIX = "\nAnswer the question using a single word or phrase."
LETTER_SUFFIX = "\nAnswer with the option's letter from the given choices directly."
YESNO_SUFFIX = "\nAnswer the question using a single word, Yes or No."


def load_short_answer(n: int, seed: int = 0) -> List[Tuple[object, str, str]]:
    """n triples in the three instruction formats the benchmarks actually use.

    This is the detail that decided whether a recovery stage helps or hurts here. Our benchmarks are
    scored by reading the first answer token under an explicit format instruction -- "Answer with the
    option's letter from the given choices directly.", "Answer the question using a single word, Yes
    or No." -- and the corpus this stage used carried neither the instructions nor the answer shapes:
    measured, zero of eight hundred usable LLaVA-NeXT rows have an answer shorter than eight
    characters. Fine-tuning on it improves long-form language modelling and unlearns the format, and
    that is exactly what the numbers showed (POPE down 17 to 27 points on every arm, under both
    cross-entropy and distillation, with caption perplexity improving). LLaVA's own recovery mixture
    does not have this problem because it is part short-answer VQA and part multiple choice, with
    those instruction suffixes attached. This function reconstructs that part explicitly: one third
    short word, one third option letter, one third yes/no, all grounded in a VQA training split whose
    images (TextVQA, from OpenImages) are disjoint from every COCO-derived evaluation suite here.
    """
    from datasets import load_dataset
    from collections import Counter
    import random
    rng = random.Random(seed)
    ds = load_dataset(SHORT_REPO, split="train", streaming=True).shuffle(seed=seed, buffer_size=512)
    raw = []
    for rec in ds:
        img = rec.get("image")
        q = (rec.get("question") or "").strip()
        answers = [a for a in (rec.get("answers") or []) if a and a.strip()]
        if img is None or not hasattr(img, "convert") or not q or not answers:
            continue
        raw.append((img, q, Counter(x.strip().lower() for x in answers).most_common(1)[0][0]))
        if len(raw) >= n + 64:            # a small surplus to draw distractors from
            break
    pool = [a for _, _, a in raw]
    out = []
    for i, (img, q, ans) in enumerate(raw[:n]):
        style = i % 3
        if style == 0:
            u, a = q + WORD_SUFFIX, ans
        elif style == 1:
            others = [x for x in rng.sample(pool, min(12, len(pool))) if x != ans][:3]
            if len(others) < 3:
                u, a = q + WORD_SUFFIX, ans
            else:
                opts = others + [ans]
                rng.shuffle(opts)
                body = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", opts))
                u = f"{q}\n{body}{LETTER_SUFFIX}"
                a = "ABCD"[opts.index(ans)]
        else:
            truthy = rng.random() < 0.5
            shown = ans if truthy else next(
                (x for x in rng.sample(pool, min(12, len(pool))) if x != ans), ans)
            u = f"Is the answer to the question \"{q}\" \"{shown}\"?{YESNO_SUFFIX}"
            a = "Yes" if shown == ans else "No"
        out.append((img.convert("RGB"), u, a))
    print(f"short-answer corpus: {SHORT_REPO}, {len(out)} samples "
          f"(word / option letter / yes-no in equal parts)", flush=True)
    return out


def load_mixed(n: int, seed: int = 0, repo: str = None,
               short_frac: float = 0.5) -> List[Tuple[object, str, str]]:
    """A recovery corpus with both answer styles, interleaved deterministically."""
    import random
    n_short = int(round(n * short_frac))
    long_part = load_instructions(n - n_short, seed=seed, repo=repo) if n - n_short > 0 else []
    short_part = load_short_answer(n_short, seed=seed) if n_short > 0 else []
    mixed = long_part + short_part
    random.Random(seed).shuffle(mixed)
    print(f"recovery corpus: {len(long_part)} long-form + {len(short_part)} short-answer "
          f"= {len(mixed)} samples", flush=True)
    return mixed


def load_instructions(n: int, seed: int = 0, repo: str = None) -> List[Tuple[object, str, str]]:
    """n (image, instruction, response) triples, streamed."""
    from datasets import load_dataset
    last = None
    for r in ([repo] if repo else CANDIDATES):
        try:
            ds = load_dataset(r, split="train", streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=512)
            out = []
            for rec in ds:
                img = rec.get("image") or rec.get("images")
                if isinstance(img, list):
                    img = img[0] if img else None
                if img is None or not hasattr(img, "convert"):
                    continue
                u, a = _first_text(rec.get("messages") or rec.get("conversations"))
                # Only empty turns are dropped. An earlier version required eight characters on both
                # sides, which silently removed every short answer -- "Yes", "No", "A", "2" -- from a
                # mixture that is part VQA by construction, leaving a corpus of long conversational
                # turns only. A stage trained on that improves long-form language modelling and
                # unlearns the answer format the benchmarks score: measured, it cost POPE 17 to 27
                # points on every arm it touched while caption perplexity improved.
                if not u or not a:
                    continue
                out.append((img.convert("RGB"), u, a))
                if len(out) >= n:
                    break
            if out:
                print(f"recovery corpus: {r}, {len(out)} samples", flush=True)
                return out
            last = f"{r}: no usable rows"
        except Exception as e:
            last = f"{r}: {type(e).__name__}: {e}"
    raise RuntimeError(f"no recovery corpus available ({last})")


def build_train_batch(bundle, triples: List[Tuple[object, str, str]]) -> Dict[str, torch.Tensor]:
    """One padded batch whose loss mask covers the response tokens only.

    Scoring the instruction as well would train the model to reproduce text it was handed, which is
    not what a recovery stage is for, and it is the same convention the calibration pass uses.
    """
    proc = bundle.processor
    texts, images, ans_len = [], [], []
    for img, u, a in triples:
        msgs = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": u}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(prompt + a)
        images.append(img)
        ans_len.append(len(proc.tokenizer(a, add_special_tokens=False)["input_ids"]))
    enc = proc(text=texts, images=images, return_tensors="pt", padding=True, **getattr(bundle, "proc_kwargs", {}))
    ids, attn = enc["input_ids"], enc["attention_mask"]
    loss_mask = torch.zeros_like(ids, dtype=torch.bool)
    for b in range(ids.shape[0]):
        last = int(attn[b].nonzero()[-1]) if attn[b].any() else ids.shape[1] - 1
        loss_mask[b, max(0, last - ans_len[b]):last] = True
    enc = {k: (v.to(bundle.device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    enc["_loss_mask"] = loss_mask.to(bundle.device)
    return enc


def lora_targets(bundle, towers=("lm", "vis")) -> List[str]:
    """Linear module names inside the requested towers' attention and FFN blocks.

    Names are shared between towers on some families -- CLIP and Vicuna both call a projection
    q_proj -- so a name list alone cannot confine the adapter to one tower. It selects the candidate
    set; freeze_tower_lora() does the confining.
    """
    names = set()
    for tname, t in bundle.towers.items():
        if tname not in towers:
            continue
        blk = t.blocks[0]
        for path in (t.attn_path, t.ffn_path):
            mod = blk
            for part in path.split("."):
                mod = getattr(mod, part)
            for n, m in mod.named_modules():
                if isinstance(m, torch.nn.Linear) and n:
                    names.add(n.split(".")[-1])
    return sorted(names)


def freeze_tower_lora(peft_model, bundle, tower: str = "vis") -> int:
    """Freeze every adapter parameter inside one tower, and report how many.

    This is what the vision--language literature does and this repository was not doing. LLaVA's
    own LoRA path skips the vision tower and the projector when it collects target modules
    (multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']); the Qwen2-VL and
    InternVL recipes freeze the vision model and adapt the language model only, on the stated
    grounds that tuning the encoder does not help consistently and destabilises the visual
    representation. Adapting both towers on a few thousand caption samples is the one deviation this
    stage had from standard practice, and it is the deviation that matches the symptom: recovery that
    costs accuracy on a healthy model and buys nothing on a destroyed one.

    LoRA's B matrix is zero-initialised, so a frozen adapter is exactly the identity: freezing after
    wrapping is equivalent to never having placed the adapter there, and it does not depend on
    module-name uniqueness across towers.
    """
    root = bundle.towers[tower].blocks
    qualified = {id(m): n for n, m in bundle.model.named_modules()}
    prefix = qualified.get(id(root))
    if prefix is None:
        raise RuntimeError(f"could not locate the {tower} tower inside the model")
    n = 0
    for name, param in peft_model.named_parameters():
        if "lora_" in name and prefix in name:
            param.requires_grad = False
            n += 1
    return n

# The multimodal projector is the one part of a VLM that every standard LoRA recipe trains rather
# than freezes: LLaVA's own script gives it its own learning rate (--mm_projector_lr 2e-5) while the
# vision tower stays frozen, and the reason shows up in our own diagnostics -- at rho=40% the
# recovered model's caption perplexity falls from 179 to 32 while its benchmark accuracy does not
# move, which is language modelling repaired and visual grounding still broken. The projector is
# what carries that grounding, and a frozen projector cannot re-align to a pruned language tower.
PROJECTOR_KEYS = ("merger", "mm_projector", "multi_modal_projector", "mlp1", "resampler",
                  "vision_projection", "projector")


def projector_parameters(bundle):
    """Full-rank trainable parameters of the multimodal projector, and the module names they sit in.

    Located by name against PROJECTOR_KEYS and by exclusion: anything inside either tower's block
    list is a tower parameter, not a projector one, whatever it is called.
    """
    qualified = {id(m): n for n, m in bundle.model.named_modules()}
    tower_prefixes = tuple(
        p for p in (qualified.get(id(t.blocks)) for t in bundle.towers.values()) if p)
    hits, names = [], []
    for name, module in bundle.model.named_modules():
        if not any(k in name for k in PROJECTOR_KEYS):
            continue
        if any(name.startswith(tp) for tp in tower_prefixes):
            continue
        own = [p for n, p in module.named_parameters(recurse=False)]
        if own:
            hits.extend(own); names.append(name)
    return hits, names
