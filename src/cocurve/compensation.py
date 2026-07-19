"""FLAP-style mean/bias compensation for structured pruning (optional, clean).

When a structured unit is removed we normally zero its contribution. Compensation
instead freezes the removed channels at their CALIBRATION MEAN, which folds a
constant bias  W @ mean_removed  into the consuming projection — preserving the
removed channels' expected output and dropping only their fluctuation (the FLAP
insight). This is closed-form, label-free and needs no fine-tuning: a single
calibration forward pass to estimate the per-channel means.

collect_comp_means() returns the means in the dict shape register_runtime_masks()
consumes:  {"ffn": {layer_idx: Tensor[intermediate]},
            "attn": {layer_idx: Tensor[num_heads, head_dim]}}
"""
from __future__ import annotations

from typing import Dict, List

import torch

from .model import (ModelBundle, attention_module, attention_projections,
                    bundle_device, layer_modules, mlp_module, mlp_projections)


def collect_comp_means(bundle: ModelBundle, calib_ids: List[torch.Tensor],
                       max_seq_len: int = 2048) -> Dict[str, Dict[int, torch.Tensor]]:
    layers = layer_modules(bundle)
    device = bundle_device(bundle)
    ffn_sum: Dict[int, torch.Tensor] = {}
    ffn_cnt: Dict[int, int] = {}
    attn_sum: Dict[int, torch.Tensor] = {}
    attn_cnt: Dict[int, int] = {}
    hooks = []

    for layer_idx, layer in enumerate(layers):
        mlp = mlp_module(layer)
        if mlp is not None:
            projections = mlp_projections(mlp)
            down_proj = projections.get("down_proj") or projections.get("w2") or projections.get("c_proj")
            if down_proj is not None:
                def make_ffn_hook(li: int):
                    def hook(_m, inputs):
                        h = inputs[0]
                        if h.dim() != 3:
                            return
                        flat = h.reshape(-1, h.shape[-1]).float()
                        s = flat.sum(0).cpu()
                        ffn_sum[li] = s if li not in ffn_sum else ffn_sum[li] + s
                        ffn_cnt[li] = ffn_cnt.get(li, 0) + flat.shape[0]
                    return hook
                hooks.append(down_proj.register_forward_pre_hook(make_ffn_hook(layer_idx)))

        attn = attention_module(layer)
        projs = attention_projections(attn) if attn is not None else {}
        o_proj = projs.get("o_proj")
        if o_proj is not None:
            def make_attn_hook(li: int):
                def hook(_m, inputs):
                    h = inputs[0]
                    if h.dim() != 3:
                        return
                    flat = h.reshape(-1, h.shape[-1]).float()
                    s = flat.sum(0).cpu()
                    attn_sum[li] = s if li not in attn_sum else attn_sum[li] + s
                    attn_cnt[li] = attn_cnt.get(li, 0) + flat.shape[0]
                return hook
            hooks.append(o_proj.register_forward_pre_hook(make_attn_hook(layer_idx)))

    try:
        with torch.no_grad():
            for ids in calib_ids:
                bundle.model(input_ids=ids[:, :max_seq_len].to(device), use_cache=False)
    finally:
        for hk in hooks:
            hk.remove()

    ffn_mean = {li: (ffn_sum[li] / max(1, ffn_cnt[li])) for li in ffn_sum}
    attn_mean = {}
    for li in attn_sum:
        v = attn_sum[li] / max(1, attn_cnt[li])
        attn_mean[li] = v.view(bundle.num_heads, bundle.head_dim)
    return {"ffn": ffn_mean, "attn": attn_mean}
