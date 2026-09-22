"""Model access for vision-language models.

Deliberately separate from ``cocurve.model``: the LLM pipeline that produced every number in
the paper is left untouched, and this module carries the two-tower specifics instead. A VLM gives
the method something the text-only experiments cannot -- a second *module family* on the far side
of a projector -- so the object of interest here is the same matrix with a wider support: attention
and FFN units of the language tower and of the vision tower, under one shared budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TowerSpec:
    """Where one tower's blocks live and how a head maps onto a projection input."""
    name: str                 # "lm" | "vis"
    blocks: Any               # nn.ModuleList
    num_layers: int
    hidden_size: int
    num_heads: int
    kv_heads: int
    head_dim: int
    intermediate_size: int
    attn_out_name: str        # projection whose INPUT carries per-head slices
    ffn_out_name: str         # projection whose INPUT carries FFN channels
    attn_path: str            # attribute path from block to the attention module
    ffn_path: str             # attribute path from block to the mlp module
    attn_slice_width: int     # width of attn_out input = num_heads * head_dim


@dataclass
class VLMBundle:
    model: Any
    processor: Any
    model_id: str
    family: str
    towers: Dict[str, TowerSpec] = field(default_factory=dict)
    # Extra keyword arguments every processor call must carry, set once per family. They exist
    # because a processor's __call__ can override what the model's own image-processor config
    # declares, and one of those overrides costs an order of magnitude in visual tokens.
    proc_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def _get(mod, path: str):
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


def _qwen_vl(model, model_id: str, family: str) -> VLMBundle:
    inner = model.model
    vis, lm = inner.visual, inner.language_model
    tc = model.config.get_text_config()
    vc = model.config.vision_config

    n_vh = int(getattr(vc, "num_heads"))
    blk0 = vis.blocks[0]
    # Read the vision width off the weights, not the config. Qwen2-VL's vision_config carries
    # hidden_size=1536, which is the width AFTER the patch merger (chosen to match the language
    # tower); the ViT itself runs at embed_dim=1280. Taking hidden_size here silently gave every
    # vision head a 20% inflated cost and made the head slices the wrong width.
    # The vision MLP is named differently in every Qwen-VL release: fc1/fc2 in Qwen2-VL,
    # gate_proj/up_proj/down_proj in Qwen2.5-VL, linear_fc1/linear_fc2 in Qwen3-VL. Probe rather than
    # branch, so the next release does not need a new case.
    ffn_out = next(n for n in ("down_proj", "linear_fc2", "fc2", "w2")
                   if hasattr(blk0.mlp, n))
    v_hidden = int(getattr(blk0.attn, "proj").weight.shape[0])
    v_inter = int(getattr(blk0.mlp, ffn_out).weight.shape[1])

    n_lh, n_lkv = int(tc.num_attention_heads), int(tc.num_key_value_heads)
    l_hidden = int(tc.hidden_size)
    l_head_dim = int(getattr(tc, "head_dim", 0) or l_hidden // n_lh)

    towers = {
        "lm": TowerSpec("lm", lm.layers, len(lm.layers), l_hidden, n_lh, n_lkv, l_head_dim,
                        int(tc.intermediate_size), "o_proj", "down_proj", "self_attn", "mlp",
                        n_lh * l_head_dim),
        "vis": TowerSpec("vis", vis.blocks, len(vis.blocks), v_hidden, n_vh, n_vh,
                         v_hidden // n_vh, int(v_inter), "proj", ffn_out, "attn", "mlp",
                         v_hidden),
    }
    return VLMBundle(model=model, processor=None, model_id=model_id, family=family, towers=towers)


def _clip_style(model, model_id: str, family: str) -> VLMBundle:
    """LLaVA-1.5, LLaVA-NeXT and InternVL: a separate vision encoder feeding a decoder.

    They differ in where the encoder lives and in what its projections are called, so the paths are
    resolved by probing rather than hard-coded per family -- the alternative is a branch per release.
    """
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.language_model
    vt = model.model.vision_tower if hasattr(model.model, "vision_tower") else model.vision_tower
    # the encoder block list, wherever it is
    layers = None
    # "encoder.layer" (singular) is InternVL's; the others are CLIP-, SigLIP- and timm-style.
    for path in ("vision_model.encoder.layers", "encoder.layers", "encoder.layer",
                 "layers", "blocks"):
        try:
            layers = _get(vt, path)
            break
        except AttributeError:
            continue
    if layers is None:
        raise ValueError(f"cannot find the vision blocks of {type(vt).__name__}")

    blk0 = layers[0]
    attn_path = next(n for n in ("self_attn", "attn", "attention") if hasattr(blk0, n))
    ffn_path = next(n for n in ("mlp", "feed_forward") if hasattr(blk0, n))
    attn_mod, ffn_mod = _get(blk0, attn_path), _get(blk0, ffn_path)
    attn_out = next(n for n in ("out_proj", "proj", "o_proj", "projection_layer")
                    if hasattr(attn_mod, n))
    ffn_out = next(n for n in ("fc2", "down_proj", "w2") if hasattr(ffn_mod, n))
    v_hidden = int(getattr(attn_mod, attn_out).weight.shape[0])
    v_inter = int(getattr(ffn_mod, ffn_out).weight.shape[1])
    vc = model.config.vision_config
    n_vh = int(getattr(vc, "num_attention_heads", 0) or getattr(vc, "num_heads"))

    tc = model.config.get_text_config()
    n_lh = int(tc.num_attention_heads)
    n_lkv = int(getattr(tc, "num_key_value_heads", n_lh) or n_lh)
    l_hidden = int(tc.hidden_size)
    l_head_dim = int(getattr(tc, "head_dim", 0) or l_hidden // n_lh)
    towers = {
        "lm": TowerSpec("lm", lm.layers, len(lm.layers), l_hidden, n_lh, n_lkv, l_head_dim,
                        int(tc.intermediate_size), "o_proj", "down_proj", "self_attn", "mlp",
                        n_lh * l_head_dim),
        "vis": TowerSpec("vis", layers, len(layers), v_hidden, n_vh, n_vh, v_hidden // n_vh,
                         v_inter, attn_out, ffn_out, attn_path, ffn_path, v_hidden),
    }
    return VLMBundle(model=model, processor=None, model_id=model_id, family=family, towers=towers)


def _llava(model, model_id: str) -> VLMBundle:
    vt = model.model.vision_tower if hasattr(model.model, "vision_tower") else model.vision_tower
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.language_model
    layers = vt.vision_model.encoder.layers
    vc = model.config.vision_config
    tc = model.config.get_text_config()
    n_vh = int(vc.num_attention_heads)
    v_hidden = int(vc.hidden_size)
    n_lh = int(tc.num_attention_heads)
    n_lkv = int(getattr(tc, "num_key_value_heads", n_lh))
    l_hidden = int(tc.hidden_size)
    l_head_dim = int(getattr(tc, "head_dim", 0) or l_hidden // n_lh)
    towers = {
        "lm": TowerSpec("lm", lm.layers, len(lm.layers), l_hidden, n_lh, n_lkv, l_head_dim,
                        int(tc.intermediate_size), "o_proj", "down_proj", "self_attn", "mlp",
                        n_lh * l_head_dim),
        "vis": TowerSpec("vis", layers, len(layers), v_hidden, n_vh, n_vh, v_hidden // n_vh,
                         int(vc.intermediate_size), "out_proj", "fc2", "self_attn", "mlp",
                         v_hidden),
    }
    return VLMBundle(model=model, processor=None, model_id=model_id, family="llava", towers=towers)


def load_vlm(model_id: str, device: str = "cuda:0", dtype=torch.bfloat16) -> VLMBundle:
    from transformers import AutoConfig, AutoProcessor
    cfg = AutoConfig.from_pretrained(model_id)
    arch = type(cfg).__name__
    dev = {"": int(device.split(":")[-1])} if device.startswith("cuda:") else device

    if arch.startswith("Qwen2VL"):
        from transformers import Qwen2VLForConditionalGeneration as K
        fam = "qwen2-vl"
    elif arch.startswith("Qwen2_5_VL"):
        from transformers import Qwen2_5_VLForConditionalGeneration as K
        fam = "qwen2.5-vl"
    elif arch.startswith("Qwen3VL"):
        from transformers import Qwen3VLForConditionalGeneration as K
        fam = "qwen3-vl"
    elif arch.startswith("LlavaNext"):
        from transformers import LlavaNextForConditionalGeneration as K
        fam = "llava-next"
    elif arch.startswith("Llava"):
        from transformers import LlavaForConditionalGeneration as K
        fam = "llava"
    elif arch.startswith("InternVL"):
        from transformers import InternVLForConditionalGeneration as K
        fam = "internvl"
    else:
        raise ValueError(f"unsupported VLM config {arch}")

    model = K.from_pretrained(model_id, dtype=dtype, device_map=dev)
    model.eval()
    if fam.startswith("qwen"):
        b = _qwen_vl(model, model_id, fam)
    else:
        b = _clip_style(model, model_id, fam)
    b.processor = AutoProcessor.from_pretrained(model_id)
    if fam == "internvl":
        # InternVL3's image-processor config declares crop_to_patches=False, but the processor's
        # __call__ turns tiling on by default and yields up to thirteen 448-square tiles per image:
        # measured, 3450 tokens per calibration sequence against 313 for Qwen3-VL and 721 for
        # LLaVA-1.5, which put the single-unit ablation pass at an eta of 78 hours. Passing the
        # model's own declared setting restores one tile, and it is applied to calibration, to the
        # curvature pass and to evaluation alike so every number on this model sees the same input.
        b.proc_kwargs = {"crop_to_patches": False}
    return b


def attn_out_proj(bundle: VLMBundle, tower: str, layer_idx: int):
    t = bundle.towers[tower]
    return getattr(_get(t.blocks[layer_idx], t.attn_path), t.attn_out_name)


def ffn_out_proj(bundle: VLMBundle, tower: str, layer_idx: int):
    t = bundle.towers[tower]
    return getattr(_get(t.blocks[layer_idx], t.ffn_path), t.ffn_out_name)
