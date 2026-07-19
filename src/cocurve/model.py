from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import to_torch_dtype


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str
    num_layers: int
    num_heads: int
    hidden_size: int
    head_dim: int
    intermediate_size: int
    kv_heads: int
    attn_pattern: str
    family: str
    model_path: str


def _layer_stack(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture for layer extraction")


def load_model_bundle(model_cfg: Dict[str, object], tokenizer_cfg: Dict[str, object]) -> ModelBundle:
    path = str(model_cfg["path"])
    dtype = to_torch_dtype(str(model_cfg.get("torch_dtype", "bfloat16")))
    device_map = model_cfg.get("device_map", "auto")
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "device_map": device_map,
        "attn_implementation": "eager",
    }
    if model_cfg.get("load_in_8bit", False):
        model_kwargs["load_in_8bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=bool(tokenizer_cfg.get("trust_remote_code", True)),
        use_fast=bool(tokenizer_cfg.get("use_fast", True)),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, **model_kwargs)
    model.eval()
    layers = _layer_stack(model)
    config = model.config
    num_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 0)))
    hidden_size = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    # Some configs (e.g. Mistral-7B-v0.3) define head_dim but set it to null;
    # getattr then returns None, not the fallback, so guard explicitly.
    _head_dim = getattr(config, "head_dim", None)
    head_dim = int(_head_dim) if _head_dim else hidden_size // max(1, num_heads)
    intermediate_size = int(getattr(config, "intermediate_size", hidden_size * 4))
    kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    if kv_heads == num_heads:
        attn_pattern = "mha"
    elif kv_heads == 1:
        attn_pattern = "mqa"
    else:
        attn_pattern = "gqa"

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_layers=len(layers),
        num_heads=num_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        kv_heads=kv_heads,
        attn_pattern=attn_pattern,
        family=str(model_cfg.get("family", "unknown")),
        model_path=path,
    )


def forward_logits(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return outputs.logits


def top_r_distribution(logits: torch.Tensor, top_r: int) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=min(top_r, probs.shape[-1]), dim=-1)
    values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return indices, values


def kl_from_top_r(
    teacher_indices: torch.Tensor,
    teacher_probs: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    # The student distribution must be normalized over the full vocabulary.
    # Softmaxing only the teacher top-r logits hides probability mass that moved
    # outside the retained teacher support and makes pruned models look too good.
    student_log_probs_full = torch.log_softmax(student_logits, dim=-1)
    student_log_probs = torch.gather(student_log_probs_full, dim=-1, index=teacher_indices)
    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-12))
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return kl


def layer_modules(bundle: ModelBundle):
    return _layer_stack(bundle.model)


def bundle_device(bundle: ModelBundle) -> torch.device:
    if hasattr(bundle.model, "hf_device_map") and getattr(bundle.model, "hf_device_map"):
        try:
            embed = bundle.model.get_input_embeddings()
            return next(embed.parameters()).device
        except Exception:
            pass
    return next(bundle.model.parameters()).device


def attention_module(layer):
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attn"):
        return layer.attn
    return None


def attention_projections(attn) -> Dict[str, torch.nn.Module]:
    projs = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if hasattr(attn, name):
            projs[name] = getattr(attn, name)
    return projs


def mlp_module(layer):
    if hasattr(layer, "mlp"):
        return layer.mlp
    if hasattr(layer, "feed_forward"):
        return layer.feed_forward
    return None


def mlp_projections(mlp) -> Dict[str, torch.nn.Module]:
    projs = {}
    for name in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "c_fc", "c_proj"):
        if hasattr(mlp, name):
            projs[name] = getattr(mlp, name)
    return projs


def supports_layer_replay(bundle: ModelBundle) -> bool:
    return hasattr(bundle.model, "model") and hasattr(bundle.model.model, "layers")


def build_causal_attention_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Build additive attention mask with shape [B, 1, T, T].
    bsz, seq_len = attention_mask.shape
    device = attention_mask.device
    min_val = torch.finfo(dtype).min
    causal = torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)
    padding = (1 - attention_mask[:, None, None, :]).to(dtype=dtype) * min_val
    return causal + padding


def position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.long().cumsum(dim=-1) - 1


def capture_layer_inputs(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    layers = layer_modules(bundle)
    captured: Dict[int, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    for idx, layer in enumerate(layers):
        def _make_hook(layer_idx: int):
            def _hook(_module, inputs):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    captured[layer_idx] = inputs[0].detach()
            return _hook
        hooks.append(layer.register_forward_pre_hook(_make_hook(idx)))

    _ = forward_logits(bundle, input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()
    return captured


def capture_single_layer_input(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    layers = layer_modules(bundle)
    captured: List[torch.Tensor] = []

    def _hook(_module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured.append(inputs[0].detach())

    handle = layers[layer_idx].register_forward_pre_hook(_hook)
    _ = forward_logits(bundle, input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()
    if not captured:
        raise RuntimeError(f"Failed to capture layer input for layer {layer_idx}")
    return captured[0]


def forward_from_layer(
    bundle: ModelBundle,
    start_layer_idx: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if not supports_layer_replay(bundle):
        raise ValueError("Layer replay is not supported for this model family.")
    model_core = bundle.model.model
    layers = layer_modules(bundle)
    device = hidden_states.device
    seq_len = hidden_states.shape[1]
    position_ids = position_ids_from_attention_mask(attention_mask).to(device)
    cache_position = torch.arange(seq_len, device=device)

    try:
        from transformers.masking_utils import create_causal_mask

        causal_mask = create_causal_mask(
            config=bundle.model.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask.to(device),
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
    except Exception:
        causal_mask = build_causal_attention_mask(attention_mask.to(device), hidden_states.dtype)

    position_embeddings = None
    rotary_emb = getattr(model_core, "rotary_emb", None)
    if rotary_emb is not None:
        position_embeddings = rotary_emb(hidden_states, position_ids)

    hs = hidden_states
    for layer in layers[start_layer_idx:]:
        layer_kwargs = {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "cache_position": cache_position,
        }
        if position_embeddings is not None:
            layer_kwargs["position_embeddings"] = position_embeddings
        try:
            out = layer(hs, **layer_kwargs)
        except TypeError:
            out = layer(hs, attention_mask=causal_mask, position_ids=position_ids)
        hs = out[0] if isinstance(out, tuple) else out

    if hasattr(model_core, "norm"):
        hs = model_core.norm(hs)
    logits = bundle.model.lm_head(hs)
    return logits
