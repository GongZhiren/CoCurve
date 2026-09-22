"""Runtime masking of VLM units, over both towers.

Semantics are the ones the text-only pipeline uses, and they matter: a unit is removed by zeroing
its slice of the INPUT to the projection that writes back into the residual stream -- the head's
slice of the attention output projection, the group's channels of the second FFN matrix. That
deletes exactly the unit's additive contribution and nothing else, which is what makes the finite
ablation a stand-in for the derivative column.

The one thing that does not carry over is tensor rank. The language tower sees
``(batch, seq, width)``; Qwen's vision tower packs all patches of a batch into ``(tokens, width)``
and runs rank-2 throughout. A hook written for rank 3 does not fail on it -- it silently passes the
tensor through, so every vision unit looks free of consequence. These hooks key on the trailing
dimension instead and are rank-agnostic.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Set

import torch

from .bundle import VLMBundle, attn_out_proj, ffn_out_proj
from .units import VLMRegistry, VLMUnit


def _slice_mask(width: int, groups: Iterable[tuple], device, dtype) -> torch.Tensor:
    m = torch.ones(width, dtype=torch.float32)
    for start, size in groups:
        m[start:start + size] = 0.0
    return m.to(device=device, dtype=dtype)


# A recovery stage that distils from the dense model needs both models in one process, and here the
# dense model IS this model with the hooks inert -- the pruning is a mask, not a deletion. This flag
# is what lets one forward pass be the teacher and the next be the student without loading a second
# copy of the weights.
_MASKS_ON = True


@contextmanager
def masks_off():
    """Run the enclosed forward passes as the unpruned model."""
    global _MASKS_ON
    prev = _MASKS_ON
    _MASKS_ON = False
    try:
        yield
    finally:
        _MASKS_ON = prev


def _make_hook(mask: torch.Tensor):
    w = int(mask.shape[0])

    def hook(_mod, inputs):
        if not _MASKS_ON or not inputs:
            return inputs
        h = inputs[0]
        if not torch.is_tensor(h) or h.shape[-1] != w:
            return inputs
        m = mask.to(device=h.device, dtype=h.dtype)
        return (h * m,) + tuple(inputs[1:])

    return hook


@contextmanager
def masked(bundle: VLMBundle, reg: VLMRegistry, removed: Set[int]):
    """Temporarily remove ``removed`` (a set of unit_ids) from the forward pass."""
    handles = []
    if removed:
        # group the removed units by (tower, layer, kind) so each projection gets one hook
        buckets: Dict[tuple, List[VLMUnit]] = {}
        for u in reg.units:
            if u.unit_id in removed:
                buckets.setdefault((u.tower, u.layer_idx, u.unit_type), []).append(u)
        for (tower, L, kind), us in buckets.items():
            t = bundle.towers[tower]
            if kind == "attn_head":
                proj = attn_out_proj(bundle, tower, L)
                groups = [(h * t.head_dim, t.head_dim)
                          for u in us for h in u.meta["query_head_indices"]]
                width = t.attn_slice_width
            else:
                proj = ffn_out_proj(bundle, tower, L)
                groups = [(u.meta["start_channel"], u.meta["group_size"]) for u in us]
                width = t.intermediate_size
            mask = _slice_mask(width, groups, "cpu", torch.float32)
            handles.append(proj.register_forward_pre_hook(_make_hook(mask)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def hook_coverage(bundle: VLMBundle, reg: VLMRegistry) -> Dict[str, int]:
    """How many units the masking layer can actually reach. A unit whose projection cannot be
    found, or whose slice width disagrees with the tensor it sees, would be silently inert; this
    is the check that says so before an experiment is run on it."""
    ok = {"lm.attn_head": 0, "lm.ffn_group": 0, "vis.attn_head": 0, "vis.ffn_group": 0}
    for u in reg.units:
        t = bundle.towers[u.tower]
        try:
            proj = attn_out_proj(bundle, u.tower, u.layer_idx) if u.unit_type == "attn_head" \
                else ffn_out_proj(bundle, u.tower, u.layer_idx)
        except Exception:
            continue
        want = t.attn_slice_width if u.unit_type == "attn_head" else t.intermediate_size
        if int(proj.weight.shape[1]) == want:
            ok[f"{u.tower}.{u.unit_type}"] += 1
    return ok
