"""Selection over both towers under one shared budget.

The solver itself is the text-only one, unchanged -- cost-normalised greedy on
1/2 H_uu + lambda * sum_{v in S} H_uv -- and that is the point: nothing about the objective needed
to be rewritten for a second tower, only the inventory it ranges over.

Two details are two-tower specific. Layer indices collide across towers (both have a layer 5), so
the per-layer cap keys on a composite index; and the first/last-layer protection is applied per
tower, since each has its own input and output boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from .bundle import VLMBundle
from .masks import masked
from .units import VLMRegistry


def layer_keys(bundle: VLMBundle, reg: VLMRegistry) -> np.ndarray:
    """A single layer index per unit that never collides between towers."""
    off = {"lm": 0, "vis": bundle.towers["lm"].num_layers}
    return np.array([off[u.tower] + u.layer_idx for u in reg.units], dtype=np.int64)


def protected_layers(bundle: VLMBundle, protect: int = 2) -> Set[int]:
    out: Set[int] = set()
    off = 0
    for t in ("lm", "vis"):
        n = bundle.towers[t].num_layers
        for i in range(protect):
            out.add(off + i)
            out.add(off + n - 1 - i)
        off += n
    return out


def solve(H: np.ndarray, reg: VLMRegistry, bundle: VLMBundle, ratio: float, lam: float,
          cap_mult: float = 1.5, protect: int = 2) -> Tuple[List[int], float]:
    from cocurve.path import solve_with_breakpoint
    pruned, actual, _ = solve_with_breakpoint(
        H.astype(np.float64), reg.cost_vector().numpy().astype(np.float64), ratio, float(lam),
        layer_keys(bundle, reg), cap_mult * ratio, protected_layers(bundle, protect))
    return sorted(int(x) for x in pruned), float(actual)


def enumerate_family(H: np.ndarray, reg: VLMRegistry, bundle: VLMBundle, ratio: float,
                     cap_mult: float = 1.5, protect: int = 2,
                     lam_max: float = 1.0, max_members: int = 4000,
                     max_iters: int = 20000) -> List[Dict]:
    """Every distinct solution of the path over lambda in [0, lam_max], with exact endpoints.

    ``max_members`` bounds distinct solutions; ``max_iters`` bounds breakpoints visited, and the two
    are not the same bound. On most models the path has a few hundred breakpoints and both are slack.
    On InternVL3-14B the breakpoints are spaced about 2e-6 apart -- some 400k of them across
    [0, 1] -- while the distinct masks number in the hundreds, so a loop bounded only by distinct
    solutions runs for tens of hours and looks like a hang. The iteration bound makes the cost
    predictable; how far it actually got is returned rather than assumed, so a truncated path is
    visible instead of silent."""
    from cocurve.path import solve_with_breakpoint
    costs = reg.cost_vector().numpy().astype(np.float64)
    Hd = H.astype(np.float64)
    lk = layer_keys(bundle, reg)
    prot = protected_layers(bundle, protect)
    out: List[Dict] = []
    seen = set()
    lam = 0.0
    _last = [0]
    iters = 0
    while lam <= lam_max and len(out) < max_members and iters < max_iters:
        iters += 1
        pruned, actual, lam_next = solve_with_breakpoint(Hd, costs, ratio, lam, lk,
                                                        cap_mult * ratio, prot)
        key = tuple(sorted(int(x) for x in pruned))
        if key not in seen:
            seen.add(key)
            out.append(dict(lam_lo=float(lam), pruned=list(key), actual_ratio=float(actual)))
        if not np.isfinite(lam_next) or lam_next <= lam + 1e-12:
            break
        lam = float(lam_next) + 1e-9
        # The enumeration is exact and therefore unbounded in the number of breakpoints a model can
        # have; printing where it is turns a silent hour into a legible one.
        if len(out) and len(out) % 200 == 0 and len(out) != _last[0]:
            _last[0] = len(out)
            print(f"    lambda path: {len(out)} solutions so far, lambda={lam:.6f} "
                  f"({iters} breakpoints)", flush=True)
    if out:
        out[0]["_coverage"] = dict(iters=iters, lam_reached=float(lam), lam_max=float(lam_max),
                                  truncated=bool(iters >= max_iters or len(out) >= max_members))
        if out[0]["_coverage"]["truncated"]:
            print(f"    lambda path TRUNCATED at {iters} breakpoints / {len(out)} solutions; "
                  f"covered lambda in [0, {lam:.6g}] of [0, {lam_max}]", flush=True)
    return out


def heldout_risk(bundle: VLMBundle, reg: VLMRegistry, removed: Sequence[int],
                 batches: List[Dict[str, torch.Tensor]], t_idx: torch.Tensor,
                 t_prob: torch.Tensor) -> float:
    """Measured support-conditioned token-level KL on held-out image-text pairs.

    This is the quantity lambda is selected on: the measured risk of the binary mask, not the
    quadratic that proposed it, and with no label or benchmark anywhere in the loop.  Both
    distributions are normalized on the dense teacher's fixed top-r support.
    """
    dev = bundle.device
    idx, prob = t_idx.to(dev), t_prob.to(dev)
    rows = []
    with torch.no_grad(), masked(bundle, reg, set(int(x) for x in removed)):
        for enc in batches:
            sm = enc["_score_mask"]
            fwd = {k: v for k, v in enc.items() if not k.startswith("_")}
            rows.append(bundle.model(**fwd, use_cache=False).logits[sm].float())
    s = torch.gather(torch.cat(rows), -1, idx)
    q = torch.softmax(s, dim=-1)
    per_pos = (prob * (torch.log(prob.clamp_min(1e-12)) - torch.log(q.clamp_min(1e-12)))).sum(-1)
    return float(per_pos.mean())


def select_lambda(bundle: VLMBundle, reg: VLMRegistry, family: List[Dict],
                  batches: List[Dict[str, torch.Tensor]], t_idx: torch.Tensor,
                  t_prob: torch.Tensor, verbose: bool = True) -> Tuple[Dict, List[Dict]]:
    scored = []
    for i, m in enumerate(family):
        r = heldout_risk(bundle, reg, m["pruned"], batches, t_idx, t_prob)
        scored.append(dict(**{k: v for k, v in m.items() if k != "pruned"}, risk=r, member=i))
        if verbose:
            print(f"    lam>={m['lam_lo']:.6f}  |S|={len(m['pruned']):5d}  "
                  f"ratio={m['actual_ratio']:.4f}  risk={r:.6f}", flush=True)
    best = min(scored, key=lambda x: x["risk"])
    return family[best["member"]], scored


def split_by(reg: VLMRegistry, removed: Sequence[int]) -> Dict[str, int]:
    rem = set(int(x) for x in removed)
    out: Dict[str, int] = {}
    for t in ("lm", "vis"):
        for k in ("attn_head", "ffn_group"):
            out[f"{t}.{k}"] = sum(1 for u in reg.by(t, k) if u.unit_id in rem)
    cost = {t: sum(u.cost for u in reg.by(t) if u.unit_id in rem) for t in ("lm", "vis")}
    tot = sum(cost.values()) or 1.0
    out["cost_share_lm"] = round(cost["lm"] / tot, 4)
    out["cost_share_vis"] = round(cost["vis"] / tot, 4)
    return out
