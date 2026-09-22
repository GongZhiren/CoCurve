"""Build the two-tower co-pruning curvature matrix for a VLM.

usage: python3 scripts/vlm_build_h.py <model_id> <out_dir> [--n 128] [--bs 4] [--top-r 64]
                                      [--ffn-groups 16] [--seed 0] [--skip 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from cocurve.vlm.bundle import load_vlm                     # noqa: E402
from cocurve.vlm.units import build_registry, summarise      # noqa: E402
from cocurve.vlm.calib import (build_batch, fingerprint,      # noqa: E402
                                   load_pairs, teacher_cache)
from cocurve.vlm.hessian import assemble, run_shard          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--top-r", type=int, default=64)
    ap.add_argument("--ffn-groups", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--no-assemble", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    b = load_vlm(a.model_id)
    b.processor.tokenizer.padding_side = "left"
    reg = build_registry(b, ffn_groups_per_layer=a.ffn_groups)
    print(f"{a.model_id}\n{summarise(reg)}", flush=True)

    pairs = load_pairs(a.n, seed=a.seed, skip=a.skip)
    batches = [build_batch(b, pairs[i:i + a.bs]) for i in range(0, len(pairs), a.bs)]
    P = int(sum(int(e["_score_mask"].sum()) for e in batches))
    print(f"calibration: {len(pairs)} pairs / {len(batches)} batches / {P} scored positions "
          f"/ fingerprint {fingerprint(pairs)}", flush=True)

    t_idx, t_prob, t_logit = teacher_cache(b, batches, top_r=a.top_r)
    print(f"teacher cache ready: {tuple(t_idx.shape)}  ({time.time()-t0:.0f}s)", flush=True)

    run_shard(b, reg, batches, t_idx, t_prob, t_logit, out,
              shard=a.shard, n_shards=a.n_shards)
    if a.no_assemble:
        print(f"shard {a.shard}/{a.n_shards} features done ({time.time()-t0:.0f}s)", flush=True)
        return
    H = assemble(reg, out, a.n_shards, P, a.top_r)

    d = np.diag(H)
    off = np.abs(H).sum(1) - np.abs(d)
    meta = dict(model_id=a.model_id, family=b.family, units=reg.n, positions=P, top_r=a.top_r,
                ffn_groups=a.ffn_groups, n_pairs=len(pairs), seed=a.seed,
                calib_fingerprint=fingerprint(pairs), minutes=round((time.time() - t0) / 60, 1),
                towers={k: dict(layers=v.num_layers, hidden=v.hidden_size, heads=v.num_heads,
                                kv=v.kv_heads, head_dim=v.head_dim, inter=v.intermediate_size)
                        for k, v in b.towers.items()},
                cost_share={f"{t}.{u}": round(sum(x.cost for x in reg.by(t, u)) / reg.total_cost(), 4)
                            for t in ("lm", "vis") for u in ("attn_head", "ffn_group")},
                diag_mean=float(d.mean()), offdiag_mean=float(off.mean()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
