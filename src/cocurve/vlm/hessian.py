"""The co-pruning curvature matrix of a VLM, from single-unit ablations.

Identical algebra to the text-only estimator, and deliberately so: the whitened per-unit response

    d_u = sqrt(p0) * ( (z0 - z_{-u}) - E_{p0}[z0 - z_{-u}] )

restricted to the dense model's top-r support, stacked into D, gives H = D^T D / P. The diagonal is
OBD saliency; the off-diagonal is the co-pruning curvature. Nothing here needs a gradient, which is
the whole reason the same construction survives the move to a two-tower model: M forward passes,
one per removable unit, over both towers at once.

The loop is embarrassingly parallel over units, and a 7B two-tower model has a couple of thousand of
them, so it can be sharded across devices. Each shard owns a strided subset -- strided, not blocked,
so any single shard is still a representative sample of towers and depths -- and writes its own
feature file with its own progress record. Nothing is shared between shards, so there is no race,
and ``assemble`` stitches them into H once they are all present.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .bundle import VLMBundle
from .masks import masked
from .units import VLMRegistry


def _feature(t_logit: torch.Tensor, t_prob: torch.Tensor, s_logit: torch.Tensor) -> torch.Tensor:
    delta = t_logit - s_logit
    centred = delta - (t_prob * delta).sum(-1, keepdim=True)
    return torch.sqrt(t_prob.clamp_min(1e-12)) * centred


def _paths(out_dir: Path, shard: int, n_shards: int) -> Tuple[Path, Path, Path]:
    tag = f"s{shard}of{n_shards}"
    return (out_dir / f"features_{tag}.memmap", out_dir / f"progress_{tag}.json",
            out_dir / f"kl_{tag}.npy")


def run_shard(bundle: VLMBundle, reg: VLMRegistry, batches: List[Dict[str, torch.Tensor]],
              t_idx: torch.Tensor, t_prob: torch.Tensor, t_logit: torch.Tensor,
              out_dir: Path, shard: int = 0, n_shards: int = 1, log_every: int = 25) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = bundle.device
    P, R = t_idx.shape
    mine = [u for i, u in enumerate(reg.units) if i % n_shards == shard]
    ids = [u.unit_id for u in mine]
    feat_path, prog_path, kl_path = _paths(out_dir, shard, n_shards)

    done: set = set()
    if prog_path.exists():
        pr = json.loads(prog_path.read_text())
        if pr.get("shape") == [len(mine), P, R] and pr.get("unit_ids") == ids:
            done = {int(x) for x in pr.get("done", [])}
    D = np.memmap(feat_path, dtype=np.float32,
                  mode=("r+" if done and feat_path.exists() else "w+"),
                  shape=(len(mine), P, R))
    kl = np.load(kl_path) if (kl_path.exists() and done) else np.zeros(len(mine), dtype=np.float64)

    idx_d, prob_d, logit_d = t_idx.to(dev), t_prob.to(dev), t_logit.to(dev)
    t0 = time.time()
    for row, u in enumerate(mine):
        if u.unit_id in done:
            continue
        parts = []
        with torch.no_grad(), masked(bundle, reg, {u.unit_id}):
            for enc in batches:
                sm = enc["_score_mask"]
                fwd = {k: v for k, v in enc.items() if not k.startswith("_")}
                parts.append(bundle.model(**fwd, use_cache=False).logits[sm].float())
        s_top = torch.gather(torch.cat(parts), -1, idx_d)
        D[row] = _feature(logit_d, prob_d, s_top).cpu().numpy()
        q = torch.softmax(s_top, dim=-1)
        kl[row] = float((prob_d * (torch.log(prob_d.clamp_min(1e-12))
                                   - torch.log(q.clamp_min(1e-12)))).sum(-1).mean())
        done.add(u.unit_id)
        if (row + 1) % log_every == 0 or row + 1 == len(mine):
            D.flush()
            np.save(kl_path, kl)
            prog_path.write_text(json.dumps({"shape": [len(mine), P, R], "unit_ids": ids,
                                             "done": sorted(done)}))
            el = time.time() - t0
            n_new = row + 1 - len(done & set(ids[:0]))
            print(f"  [shard {shard}/{n_shards}] [{len(done)}/{len(mine)}] {u.key:30s} "
                  f"elapsed {el/60:.1f}m eta "
                  f"{el / max(1, row + 1) * (len(mine) - row - 1) / 60:.1f}m", flush=True)
    D.flush()
    np.save(kl_path, kl)
    prog_path.write_text(json.dumps({"shape": [len(mine), P, R], "unit_ids": ids,
                                     "done": sorted(done)}))


def assemble(reg: VLMRegistry, out_dir: Path, n_shards: int, P: int, R: int,
             device: Optional[str] = None) -> np.ndarray:
    """Stitch every shard's features into H = D^T D / P, and the per-unit exact KL beside it."""
    M = reg.n
    F = np.zeros((M, P * R), dtype=np.float32)
    kl = np.zeros(M, dtype=np.float64)
    seen = np.zeros(M, dtype=bool)
    for sh in range(n_shards):
        fp, pp, kp = _paths(out_dir, sh, n_shards)
        pr = json.loads(pp.read_text())
        ids = [int(x) for x in pr["unit_ids"]]
        missing = set(ids) - {int(x) for x in pr["done"]}
        if missing:
            raise RuntimeError(f"shard {sh} incomplete: {len(missing)} units unfinished")
        Ds = np.memmap(fp, dtype=np.float32, mode="r", shape=(len(ids), P, R))
        ks = np.load(kp)
        for row, uid in enumerate(ids):
            F[uid] = np.asarray(Ds[row]).reshape(-1)
            kl[uid] = ks[row]
            seen[uid] = True
        del Ds
    if not seen.all():
        raise RuntimeError(f"{int((~seen).sum())} units missing across shards")
    try:
        dv = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        Ft = torch.from_numpy(F).to(dv)
        H = (Ft @ Ft.T).double().cpu().numpy() / float(P)
        del Ft
        torch.cuda.empty_cache()
    except Exception:
        H = (F.astype(np.float64) @ F.astype(np.float64).T) / float(P)
    np.save(out_dir / "H.npy", H.astype(np.float32))
    np.save(out_dir / "single_unit_kl.npy", kl)
    return H.astype(np.float32)
