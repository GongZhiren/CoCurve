from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .io import ArtifactStore


def build_h_matrix(
    cfg: Dict[str, object],
    store: ArtifactStore,
    features: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if features is None:
        meta = store.run_dir / "cache/unit_features_meta.json"
        if not meta.exists():
            raise FileNotFoundError("Missing unit_features_meta.json. Run ablation stage first.")
        import json

        with meta.open("r", encoding="utf-8") as f:
            feature_meta = json.load(f)
        shape = tuple(feature_meta["shape"])
        feature_path = feature_meta["path"]
        features = np.memmap(feature_path, dtype=np.float32, mode="r", shape=shape)
        num_selected_tokens = int(feature_meta["num_selected_tokens"])
    else:
        num_selected_tokens = int(features.shape[1])

    # features: [num_units, num_selected_tokens, top_r]
    num_units = int(features.shape[0])
    h = np.zeros((num_units, num_units), dtype=np.float64)
    block_size = int(cfg["calibration"]["ablation"].get("chunk_size_units", 32))
    if block_size <= 0:
        block_size = 32
    flat_dim = int(features.shape[1] * features.shape[2])

    for i in range(0, num_units, block_size):
        i_end = min(num_units, i + block_size)
        fi = np.asarray(features[i:i_end], dtype=np.float32).reshape(i_end - i, flat_dim)
        for j in range(i, num_units, block_size):
            j_end = min(num_units, j + block_size)
            fj = np.asarray(features[j:j_end], dtype=np.float32).reshape(j_end - j, flat_dim)
            block = (fi @ fj.T) / max(1, num_selected_tokens)
            h[i:i_end, j:j_end] = block
            if i != j:
                h[j:j_end, i:i_end] = block.T
    h = h.astype(np.float32)

    matrix_cfg = cfg["pruning"]["matrix"]
    if matrix_cfg.get("symmetrize", True):
        h = 0.5 * (h + h.T)

    diag = np.diag(h).copy()
    min_tol = float(matrix_cfg.get("min_diag_tolerance", -1e-8))
    if matrix_cfg.get("assert_non_negative_diagonal", True) and np.min(diag) < min_tol:
        bad = float(np.min(diag))
        raise ValueError(f"H diagonal has negative values below tolerance: min={bad:.6e}")

    store.save_numpy("matrices/H.npy", h.astype(np.float32))
    store.save_numpy("matrices/H_diag.npy", diag.astype(np.float32))
    return h, diag
