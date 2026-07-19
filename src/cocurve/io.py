from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .utils import dump_json, ensure_dir


class ArtifactStore:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = Path(run_dir)
        self.meta_dir = ensure_dir(self.run_dir / "meta")
        self.logs_dir = ensure_dir(self.run_dir / "logs")
        self.cache_dir = ensure_dir(self.run_dir / "cache")
        self.matrices_dir = ensure_dir(self.run_dir / "matrices")
        self.masks_dir = ensure_dir(self.run_dir / "masks")
        self.eval_dir = ensure_dir(self.run_dir / "eval")

    def snapshot_config(self, source_config_path: str) -> None:
        dst = self.meta_dir / "config_snapshot.yaml"
        shutil.copyfile(source_config_path, dst)

    def save_json(self, relative: str, payload: Dict[str, Any]) -> None:
        dump_json(self.run_dir / relative, payload)

    def append_jsonl(self, relative: str, payload: Dict[str, Any]) -> None:
        path = self.run_dir / relative
        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{json.dumps(payload, ensure_ascii=False)}\n")

    def save_numpy(self, relative: str, array: np.ndarray) -> None:
        path = self.run_dir / relative
        ensure_dir(path.parent)
        np.save(path, array)

    def save_lines(self, relative: str, lines: Iterable[str]) -> None:
        path = self.run_dir / relative
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(f"{line}\n")

    def list_json_files(self, root: str, suffixes: List[str]) -> List[Path]:
        root_path = Path(root)
        if not root_path.exists():
            return []
        files: List[Path] = []
        for suffix in suffixes:
            files.extend(root_path.rglob(suffix))
        return sorted(files)
