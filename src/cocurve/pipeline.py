from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .ablation import run_single_unit_ablations
from .calibration import calibration_signature, run_full_model_collection
from .config import load_config, model_config, validate_config
from .eval import evaluate_json_benchmarks, prepare_standard_benchmarks, run_quality_gates
from .fisher import build_h_matrix
from .io import ArtifactStore
from .model import load_model_bundle
from .prune import apply_physical_prune_inplace
from .solver import greedy_budget_prune
from .types import RunContext
from .units import build_unit_registry, unit_cost_vector
from .utils import dump_json, env_git_commit, set_seed, timestamp


def _protected_layers(num_layers: int, protect_first: int, protect_last: int) -> set:
    """Layer indices that must never be pruned (first/last bands).

    Early layers build core token representations and the final layers feed the
    LM head; gutting either tends to inflate perplexity disproportionately. This
    is a single clean knob rather than per-layer special-casing.
    """
    protected: set = set()
    if protect_first > 0:
        protected.update(range(0, min(protect_first, num_layers)))
    if protect_last > 0:
        protected.update(range(max(0, num_layers - protect_last), num_layers))
    return protected


class CoCurvePipeline:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.cfg = load_config(config_path)
        validate_config(self.cfg)

    def _run_context(
        self,
        model_key: str | None = None,
        run_name: str | None = None,
        run_dir: str | None = None,
    ) -> RunContext:
        project_cfg = self.cfg["project"]
        run_cfg = self.cfg["run"]
        model_name = model_key or run_cfg["model_key"]
        run_name_local = run_name or run_cfg["run_name"]
        if run_dir is None:
            run_id = f"{run_name_local}_{timestamp()}"
            run_dir_path = Path(project_cfg["output_root"]) / model_name / run_id
        else:
            run_dir_path = Path(run_dir)
        return RunContext(model_key=model_name, run_name=run_name_local, run_dir=str(run_dir_path), seed=int(project_cfg["seed"]))

    @staticmethod
    def _require(path: Path, hint: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}. {hint}")

    def run(
        self,
        stages: Sequence[str] | None = None,
        model_key: str | None = None,
        run_name: str | None = None,
        run_dir: str | None = None,
    ) -> Dict[str, object]:
        stages = list(stages or ["all"])
        stage_set = set(stages)
        run_all = "all" in stage_set
        ctx = self._run_context(model_key=model_key, run_name=run_name, run_dir=run_dir)
        set_seed(ctx.seed)
        store = ArtifactStore(ctx.run_dir)
        run_started = time.perf_counter()

        local_cfg = dict(self.cfg)
        if model_key is not None:
            local_cfg["run"] = dict(self.cfg["run"])
            local_cfg["run"]["model_key"] = model_key

        need_collect = run_all or bool(stage_set & {"collect", "ablation", "matrix", "solve", "quality", "json_eval", "apply"})
        need_ablation = run_all or bool(stage_set & {"ablation", "matrix", "solve", "quality", "json_eval", "apply"})
        need_matrix = run_all or bool(stage_set & {"matrix", "solve", "quality", "json_eval", "apply"})
        need_solve = run_all or bool(stage_set & {"solve", "quality", "json_eval", "apply"})
        need_quality = run_all or "quality" in stage_set
        need_json_eval = run_all or "json_eval" in stage_set
        need_std_prep = run_all or "std_prep" in stage_set
        need_apply = run_all or "apply" in stage_set
        need_model = need_collect or need_ablation or need_matrix or need_solve or need_quality or need_json_eval or need_apply

        model_cfg = model_config(local_cfg)
        pruning_cfg = local_cfg["pruning"]
        bundle = None
        registry = None
        costs = None
        if need_model:
            bundle = load_model_bundle(model_cfg, local_cfg["model"]["tokenizer"])
            registry = build_unit_registry(
                num_layers=bundle.num_layers,
                num_heads=bundle.num_heads,
                hidden_size=bundle.hidden_size,
                intermediate_size=bundle.intermediate_size,
                ffn_groups_per_layer=int(pruning_cfg["units"]["ffn_groups_per_layer"]),
                cost_type=str(pruning_cfg["units"]["cost_type"]),
                kv_heads=bundle.kv_heads,
                head_dim=bundle.head_dim,
            )
            costs = unit_cost_vector(registry).numpy()

        meta = {
            "model_key": ctx.model_key,
            "model_path": model_cfg["path"],
            "run_name": ctx.run_name,
            "num_layers": bundle.num_layers if bundle is not None else None,
            "num_heads": bundle.num_heads if bundle is not None else None,
            "num_kv_heads": bundle.kv_heads if bundle is not None else None,
            "attention_pattern": bundle.attn_pattern if bundle is not None else None,
            "num_units": registry.num_units if registry is not None else None,
            "git_commit": env_git_commit(),
            "config_path": self.config_path,
        }
        store.save_json("meta/run_meta.json", meta)
        store.save_json(
            "meta/stage_plan.json",
            {
                "requested_stages": sorted(stage_set),
                "run_all": run_all,
                "need_collect": need_collect,
                "need_ablation": need_ablation,
                "need_matrix": need_matrix,
                "need_solve": need_solve,
                "need_quality": need_quality,
                "need_json_eval": need_json_eval,
                "need_std_prep": need_std_prep,
                "need_apply": need_apply,
            },
        )
        if self.cfg["run"].get("copy_config_to_meta", True):
            store.snapshot_config(self.config_path)
        store.append_jsonl(
            "logs/stage_events.jsonl",
            {"event": "run_started", "ts": timestamp(), "model_key": ctx.model_key, "run_dir": ctx.run_dir},
        )

        stage_metrics: Dict[str, Dict[str, object]] = {}

        def _run_stage(name: str, fn):
            started = time.perf_counter()
            store.append_jsonl("logs/stage_events.jsonl", {"event": "stage_started", "stage": name, "ts": timestamp()})
            try:
                out = fn()
                elapsed = round(time.perf_counter() - started, 6)
                stage_metrics[name] = {"status": "ok", "elapsed_sec": elapsed}
                store.append_jsonl(
                    "logs/stage_events.jsonl",
                    {"event": "stage_finished", "stage": name, "elapsed_sec": elapsed, "ts": timestamp()},
                )
                return out
            except Exception as exc:
                elapsed = round(time.perf_counter() - started, 6)
                stage_metrics[name] = {
                    "status": "failed",
                    "elapsed_sec": elapsed,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                store.save_json("meta/stage_metrics.json", stage_metrics)
                store.append_jsonl(
                    "logs/stage_events.jsonl",
                    {
                        "event": "stage_failed",
                        "stage": name,
                        "elapsed_sec": elapsed,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-12:],
                        "ts": timestamp(),
                    },
                )
                raise

        if need_collect and (run_all or "collect" in stage_set):
            assert bundle is not None
            top_idx_np, _top_prob_np, positions_np = _run_stage(
                "collect",
                lambda: run_full_model_collection(bundle, local_cfg, store),
            )
            stage_metrics["collect"].update(
                {
                    "num_samples": int(top_idx_np.shape[0]),
                    "seq_len_minus_one": int(top_idx_np.shape[1]),
                    "top_r": int(top_idx_np.shape[2]),
                    "selected_token_count": int(positions_np.sum()),
                }
            )
        elif need_collect:
            self._require(Path(ctx.run_dir) / "cache/top_indices.npy", "Run collect stage first.")
            meta_path = Path(ctx.run_dir) / "cache/calibration_meta.json"
            self._require(meta_path, "Run collect stage first.")
            with meta_path.open("r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            cached_signature = cached_meta.get("calibration_signature")
            current_signature = calibration_signature(local_cfg["calibration"])
            if cached_signature != current_signature:
                raise ValueError(
                    "Existing collect cache was built with a different calibration file/config. "
                    "Use a new --run-dir or rerun the collect stage before reusing downstream artifacts."
                )
            stage_metrics["collect"] = {"status": "reused", "elapsed_sec": 0.0}

        if need_ablation and (run_all or "ablation" in stage_set):
            assert bundle is not None and registry is not None
            _features, single_unit_kl = _run_stage(
                "ablation",
                lambda: run_single_unit_ablations(bundle, local_cfg, store, registry.units),
            )
            stage_metrics["ablation"].update(
                {
                    "num_units": int(registry.num_units),
                    "single_unit_kl_mean": float(np.mean(single_unit_kl)),
                    "single_unit_kl_std": float(np.std(single_unit_kl)),
                }
            )
        elif need_ablation:
            self._require(Path(ctx.run_dir) / "cache/unit_features_meta.json", "Run ablation stage first.")
            stage_metrics["ablation"] = {"status": "reused", "elapsed_sec": 0.0}

        if need_matrix and (run_all or "matrix" in stage_set):
            h, diag = _run_stage("matrix", lambda: build_h_matrix(local_cfg, store))
            stage_metrics["matrix"].update(
                {
                    "matrix_shape": [int(h.shape[0]), int(h.shape[1])],
                    "diag_min": float(np.min(diag)),
                    "diag_max": float(np.max(diag)),
                    "diag_mean": float(np.mean(diag)),
                }
            )
        elif need_matrix:
            h = np.load(Path(ctx.run_dir) / "matrices/H.npy")
            stage_metrics["matrix"] = {"status": "reused", "elapsed_sec": 0.0, "matrix_shape": [int(h.shape[0]), int(h.shape[1])]}
        else:
            matrix_dim = registry.num_units if registry is not None else 1
            h = np.zeros((matrix_dim, matrix_dim), dtype=np.float32)

        if need_solve and (run_all or "solve" in stage_set):
            assert costs is not None
            result = _run_stage(
                "solve",
                lambda: greedy_budget_prune(
                    h=h,
                    costs=costs,
                    prune_ratio=float(pruning_cfg["solver"]["prune_ratio"]),
                    allow_budget_overshoot=bool(pruning_cfg["solver"]["allow_budget_overshoot"]),
                    normalize_by_cost=bool(pruning_cfg["solver"]["normalize_by_cost"]),
                    unit_layers=[int(spec.layer_idx) for spec in registry.units],
                    max_pruned_cost_fraction_per_layer=pruning_cfg["solver"].get("max_pruned_cost_fraction_per_layer"),
                    protected_layers=_protected_layers(
                        num_layers=max((int(s.layer_idx) for s in registry.units), default=-1) + 1,
                        protect_first=int(pruning_cfg["solver"].get("protect_first_layers", 0) or 0),
                        protect_last=int(pruning_cfg["solver"].get("protect_last_layers", 0) or 0),
                    ),
                    interaction_strength=float(pruning_cfg["solver"].get("interaction_strength", 1.0)),
                ),
            )
            store.save_json(
                "masks/prune_solution.json",
                {
                    "selected_units": result.selected_units,
                    "pruned_units": result.pruned_units,
                    "total_cost": result.total_cost,
                    "pruned_cost": result.pruned_cost,
                    "target_pruned_cost": result.target_pruned_cost,
                    "target_prune_ratio": result.target_prune_ratio,
                    "actual_prune_ratio": result.actual_prune_ratio,
                    "overshoot_cost": result.overshoot_cost,
                    "score_trace": result.score_trace,
                },
            )
            stage_metrics["solve"].update(
                {
                    "num_selected_units": int(len(result.selected_units)),
                    "num_pruned_units": int(len(result.pruned_units)),
                    "target_prune_ratio": float(result.target_prune_ratio),
                    "actual_prune_ratio": float(result.actual_prune_ratio),
                    "overshoot_cost": float(result.overshoot_cost),
                }
            )
        elif need_solve:
            solution = Path(ctx.run_dir) / "masks/prune_solution.json"
            self._require(solution, "Run solve stage first.")
            payload = solution.read_text(encoding="utf-8")
            result = json.loads(payload)
            stage_metrics["solve"] = {"status": "reused", "elapsed_sec": 0.0}
        else:
            matrix_dim = registry.num_units if registry is not None else 0
            result = {"selected_units": list(range(matrix_dim))}

        selected_units = result.selected_units if hasattr(result, "selected_units") else result["selected_units"]

        if need_quality:
            assert bundle is not None and registry is not None
            q_report = _run_stage(
                "quality",
                lambda: run_quality_gates(
                    bundle=bundle,
                    cfg=local_cfg,
                    store=store,
                    units=registry.units,
                    selected_units=selected_units,
                ),
            )
            quality_payload = {
                "surrogate_vs_real_spearman": q_report.surrogate_vs_real_spearman,
                "surrogate_vs_real_pearson": q_report.surrogate_vs_real_pearson,
                "diagonal_damage_spearman": q_report.diagonal_damage_spearman,
                "mask_vs_physical_max_abs_err": q_report.mask_vs_physical_max_abs_err,
                "passed": q_report.passed,
            }
            stage_metrics["quality"].update(quality_payload)
        else:
            quality_payload = {}

        if need_json_eval:
            assert bundle is not None and registry is not None
            json_eval_report = _run_stage(
                "json_eval",
                lambda: evaluate_json_benchmarks(bundle, local_cfg, store, selected_units=selected_units, units=registry.units),
            )
            stage_metrics["json_eval"].update({"report_keys": sorted(list(json_eval_report.keys()))})
        else:
            json_eval_report = {}

        if need_std_prep:
            standard_report = _run_stage("std_prep", lambda: prepare_standard_benchmarks(local_cfg, store))
            stage_metrics["std_prep"].update({"report_keys": sorted(list(standard_report.keys()))})
        else:
            standard_report = {}

        if need_apply:
            assert bundle is not None and registry is not None
            if bool(local_cfg["pruning"]["physical_prune"]["enabled"]):
                attn_by_layer: Dict[int, List] = {}
                ffn_by_layer: Dict[int, List] = {}
                for spec in registry.units:
                    if spec.unit_type == "attn_head":
                        attn_by_layer.setdefault(spec.layer_idx, []).append(spec)
                    else:
                        ffn_by_layer.setdefault(spec.layer_idx, []).append(spec)
                backups = _run_stage(
                    "apply",
                    lambda: apply_physical_prune_inplace(bundle, attn_by_layer, ffn_by_layer, set(selected_units)),
                )
                structural_plan = {
                    "selected_units": sorted(int(u) for u in selected_units),
                    "layers": [],
                }
                selected_set = set(selected_units)
                for layer_idx in range(bundle.num_layers):
                    layer_attn_specs = attn_by_layer.get(layer_idx, [])
                    kept_q_heads: List[int] = []
                    kept_kv_heads: List[int] = []
                    for spec in layer_attn_specs:
                        if spec.unit_id in selected_set:
                            kept_q_heads.extend(int(h) for h in spec.metadata.get("query_head_indices", []))
                            kept_kv_heads.append(int(spec.metadata.get("kv_head_index", -1)))
                    layer_ffn_specs = ffn_by_layer.get(layer_idx, [])
                    kept_ffn_spans: List[List[int]] = []
                    for spec in layer_ffn_specs:
                        if spec.unit_id in selected_set:
                            start = int(spec.metadata.get("start_channel", 0))
                            size = int(spec.metadata.get("group_size", 0))
                            kept_ffn_spans.append([start, start + size])
                    structural_plan["layers"].append(
                        {
                            "layer_idx": int(layer_idx),
                            "kept_query_heads": sorted(set(kept_q_heads)),
                            "kept_kv_heads": sorted(set([h for h in kept_kv_heads if h >= 0])),
                            "kept_ffn_channel_spans": kept_ffn_spans,
                        }
                    )
                store.save_json("masks/structural_prune_plan.json", structural_plan)
                store.save_json("masks/applied_physical_prune.json", {"enabled": True, "num_selected_units": len(selected_units)})
                stage_metrics["apply"].update(
                    {"num_selected_units": int(len(selected_units)), "modified_tensor_slices": int(len(backups))}
                )
                if bool(local_cfg["pruning"]["physical_prune"]["save_pruned_model"]):
                    save_dir = Path(ctx.run_dir) / "pruned_model"
                    bundle.model.save_pretrained(save_dir)
                    bundle.tokenizer.save_pretrained(save_dir)
                    store.save_json("masks/pruned_model_saved.json", {"path": str(save_dir)})
            else:
                stage_metrics["apply"] = {"status": "disabled", "elapsed_sec": 0.0}

        total_elapsed = round(time.perf_counter() - run_started, 6)
        store.save_json("meta/stage_metrics.json", stage_metrics)
        store.append_jsonl(
            "logs/stage_events.jsonl",
            {"event": "run_finished", "ts": timestamp(), "elapsed_sec": total_elapsed, "stage_count": len(stage_metrics)},
        )
        summary = {
            "run_dir": ctx.run_dir,
            "model_key": ctx.model_key,
            "quality_gate": quality_payload,
            "json_eval": json_eval_report,
            "standard_prepared": standard_report,
            "stage_metrics": stage_metrics,
            "elapsed_sec": total_elapsed,
        }
        dump_json(Path(ctx.run_dir) / "meta/summary.json", summary)
        return summary

    def run_multi_models(
        self,
        model_keys: List[str],
        stages: Sequence[str] | None = None,
        run_base_dir: str | None = None,
    ) -> Dict[str, object]:
        reports: Dict[str, object] = {}
        for model_key in model_keys:
            model_run_dir = None
            if run_base_dir is not None:
                model_run_dir = str(Path(run_base_dir) / model_key)
            reports[model_key] = self.run(
                stages=list(stages) if stages is not None else None,
                model_key=model_key,
                run_name=f"multi_{model_key}",
                run_dir=model_run_dir,
            )
        return reports
