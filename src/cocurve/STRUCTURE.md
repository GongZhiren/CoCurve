# CoCurve Code Layout

- `model.py`, `units.py`: model loading and the shared structured-unit registry.
- `calibration.py`, `ablation.py`, `fisher.py`: teacher outputs, unit
  interventions, and curvature construction.
- `solver.py`, `path.py`: budgeted greedy selection and exact interaction-path
  enumeration.
- `prune.py`: runtime masks and physical checkpoint slicing.
- `eval.py`, `benchmark_scoring.py`, `prompting.py`: language-model evaluation.
- `vlm/`: the corresponding multimodal calibration, selection, pruning,
  evaluation, and recovery path.
- `artifacts.py`: publication artifact discovery and integrity verification.
- `pipeline.py`: configuration-driven orchestration for the LLM stages.

User-facing commands live in `scripts/`; the recommended reproduction paths
are documented in `README.md` and `REPRODUCIBILITY.md`.
