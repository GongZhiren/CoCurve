# News

## 2026-11-19 — reconstruction and checkpoint reuse

- Added fixed-mask OBS and FFN refinement as optional CoCurve recovery paths.
- Added nine evaluated FFN-refinement structures with attention and per-layer
  budget consistency checks.
- Added LoRA recovery starting from an already reconstructed checkpoint.
- Added loading of heterogeneous sliced checkpoints and streaming export into
  portable safetensors shards.
- Fixed fully removed FFNs to eliminate their parameters and prevent recovery
  from reviving deleted computation.
- Added numerical, serialization, integrity, and small-model training checks.

## 2026-09-22 — paper reproduction release

- Added publication-locked curvature matrices, unit registries, masks, and
  reference metrics for six LLMs and three VLMs.
- Added checksum validation and exact reconstruction tests for all 42 released
  operating points.
- Added the exact held-out interaction-path selector used by the paper.
- Added direct evaluation, lightweight LLM/VLM recovery, NF4/INT8, physical
  slicing, and isolated efficiency workflows.
- Added packaging, continuous integration, release checks, and a detailed
  reproducibility guide.

## 2026-07 — initial release

- Released the core LLM CoCurve pipeline.
