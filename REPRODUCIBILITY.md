# Reproducibility guide

This document separates three reproduction levels so users do not pay for work
they do not need.

## Level A: verify a paper result

Use the released mask and run evaluation:

```bash
cocurve-artifacts verify --model llama-3.1-8b-instruct
python scripts/prepare_paper_datasets.py
python scripts/evaluate_paper_mask.py \
  --model llama-3.1-8b-instruct --ratio 30 --tasks all
```

This is the recommended path for checking reported quality. It skips all
curvature computation and uses the exact mask evaluated in the paper.

## Level B: reproduce mask selection from H

```bash
python scripts/reproduce_paper_mask.py \
  --model llama-3.1-8b-instruct --ratio 30
```

This verifies four linked objects:

1. `registry.json` fixes the ordered unit inventory and parameter cost;
2. `H.npy` fixes the diagonal and pairwise interaction scores;
3. `prune_solution.json` fixes the target, guard, and interaction strength;
4. the public solver must reproduce the exact evaluated unit set.

CI checks representative LLM and VLM points. Before a release, `make
verify-paper-masks` checks all 42 points.

## Level C: reproduce end to end

### LLM protocol

- calibration: 128 C4 sequences, 2048 tokens, seed 42;
- interaction selection: 16 disjoint C4 sequences, measured teacher KL;
- units: KV-aligned attention groups and 16 FFN groups per layer;
- budget: removable parameter count;
- guard: first and last two layers protected, per-layer cap `1.5 * rho`;
- teacher support for H: top 256 tokens;
- evaluation: context 2048, perplexity stride 512.

Run the stages in order:

```bash
python scripts/build_calibration_mix.py --project-root . --source c4 \
  --no-fallback --num-samples 128 --holdout-samples 16 --seq-len 2048
python scripts/run_collect_full.py --config configs/paper.yaml \
  --model-key llama-3.1-8b-instruct --run-dir outputs/end-to-end
python scripts/run_unit_ablations.py --config configs/paper.yaml \
  --model-key llama-3.1-8b-instruct --run-dir outputs/end-to-end
python scripts/run_build_H.py --config configs/paper.yaml \
  --model-key llama-3.1-8b-instruct --run-dir outputs/end-to-end
python scripts/select_lambda.py --config configs/paper.yaml \
  --model-key llama-3.1-8b-instruct \
  --h outputs/end-to-end/matrices/H.npy \
  --holdout data/calibration/calibration_holdout.jsonl \
  --ratio 0.30 --output outputs/end-to-end/selection
```

The selected mask can then be evaluated with `run_standard_eval.py`.

### VLM protocol

- calibration: 128 Flickr30k image-caption pairs, seed 0;
- interaction selection: 48 disjoint pairs;
- teacher support: top 64 tokens;
- units: both language and vision towers, one shared parameter budget;
- guard: first and last two layers of each tower protected;
- evaluation: 1,000 items each on MMBench, SEED-Bench, ScienceQA, AI2D,
  MMStar, POPE, and MME.

Use `vlm_build_h.py` followed by `run_vlm.py` as shown in the README.

## Recovery

The LLM recipe uses cleaned Alpaca, rank 16, alpha 32, dropout 0, 1024-token
sequences, batch 2, accumulation 8, one epoch, and peak learning rate `1e-4`.
All attention and FFN projections are targeted. The compact adapter is saved by
default.

The VLM recipe uses 3,000 mixed visual-instruction examples, rank 128, alpha
256, global batch 16, language-tower learning rate `2.5e-5`, and projector
learning rate `2e-5`; the vision encoder is frozen.

## Quantization

`run_standard_eval.py` supports:

- `--load-in-4bit`: NF4, double quantization, bf16 compute;
- `--load-in-8bit`: LLM.int8;
- `--bf16`: explicitly disable a model registry's quantization default.

Quantization changes numerical weights but never changes the released structural
mask.

## Efficiency

Use `run_efficiency_eval.py --mode physical_slice` for the bf16 LLM sweep,
`run_deployment_eval.py` for the sliced bf16/INT8/NF4 cells, and
`run_vlm_efficiency.py` for the two-tower VLM measurements. Reported latency,
throughput, and peak memory require an isolated GPU. Dense and pruned runs must
use the same GPU model, software environment, prompts, sequence length, batch,
warm-up count, and repeat count. Runtime masks and zeroed tensors do not measure
structural speedup; only `physical_slice` does.

## Artifact contract

- Unit IDs are meaningful only under the colocated `registry.json`.
- A mask is valid only for its exact model ID and registry.
- `actual_prune_ratio` is recomputed from registry costs during release checks.
- `reference_metrics.json` is a compact regression target, not an input.
- `manifest.json` contains byte sizes and SHA256 checksums for every data file.
- `H.npy` is tracked with Git LFS. A tiny pointer file indicates an incomplete
  clone; run `git lfs pull`.

## Expected variation

Mask reconstruction is deterministic and must be exact. Quality evaluation can
show small platform-level floating-point differences. LoRA and efficiency runs
have additional stochastic and hardware variation; preserve seeds and report the
actual environment saved in the output metadata.

## Troubleshooting

- **Artifact verification fails:** run `git lfs pull` and retry.
- **Gated model fails to download:** accept its license and authenticate with
  Hugging Face.
- **70B runs out of memory:** use the paper's NF4 setting and a device with
  sufficient free memory.
- **VLM processor import fails:** install `pip install -e '.[vlm]'` and use the
  Transformers version in `requirements-tested.txt`.
- **Efficiency is unstable:** remove other GPU processes and rerun dense and
  sliced measurements in the same session.
