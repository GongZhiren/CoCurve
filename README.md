# CoCurve: Cross-Module Co-Pruning Curvature for Structured LLM Pruning

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2607.17568)
[![Project Page](https://img.shields.io/badge/Project-Page-1a56ad.svg)](https://gongzhiren.github.io/CoCurve-website/)
[![CI](https://github.com/GongZhiren/CoCurve/actions/workflows/ci.yml/badge.svg)](https://github.com/GongZhiren/CoCurve/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Prune the edges, not just the nodes.**

CoCurve performs structured pruning over one shared inventory of attention and
feed-forward units. A label-free second-order expansion of teacher KL yields a
Fisher Gram matrix whose diagonal scores individual units and whose signed
off-diagonal entries measure their co-pruning interactions. CoCurve then selects
the interaction strength on held-out calibration text and solves one shared,
parameter-budgeted pruning problem.

This repository provides:

- the complete LLM and VLM CoCurve implementation;
- publication-locked `H.npy`, unit registries, masks, and reference metrics;
- a fast evaluation path that does not recompute curvature;
- full end-to-end curvature construction and mask selection;
- the paper's lightweight LoRA, NF4/INT8, physical slicing, and efficiency paths.

Baseline implementations, plotting code, interpretability analyses, and
review-only diagnostics are intentionally outside this release.

![CoCurve overview](assets/overview.png)

## News

- **2026-09-22 — paper reproduction release.** Added the complete LLM/VLM
  pipeline, all publication masks and curvature matrices, reference metrics,
  exact mask verification, recovery and quantization recipes, packaging, tests,
  and CI.
- **2026-07 — initial public release.** Released the core LLM pipeline.

## Installation

Git LFS is required because the repository includes the paper's curvature
matrices (about 55 MiB in total).

```bash
git lfs install
git clone https://github.com/GongZhiren/CoCurve.git
cd CoCurve
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install only the optional components you use:

```bash
pip install -e '.[recovery]'      # LoRA
pip install -e '.[quantization]'  # NF4 / INT8
pip install -e '.[vlm]'           # vision-language models
pip install -e '.[all]'           # everything above
```

Some paper models are gated on Hugging Face. Authenticate with
`huggingface-cli login` after accepting their model licenses.

## Five-minute reproduction

### 1. Verify the release

```bash
cocurve-artifacts list
cocurve-artifacts verify
```

Every file is size- and SHA256-checked against
[`manifest.json`](artifacts/paper-v1/manifest.json).

### 2. Reproduce a paper mask from released curvature

This requires no model weights, dataset download, or GPU:

```bash
python scripts/reproduce_paper_mask.py \
  --model llama-3.1-8b-instruct --ratio 30
```

The command reconstructs the mask from the released `H.npy`, unit registry,
budget, guard, and interaction strength, then requires exact agreement with the
publication mask.

### 3. Evaluate the released mask

```bash
python scripts/evaluate_paper_mask.py \
  --model llama-3.1-8b-instruct --ratio 30 \
  --tasks wikitext
```

Use `--tasks all` for the paper's three perplexity corpora and 12 downstream
tasks. `--precision nf4` and `--precision int8` reproduce the low-bit
composition path. Precision defaults to the paper setting (`NF4` for 70B,
`bf16` otherwise).

Before `--tasks all`, materialize the five paper datasets that are evaluated
from fixed local JSONL snapshots:

```bash
python scripts/prepare_paper_datasets.py
```

## Released paper artifacts

Artifacts live under `artifacts/paper-v1/{llm,vlm}/<model>/`.

| Family | Model key | Ratios | Paper precision |
|---|---|---:|---|
| LLM | `llama-3.2-3b` | 10–50% | bf16 |
| LLM | `falcon3-7b` | 10–50% | bf16 |
| LLM | `llama-3.1-8b-instruct` | 10–50% | bf16 |
| LLM | `mistral-nemo-12b` | 10–50% | bf16 |
| LLM | `mistral-small-24b` | 10–50% | bf16 |
| LLM | `llama-3.1-70b-instruct` | 10–50% | NF4 |
| VLM | `qwen2.5-vl-7b` | 20–50% | bf16 |
| VLM | `qwen3-vl-8b` | 20–50% | bf16 |
| VLM | `internvl3-8b` | 20–50% | bf16 |

Each model directory contains:

```text
registry.json                       # ordered unit inventory and parameter costs
matrices/H.npy                      # publication curvature matrix
rXX/masks/prune_solution.json       # exact evaluated mask
rXX/reference_metrics.json          # compact paper-facing measurements
```

The mask is the primary provenance object: it is the exact set evaluated in the
paper. The registry fixes every unit ID, and the manifest binds all files by
checksum. For two boundary-adjacent 24B path solutions, the mask additionally
stores an interior path representative used only for cross-version bit-exact
solver verification; the selected calibration value is retained separately.

## Full LLM pipeline

### 1. Build the paper calibration split

```bash
python scripts/build_calibration_mix.py \
  --project-root . --source c4 --no-fallback \
  --num-samples 128 --holdout-samples 16 --seq-len 2048 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

### 2. Construct the co-pruning curvature matrix

```bash
RUN=outputs/llama8b
CFG=configs/paper.yaml

python scripts/run_collect_full.py \
  --config $CFG --model-key llama-3.1-8b-instruct --run-dir $RUN
python scripts/run_unit_ablations.py \
  --config $CFG --model-key llama-3.1-8b-instruct --run-dir $RUN
python scripts/run_build_H.py \
  --config $CFG --model-key llama-3.1-8b-instruct --run-dir $RUN
```

The estimator uses 128 C4 sequences of length 2048, top-256 teacher support,
16 FFN groups per layer, and KV-aligned attention groups.

### 3. Select interaction strength and solve

```bash
python scripts/select_lambda.py \
  --config $CFG --model-key llama-3.1-8b-instruct \
  --h $RUN/matrices/H.npy \
  --holdout data/calibration/calibration_holdout.jsonl \
  --ratio 0.30 --output $RUN/selection
```

The script exactly enumerates the critical-value path over `lambda in [0,1]`,
measures each distinct mask by held-out teacher KL, and writes both the selected
mask and its path metadata. It does not tune on any downstream benchmark.

To run the original staged pipeline with a known interaction strength, set it in
`configs/pruning.yaml` and use `scripts/run_greedy_prune.py`.

### 4. Evaluate, recover, or slice

```bash
# Full paper evaluation
python scripts/run_standard_eval.py \
  --config $CFG --model-key llama-3.1-8b-instruct \
  --run-dir $RUN/eval --mask-run-dir $RUN/selection --tasks all

# Paper LoRA recipe; compact adapter saved by default
python scripts/run_lora_recovery.py \
  --config $CFG --model-key llama-3.1-8b-instruct \
  --mask-run-dir $RUN/selection --out $RUN/recovery

# True sliced-matrix efficiency measurement; use an otherwise idle GPU
python scripts/run_efficiency_eval.py \
  --config $CFG --model-key llama-3.1-8b-instruct \
  --run-dir $RUN/efficiency --mask-run-dir $RUN/selection \
  --mode physical_slice
```

Efficiency numbers are hardware-sensitive. Measure dense and sliced checkpoints
on the same isolated GPU, software stack, sequence length, batch, warm-up,
and repeat count.

## Vision-language pipeline

Build the two-tower matrix over 128 Flickr30k image-caption pairs:

```bash
python scripts/vlm_build_h.py \
  Qwen/Qwen2.5-VL-7B-Instruct outputs/qwen25vl/H \
  --n 128 --top-r 64 --ffn-groups 16
```

Select a mask and evaluate the seven paper benchmarks:

```bash
python scripts/run_vlm.py \
  Qwen/Qwen2.5-VL-7B-Instruct outputs/qwen25vl/H \
  outputs/qwen25vl/r30.json --ratio 0.30 --limit 1000
```

Or skip selection and evaluate a released mask:

```bash
python scripts/run_vlm.py \
  Qwen/Qwen2.5-VL-7B-Instruct \
  artifacts/paper-v1/vlm/qwen2.5-vl-7b/matrices \
  outputs/qwen25vl/released-r30.json --ratio 0.30 \
  --mask artifacts/paper-v1/vlm/qwen2.5-vl-7b/r30/masks/prune_solution.json
```

The VLM recovery script uses the paper's common recipe: rank 128, alpha 256,
3,000 mixed visual-instruction examples, global batch 16, frozen vision tower,
and a separately trained multimodal projector.

```bash
python scripts/recover_vlm.py \
  Qwen/Qwen2.5-VL-7B-Instruct \
  artifacts/paper-v1/vlm/qwen2.5-vl-7b/r30/masks/prune_solution.json \
  outputs/qwen25vl/recovery-r30
```

## Repository layout

```text
src/cocurve/       core LLM implementation
src/cocurve/vlm/   VLM inventory, masks, curvature, evaluation, recovery
scripts/           stage entry points and paper reproduction commands
configs/           portable paper configuration
artifacts/paper-v1 publication-locked H, registries, masks, and metrics
tests/             artifact, solver, and release smoke tests
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete protocol,
artifact contract, expected outputs, and troubleshooting.

## Citation

```bibtex
@article{gong2026cocurve,
  title   = {CoCurve: Cross-Module Co-Pruning Curvature for Structured LLM Pruning},
  author  = {Gong, Zhiren and Zeng, Zihao and Wang, Tiantong and Anada, Honoka and
             Wang, Yixin and Wang, Zijie and Xiao, Ming and Yuen, Chau and
             Lim, Wei Yang Bryan},
  journal = {arXiv preprint arXiv:2607.17568},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE). Model weights and datasets retain
their original licenses and are not redistributed here.
