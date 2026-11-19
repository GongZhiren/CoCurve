# Reconstruction and reusable checkpoints

## Two recovery paths

CoCurve selects structures. Reconstruction is optional and uses independent
calibration text:

| Method | Attention groups | Per-layer FFN count | FFN identities |
|---|---|---|---|
| `obs` | Fixed | Fixed | Fixed |
| `ffn_refinement` | Fixed | Fixed | Refined |

OBS compensates retained output-projection weights using the constrained
least-squares solution. FFN refinement uses iterative block OBS within each
FFN, following SlimGPT (Ling et al., NeurIPS 2024). It does not replace CoCurve's
attention selection or per-layer allocation. The numerical core is implemented
locally; no baseline checkout is required.

The defaults match the complete-pipeline experiments: 256 fresh C4 calibration
windows of 2,048 tokens and 1% damping. Supply a JSONL file with a `text` field
whose documents are disjoint from the graph calibration, held-out selection,
and evaluation data. The command packs text continuously, without padding.
Load the base on CPU; one decoder layer at a time moves to the selected device.

For a new calibration draw, the existing materializer can write a separate file:

```bash
python scripts/build_calibration_mix.py --source c4 --no-fallback \
  --seed 20261107 --num-samples 288 --holdout-samples 0 \
  --output data/calibration/reconstruction.jsonl \
  --holdout-output data/calibration/reconstruction_unused.jsonl \
  --manifest data/calibration/reconstruction_manifest.json \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

Keep this draw separate from graph/selection data. The extra windows provide
room for retokenization when the recovery command packs its 256 inputs.

```bash
python scripts/run_reconstruction.py \
  --model-key llama-3.1-8b-instruct \
  --mask artifacts/paper-v1/llm/llama-3.1-8b-instruct/r30/masks/prune_solution.json \
  --calibration data/calibration/reconstruction.jsonl \
  --method ffn_refinement \
  --device cuda:0 --output outputs/llama8b-refined-r30
```

The result contains safetensors weights, tokenizer, portable reconstruction
metadata, and the actual channel-level mask. That mask has one unit per FFN
channel, rather than the original 16 groups per layer. Its
`ffn_groups_per_layer` field records this distinction.

## Load without reconstructing again

Different layers can retain different head and channel counts. A stock dense
Hugging Face config alone does not encode those shapes. The loader below checks
the safetensors headers and shard index, rebuilds the stored projection shapes
on the meta device, and then loads the weights. When a release manifest is
present, it also verifies the recorded file hashes:

```python
from cocurve.checkpoints import load_checkpoint
from transformers import AutoTokenizer

path = "outputs/llama8b-refined-r30"
model = load_checkpoint(path, device="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
inputs = tokenizer("Explain structured pruning briefly.", return_tensors="pt").to("cuda:0")
tokens = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

The loader supports floating-point Llama/Mistral-style decoder checkpoints,
including fully removed FFNs. Packed low-bit weights use their quantizer's
loading route instead. Quantization and structural deletion are separate steps.

## Continue with common LoRA

```bash
python scripts/run_lora_recovery.py \
  --model-key llama-3.1-8b-instruct \
  --reconstructed-checkpoint outputs/llama8b-refined-r30 \
  --epochs 1 --out outputs/llama8b-refined-r30-lora
```

Do not also specify `--mask-run-dir`: the input model is already reconstructed
and sliced. The default saves a compact adapter. To reload it, first load the
same reconstructed checkpoint and attach the adapter with
`peft.PeftModel.from_pretrained`. An adapter depends on the exact reconstructed
weights and structure, not merely the dense base model ID.

Use `--save-merged` to save the full recovered checkpoint. The saver uses
1.9 GB shards, checks their headers, and records SHA256 checksums. Such shards
fit both Hugging Face repositories and GitHub Release assets. Source code and
model weights can be hosted independently.

To repackage an older sliced checkpoint without loading or changing its
weights, use:

```bash
python scripts/export_checkpoint.py --source CHECKPOINT_DIRECTORY \
  --output outputs/checkpoint-export
```

## Download status and verification

The 42 original paper masks and nine curvature matrices are already included
in `artifacts/paper-v1`. These allow direct mask evaluation or selection replay.
The new reconstruction and checkpoint-loading interfaces are included here.

Full model checkpoint downloads are being prepared separately; this repository
does not yet advertise a verified weight download. Checkpoint publication
requires complete shards, the exact source mask and recovery protocol, and a
prediction check against the accepted evaluation record. A result JSON alone
is not a downloadable model.

In particular, older C4-recovered checkpoints are distinct from the paper's
Alpaca-recovered checkpoints, even when the model and pruning ratio match.
Models and adapters retain the licenses of their base models. The MIT license
for this repository applies to the code.
