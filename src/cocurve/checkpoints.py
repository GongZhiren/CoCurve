"""Portable loading of dense checkpoints with heterogeneous per-layer shapes."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import torch
from torch import nn

from .prune import ResidualConstantFFN

_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
                "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def inspect_checkpoint(directory):
    """Check every shard's header, byte ranges and index without loading weights."""
    root = Path(directory)
    index = root / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    files = sorted(set(mapping.values())) if mapping else ["model.safetensors"]
    tensors, sizes = {}, {}
    for name in files:
        if Path(name).name != name:
            raise ValueError("Checkpoint shard names must not contain paths")
        path = root / name
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Incomplete safetensors header: {name}")
            length = struct.unpack("<Q", prefix)[0]
            if length > 100_000_000 or length + 8 > size:
                raise ValueError(f"Invalid safetensors header length: {name}")
            header = json.loads(stream.read(length))
        end, ranges = 0, []
        for key, row in header.items():
            if key == "__metadata__":
                continue
            if key in tensors or (mapping is not None and mapping.get(key) != name):
                raise ValueError(f"Tensor/index mismatch: {key}")
            start, stop = row["data_offsets"]
            if not 0 <= start <= stop <= size - length - 8:
                raise ValueError(f"Invalid tensor byte range: {key}")
            if row["dtype"] not in _DTYPE_BYTES:
                raise ValueError(f"Unsupported tensor dtype: {row['dtype']}")
            if math.prod(row["shape"]) * _DTYPE_BYTES[row["dtype"]] != stop - start:
                raise ValueError(f"Tensor shape/size mismatch: {key}")
            tensors[key] = {"shape": row["shape"], "dtype": row["dtype"], "file": name,
                            "data_offsets": row["data_offsets"], "data_start": length + 8}
            ranges.append((start, stop))
            end = max(end, stop)
        cursor = 0
        for start, stop in sorted(ranges):
            if start != cursor:
                raise ValueError(f"Overlapping or missing tensor bytes: {name}")
            cursor = stop
        if end != size - length - 8:
            raise ValueError(f"Incomplete or trailing checkpoint bytes: {name}")
        sizes[name] = size
    if mapping is not None and set(mapping) != set(tensors):
        raise ValueError("Index lists tensors absent from the weight files")
    return {"tensors": tensors, "files": sizes, "total_bytes": sum(sizes.values())}


def _resize_model(model, tensors):
    """Recreate the stored matrix dimensions before assigning any weights."""
    for name, module in list(model.named_modules()):
        key = name + ".weight"
        if isinstance(module, nn.Linear) and key in tensors:
            shape = tensors[key]["shape"]
            if len(shape) != 2 or min(shape) < 1:
                raise ValueError(f"Invalid linear shape: {key}")
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            setattr(parent, attr, nn.Linear(shape[1], shape[0], bias=name + ".bias" in tensors,
                                           device="meta", dtype=module.weight.dtype))
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Checkpoint loading currently supports Llama/Mistral-style decoder stacks")
    for index, layer in enumerate(layers):
        attn = layer.self_attn
        q_heads = attn.q_proj.out_features // attn.head_dim
        kv_heads = attn.k_proj.out_features // attn.head_dim
        if q_heads < 1 or kv_heads < 1 or q_heads % kv_heads:
            raise ValueError("Invalid stored attention grouping")
        for attr, value in (("num_key_value_groups", q_heads // kv_heads),
                            ("num_heads", q_heads), ("num_attention_heads", q_heads),
                            ("num_key_value_heads", kv_heads)):
            if hasattr(attn, attr):
                setattr(attn, attr, value)
        prefix = f"model.layers.{index}.mlp."
        if not any(key.startswith(prefix) and key != prefix + "bias" for key in tensors):
            bias = torch.empty(model.config.hidden_size, device="meta") if prefix + "bias" in tensors else None
            layer.mlp = ResidualConstantFFN(bias)
        elif hasattr(layer.mlp, "intermediate_size"):
            layer.mlp.intermediate_size = layer.mlp.down_proj.in_features


def load_checkpoint(directory, *, device="cpu", dtype=None):
    """Load a sliced floating-point checkpoint without allocating its dense base.

    Returns a model in eval mode. For tokenization, use AutoTokenizer on the
    same directory. Packed low-bit checkpoints require their quantizer's loader.
    """
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    root = Path(directory)
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if getattr(config, "quantization_config", None):
        raise ValueError("Use the original quantizer to load packed low-bit weights")
    info = verify_checkpoint(root)
    if dtype is None:
        kinds = {row["dtype"] for row in info["tensors"].values()}
        dtype = torch.bfloat16 if "BF16" in kinds else torch.float16 if "F16" in kinds else torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=False, torch_dtype=dtype, attn_implementation="eager")
        _resize_model(model, info["tensors"])
    model.tie_weights()
    expected = set(model.state_dict())
    actual = set(info["tensors"])
    missing = expected - actual
    if config.tie_word_embeddings:
        missing -= {"lm_head.weight"}
    if missing or actual - expected:
        raise ValueError(f"Checkpoint state mismatch: {len(missing)} missing, {len(actual - expected)} extra")
    model = load_checkpoint_and_dispatch(model, str(root), device_map={"": str(device)},
                                         dtype=dtype, strict=True)
    model.tie_weights()
    if (root / "generation_config.json").is_file():
        model.generation_config = GenerationConfig.from_pretrained(root, local_files_only=True)
    return model.eval()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(directory):
    """Validate tensor layout and check every hash if a release manifest exists."""
    root = Path(directory)
    info = inspect_checkpoint(root)
    manifest = root / "checkpoint_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        if payload.get("format") != "cocurve-sliced-safetensors-v1":
            raise ValueError("Unknown checkpoint manifest format")
        if not set(info["files"]) <= set(payload["files"]):
            raise ValueError("Manifest does not cover all checkpoint shards")
        for name, record in payload["files"].items():
            if Path(name).name != name:
                raise ValueError("Manifest file names must not contain paths")
            path = root / name
            if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return info


def save_checkpoint(model, directory, *, tokenizer=None, max_shard_size="1900MB"):
    """Write safetensors shards suitable for GitHub Releases as well as HF."""
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use an empty output directory to avoid stale checkpoint shards")
    root.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(root, safe_serialization=True, max_shard_size=max_shard_size)
    if tokenizer is not None:
        tokenizer.save_pretrained(root)
    for name in ("config.json", "tokenizer_config.json"):
        path = root / name
        if path.exists():
            data = json.loads(path.read_text())
            for key in ("_name_or_path", "name_or_path"):
                data.pop(key, None)
            path.write_text(json.dumps(data, indent=2) + "\n")
    inspect_checkpoint(root)
    manifest = {"format": "cocurve-sliced-safetensors-v1",
                "loader": "cocurve.checkpoints.load_checkpoint",
                "files": {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                          for path in sorted(root.iterdir()) if path.is_file()}}
    (root / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_checkpoint(source, destination, *, max_shard_bytes=1_900_000_000):
    """Re-shard an existing checkpoint by copying tensor bytes, without a model load.

    This preserves every weight bit. It copies only model/tokenizer files,
    excluding training logs, local paths, and experiment directories.
    """
    source, target = Path(source), Path(destination)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Export destination must be empty")
    info = verify_checkpoint(source)
    groups, group, used = [], [], 0
    for name, row in sorted(info["tensors"].items()):
        size = row["data_offsets"][1] - row["data_offsets"][0]
        if size + 100_000 > max_shard_bytes:
            raise ValueError(f"Tensor {name} cannot fit the requested shard size")
        if group and used + size + 100_000 > max_shard_bytes:
            groups.append(group)
            group, used = [], 0
        group.append((name, row))
        used += size
    if group:
        groups.append(group)
    target.mkdir(parents=True, exist_ok=True)
    weight_map, records = {}, {}
    for index, group in enumerate(groups, 1):
        filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
        header, offset = {"__metadata__": {"format": "pt"}}, 0
        for name, row in group:
            size = row["data_offsets"][1] - row["data_offsets"][0]
            header[name] = {"dtype": row["dtype"], "shape": row["shape"],
                            "data_offsets": [offset, offset + size]}
            offset += size
            weight_map[name] = filename
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * ((-len(raw)) % 8)
        prefix = struct.pack("<Q", len(raw)) + raw
        sha = hashlib.sha256(prefix)
        with (target / filename).open("wb") as out:
            out.write(prefix)
            for _, row in group:
                start, end = row["data_offsets"]
                remaining = end - start
                with (source / row["file"]).open("rb") as stream:
                    stream.seek(row["data_start"] + start)
                    while remaining:
                        block = stream.read(min(8 * 1024**2, remaining))
                        if not block:
                            raise ValueError("Source checkpoint changed or became incomplete")
                        out.write(block)
                        sha.update(block)
                        remaining -= len(block)
        records[filename] = {"bytes": len(prefix) + offset, "sha256": sha.hexdigest()}
    (target / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(
            math.prod(row["shape"]) * _DTYPE_BYTES[row["dtype"]]
            for row in info["tensors"].values())}, "weight_map": weight_map}, indent=2) + "\n")
    portable_files = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "added_tokens.json", "tokenizer.model",
                      "vocab.json", "merges.txt", "chat_template.jinja", "LICENSE", "NOTICE",
                      "USE_POLICY.md"}
    for name in sorted(portable_files):
        path = source / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if name in {"config.json", "tokenizer_config.json"}:
            data = json.loads(raw)
            for key in ("_name_or_path", "name_or_path", "tokenizer_file"):
                data.pop(key, None)
            raw = (json.dumps(data, indent=2) + "\n").encode()
        (target / name).write_bytes(raw)
    for path in sorted(target.iterdir()):
        if path.name not in records:
            records[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    exported = inspect_checkpoint(target)
    for name, row in info["tensors"].items():
        if any(exported["tensors"][name][key] != row[key] for key in ("shape", "dtype")):
            raise ValueError("Exported tensor layout differs from source")
    manifest = {"format": "cocurve-sliced-safetensors-v1",
                "loader": "cocurve.checkpoints.load_checkpoint",
                "tensor_bytes_preserved": True, "files": records}
    (target / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
