#!/usr/bin/env python3
"""Linearly interpolate SFT and RL Hugging Face checkpoints.

The merge is defined as::

    theta_merged = alpha * theta_sft + (1 - alpha) * theta_rl

Both inputs must use the same architecture and safetensors parameter names.
Tensors are processed shard by shard, and the output shard layout and metadata
follow the RL checkpoint. Gemma 3 tied embeddings are written without a
separate ``lm_head.weight`` tensor.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INDEX_FILENAME = "model.safetensors.index.json"
SINGLE_SHARD_FILENAME = "model.safetensors"
MANIFEST_FILENAME = "merge_manifest.json"


def find_embedding_keys(keys: list[str]) -> tuple[str | None, str | None]:
    """Find the architecture-specific lm_head and embedding tensor names."""
    lm_head = next((key for key in keys if key.endswith("lm_head.weight")), None)
    embed = next((key for key in keys if key.endswith("model.embed_tokens.weight")), None)
    return lm_head, embed


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def build_reader(
    model_dir: str,
) -> tuple[Callable[[str], torch.Tensor | None], list[str]]:
    """Return a lazy tensor reader and the checkpoint's parameter names."""
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory not found: {root}")

    index_path = root / INDEX_FILENAME
    single_shard_path = root / SINGLE_SHARD_FILENAME
    if index_path.is_file():
        index = load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Missing weight_map in {index_path}")
    elif single_shard_path.is_file():
        with safe_open(single_shard_path, framework="pt") as handle:
            weight_map = {key: SINGLE_SHARD_FILENAME for key in handle.keys()}
    else:
        raise FileNotFoundError(
            f"No {INDEX_FILENAME} or {SINGLE_SHARD_FILENAME} found in {root}"
        )

    if not all(isinstance(key, str) and isinstance(shard, str) for key, shard in weight_map.items()):
        raise ValueError(f"Invalid weight_map entries in {root}")

    for shard in set(weight_map.values()):
        if not (root / shard).is_file():
            raise FileNotFoundError(f"Checkpoint shard not found: {root / shard}")

    handles: dict[str, Any] = {}

    def read(key: str) -> torch.Tensor | None:
        shard = weight_map.get(key)
        if shard is None:
            return None
        if shard not in handles:
            handles[shard] = safe_open(root / shard, framework="pt")
        return handles[shard].get_tensor(key)

    return read, list(weight_map)


def validate_alpha(alpha: float) -> None:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be finite and within [0, 1], got {alpha}")


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {out_dir}")
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}. Use --overwrite to replace generated files."
            )
        for path in out_dir.iterdir():
            if path.name in {INDEX_FILENAME, MANIFEST_FILENAME} or path.suffix == ".safetensors":
                path.unlink()
    else:
        out_dir.mkdir(parents=True)


def merge(
    model_a: str,
    model_b: str,
    alpha: float,
    out_dir: str,
    overwrite: bool = False,
) -> None:
    """Merge model A (SFT) and model B (RL) into ``out_dir``."""
    validate_alpha(alpha)
    sft_weight, rl_weight = alpha, 1.0 - alpha
    read_sft, sft_keys = build_reader(model_a)
    read_rl, rl_keys = build_reader(model_b)

    _, sft_embed = find_embedding_keys(sft_keys)
    rl_lm_head, rl_embed = find_embedding_keys(rl_keys)
    if sft_embed is None or rl_embed is None:
        raise KeyError("Could not find model.embed_tokens.weight in both models")

    rl_root = Path(model_b)
    index_path = rl_root / INDEX_FILENAME
    if index_path.is_file():
        rl_index = load_json(index_path)
        weight_map = rl_index["weight_map"]
        has_index = True
    else:
        rl_index = {"metadata": {}}
        weight_map = {key: SINGLE_SHARD_FILENAME for key in rl_keys}
        has_index = False

    output_keys = [key for key in rl_keys if key != rl_lm_head]
    sft_key_for_rl_key: dict[str, str] = {}
    for key in output_keys:
        if key in sft_keys:
            sft_key_for_rl_key[key] = key
        elif key == rl_embed:
            sft_key_for_rl_key[key] = sft_embed
        else:
            raise KeyError(f"RL tensor {key!r} is missing from the SFT checkpoint")

    shard_to_keys: dict[str, list[str]] = {}
    for key in output_keys:
        shard_to_keys.setdefault(weight_map[key], []).append(key)

    output_root = Path(out_dir)
    prepare_output_dir(output_root, overwrite)

    total_size = 0
    for shard, keys in shard_to_keys.items():
        merged_tensors: dict[str, torch.Tensor] = {}
        for key in keys:
            sft_tensor = read_sft(sft_key_for_rl_key[key])
            rl_tensor = read_rl(key)
            if sft_tensor is None or rl_tensor is None:
                raise KeyError(f"Failed to load tensor {key!r} from both checkpoints")
            if sft_tensor.shape != rl_tensor.shape:
                raise ValueError(
                    f"Tensor shape mismatch for {key}: SFT={tuple(sft_tensor.shape)}, "
                    f"RL={tuple(rl_tensor.shape)}"
                )

            merged = sft_weight * sft_tensor.float() + rl_weight * rl_tensor.float()
            merged_tensors[key] = merged.to(sft_tensor.dtype).contiguous()
            total_size += merged_tensors[key].numel() * merged_tensors[key].element_size()

        shard_path = output_root / shard
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(merged_tensors, shard_path, metadata={"format": "pt"})
        print(f"wrote {shard} ({len(keys)} tensors)")
        del merged_tensors
        gc.collect()

    if has_index:
        metadata = dict(rl_index.get("metadata", {}))
        metadata["total_size"] = total_size
        merged_index = {
            "metadata": metadata,
            "weight_map": {key: weight_map[key] for key in output_keys},
        }
        with (output_root / INDEX_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(merged_index, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    for source in rl_root.iterdir():
        if source.suffix == ".safetensors" or source.name in {INDEX_FILENAME, MANIFEST_FILENAME}:
            continue
        if source.is_file():
            shutil.copy2(source, output_root / source.name)

    config_path = output_root / "config.json"
    if config_path.is_file():
        config = load_json(config_path)
        config["tie_word_embeddings"] = True
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    manifest = {
        "merge_method": "linear_interpolation",
        "formula": "alpha * SFT + (1 - alpha) * RL",
        "alpha": alpha,
        "model_a": {"role": "SFT", "name": Path(model_a).name, "weight": sft_weight},
        "model_b": {"role": "RL", "name": Path(model_b).name, "weight": rl_weight},
        "output_layout": "model_b",
        "tie_word_embeddings": True,
    }
    with (output_root / MANIFEST_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"done: {output_root}")


def alpha_arg(value: str) -> float:
    alpha = float(value)
    try:
        validate_alpha(alpha)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return alpha


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge SFT and RL safetensors checkpoints by linear interpolation."
    )
    parser.add_argument("--model_a", required=True, help="SFT Hugging Face model directory (weight alpha)")
    parser.add_argument("--model_b", required=True, help="RL Hugging Face model directory (weight 1-alpha)")
    parser.add_argument("--alpha", required=True, type=alpha_arg, help="SFT weight in [0, 1]")
    parser.add_argument("--out", required=True, help="output Hugging Face model directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace generated weight, index, and manifest files in an existing output directory",
    )
    args = parser.parse_args()

    print(f"merge: {args.alpha} * SFT + {1.0 - args.alpha} * RL -> {args.out}")
    merge(args.model_a, args.model_b, args.alpha, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
