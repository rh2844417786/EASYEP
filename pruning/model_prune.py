#!/usr/bin/env python3
"""Physically prune per-expert Hugging Face safetensors checkpoints.

The script is architecture-derived rather than hard-coded to layers 3..60.
DeepSeek-V4 hash-routed layers are rejected intentionally: removing an expert
without remapping every token-to-expert entry would create an invalid model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import shutil
from typing import Any

import torch
from tqdm import tqdm


EXPERT_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)(?:\..+)$")


def parse_expert_key(key: str) -> tuple[int, int] | None:
    match = EXPERT_KEY.match(key)
    return (int(match.group(1)), int(match.group(2))) if match else None


def load_mask(path: Path) -> torch.Tensor:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    mask = torch.tensor(value, dtype=torch.float32)
    if mask.ndim != 2 or not mask.numel():
        raise ValueError(f"mask must be a non-empty 2D array, got shape={tuple(mask.shape)}")
    if not torch.isfinite(mask).all():
        raise ValueError("mask contains NaN/Inf")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only 0/1 values")
    return mask.bool()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def inspect_layout(weight_map: dict[str, str]) -> tuple[list[int], dict[int, set[int]]]:
    experts: dict[int, set[int]] = defaultdict(set)
    for key in weight_map:
        parsed = parse_expert_key(key)
        if parsed:
            layer, expert = parsed
            experts[layer].add(expert)
    if not experts:
        stacked = [key for key in weight_map if ".mlp.experts." in key]
        detail = f" Examples: {stacked[:3]}" if stacked else ""
        raise ValueError(
            "checkpoint has no per-expert keys matching "
            "model.layers.<L>.mlp.experts.<E>.*; stacked expert tensors are unsupported." + detail
        )
    return sorted(experts), experts


def validate_v4_hash_routing(config: dict[str, Any], weight_map: dict[str, str]) -> None:
    num_hash_layers = int(config.get("num_hash_layers", config.get("n_hash_layers", 0)) or 0)
    mapping_keys = [key for key in weight_map if "tid2eid" in key or "tie2eid" in key]
    if num_hash_layers > 0 or mapping_keys:
        raise ValueError(
            "physical pruning is blocked for this DeepSeek-V4 checkpoint: it has "
            f"{num_hash_layers} hash-routed layers and token-to-expert tables. A global "
            "n_routed_experts reduction would leave token mappings pointing at deleted experts. "
            "Implement and validate a layer-specific hash-routing remap before pruning weights; "
            "the V3/R1 renumbering used by EASY-EP is not sufficient."
        )


def build_renames(
    weight_map: dict[str, str],
    mask: torch.Tensor,
    expert_layers: list[int],
    experts_by_layer: dict[int, set[int]],
) -> tuple[dict[str, str], dict[str, str], int]:
    if mask.shape[0] != len(expert_layers):
        raise ValueError(
            f"mask has {mask.shape[0]} rows but checkpoint has {len(expert_layers)} expert layers: "
            f"{expert_layers}"
        )
    observed_experts = max(max(values) for values in experts_by_layer.values()) + 1
    if mask.shape[1] != observed_experts:
        raise ValueError(
            f"mask has {mask.shape[1]} columns but checkpoint expert ids imply {observed_experts} experts"
        )
    expected_ids = set(range(observed_experts))
    for layer, ids in experts_by_layer.items():
        if ids != expected_ids:
            missing = sorted(expected_ids - ids)
            raise ValueError(f"layer {layer} does not contain a complete expert set; missing={missing[:10]}")

    selected_by_layer: dict[int, list[int]] = {}
    for row, layer in enumerate(expert_layers):
        selected_by_layer[layer] = torch.where(mask[row])[0].tolist()
    keep_counts = {len(selected) for selected in selected_by_layer.values()}
    if len(keep_counts) != 1 or 0 in keep_counts:
        raise ValueError(
            "all layers must keep the same positive number of experts because config.json has one "
            f"global n_routed_experts; counts={sorted(keep_counts)}"
        )
    kept_experts = next(iter(keep_counts))
    new_ids = {
        layer: {old_id: new_id for new_id, old_id in enumerate(selected)}
        for layer, selected in selected_by_layer.items()
    }

    filtered: dict[str, str] = {}
    renames: dict[str, str] = {}
    seen_renamed: set[str] = set()
    for key, filename in weight_map.items():
        parsed = parse_expert_key(key)
        if parsed is None:
            filtered[key] = filename
            renames[key] = key
            seen_renamed.add(key)
            continue
        layer, expert = parsed
        if expert not in new_ids[layer]:
            continue
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        renamed = key.replace(
            prefix,
            f"model.layers.{layer}.mlp.experts.{new_ids[layer][expert]}",
            1,
        )
        if renamed in seen_renamed:
            raise ValueError(f"renaming collision for {renamed}")
        filtered[key] = filename
        renames[key] = renamed
        seen_renamed.add(renamed)
    return filtered, renames, kept_experts


def ensure_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def copy_auxiliary_files(input_dir: Path, output_dir: Path) -> None:
    for source in input_dir.iterdir():
        if not source.is_file() or source.suffix == ".safetensors":
            continue
        if source.name in {"model.safetensors.index.json", "filtered.json"}:
            continue
        shutil.copy2(source, output_dir / source.name)


def prune_checkpoint(args: argparse.Namespace) -> None:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError(
            "model pruning requires safetensors (`pip install safetensors`)"
        ) from exc
    index_path = args.input_dir / "model.safetensors.index.json"
    config_path = args.input_dir / "config.json"
    if not index_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("input_dir must contain model.safetensors.index.json and config.json")

    index = load_json(index_path)
    config = load_json(config_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid weight_map in {index_path}")
    validate_v4_hash_routing(config, weight_map)
    expert_layers, experts_by_layer = inspect_layout(weight_map)
    mask = load_mask(args.mask_json)
    filtered, renames, kept_experts = build_renames(
        weight_map, mask, expert_layers, experts_by_layer
    )

    print(
        f"expert layers={expert_layers}; experts {mask.shape[1]} -> {kept_experts}; "
        f"weight tensors {len(weight_map)} -> {len(filtered)}"
    )
    if args.dry_run:
        return

    ensure_output_directory(args.output_dir)
    output_weight_map = {renames[key]: filename for key, filename in filtered.items()}
    tensor_bytes = 0
    written_keys: set[str] = set()
    input_files = sorted(set(weight_map.values()))
    for filename in tqdm(input_files, desc="prune safetensors"):
        source = args.input_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"index references missing shard: {source}")
        state = load_file(str(source), device="cpu")
        output_state = {
            renames[key]: tensor
            for key, tensor in state.items()
            if key in filtered
        }
        expected_keys = {key for key, shard in filtered.items() if shard == filename}
        missing_keys = expected_keys - set(state)
        if missing_keys:
            raise ValueError(f"{filename} is missing indexed tensors: {sorted(missing_keys)[:5]}")
        if output_state:
            save_file(output_state, str(args.output_dir / filename))
            tensor_bytes += sum(tensor.numel() * tensor.element_size() for tensor in output_state.values())
            written_keys.update(output_state)

    missing_output_keys = set(output_weight_map) - written_keys
    if missing_output_keys:
        raise ValueError(
            f"output checkpoint did not write indexed tensors: {sorted(missing_output_keys)[:5]}"
        )

    metadata = dict(index.get("metadata") or {})
    metadata["total_size"] = tensor_bytes
    output_index = {"metadata": metadata, "weight_map": output_weight_map}
    with (args.output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(output_index, handle, indent=2)

    copy_auxiliary_files(args.input_dir, args.output_dir)
    output_config = dict(config)
    if "n_routed_experts" not in output_config:
        raise ValueError("config.json has no n_routed_experts field")
    output_config["n_routed_experts"] = kept_experts
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(output_config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"pruned checkpoint saved to {args.output_dir}; config n_routed_experts={kept_experts}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune per-expert Hugging Face safetensors")
    parser.add_argument("--mask-json", "--mask_json", dest="mask_json", type=Path, required=True)
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate and print the pruning plan only")
    return parser


def main() -> int:
    prune_checkpoint(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
