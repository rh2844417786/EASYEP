#!/usr/bin/env python3
"""Physically prune only the 40 dynamic-MoE layers of DeepSeek-V4-Flash.

Layers 0..2 are hash-routed. Their 256 experts, gate weights, and tid2eid tables
are preserved byte-for-byte. Layers 3..42 can be pruned to any uniform expert
count that still satisfies the model's per-token Top-K requirement, and their
retained expert weights are renumbered contiguously. Every router weight/bias
tensor remains complete; the runtime applies the EASY-EP mask and maps selected
router IDs to the compact physical expert IDs.

The output config keeps the global n_routed_experts=256 and stores the exact
per-layer EASY-EP mask for the patched SGLang v0.5.16 runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
from typing import Any, Iterable


EXPERT_KEY = re.compile(
    r"^(?P<root>(?:model\.)?layers)\.(?P<layer>\d+)\."
    r"(?P<block>ffn|mlp)\.experts\.(?P<expert>\d+)(?P<suffix>\..+)$"
)
GATE_KEY = re.compile(
    r"^(?P<root>(?:model\.)?layers)\.(?P<layer>\d+)\."
    r"(?P<block>ffn|mlp)\.gate\.(?P<field>weight|bias|tid2eid)$"
)
INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "easyep_pruning_manifest.json"
MARKER_NAME = ".easyep_pruning_incomplete.json"
MASK_COPY_NAME = "easyep_expert_mask.json"


@dataclass(frozen=True)
class V4Layout:
    num_layers: int
    hash_layers: int
    original_experts: int
    target_dynamic_experts: int
    counts_by_layer: tuple[int, ...]
    selected_by_layer: tuple[tuple[int, ...], ...]

    @property
    def dynamic_layers(self) -> int:
        return self.num_layers - self.hash_layers

    @property
    def dynamic_prune_fraction(self) -> float:
        return 1.0 - self.target_dynamic_experts / self.original_experts

    @property
    def main_layer_slot_prune_fraction(self) -> float:
        original = self.num_layers * self.original_experts
        retained = sum(self.counts_by_layer)
        return 1.0 - retained / original


@dataclass(frozen=True)
class PrunePlan:
    layout: V4Layout
    source_keys: int
    output_keys: int
    dropped_keys: int
    renamed_keys: int
    output_weight_map: dict[str, str]
    key_actions: dict[str, str | None]
    fingerprint: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def config_int(config: dict[str, Any], *names: str) -> int:
    for name in names:
        value = config.get(name)
        if value is not None:
            return int(value)
    raise ValueError(f"config is missing all of: {', '.join(names)}")


def load_mask(path: Path) -> list[list[int]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError(f"mask must be a non-empty 2D JSON array: {path}")
    rows: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            raise ValueError(f"mask row {row_index} is not an array")
        normalized: list[int] = []
        for column, item in enumerate(row):
            if item not in (0, 1, 0.0, 1.0, False, True):
                raise ValueError(
                    f"mask[{row_index}][{column}] must be binary, found {item!r}"
                )
            normalized.append(int(item))
        rows.append(normalized)
    return rows


def build_layout(
    config: dict[str, Any], mask: list[list[int]], target_experts: int
) -> V4Layout:
    model_type = str(config.get("model_type", ""))
    architectures = [str(item) for item in config.get("architectures", [])]
    if model_type != "deepseek_v4" and not any("V4" in item for item in architectures):
        raise ValueError("input config is not a DeepSeek-V4 checkpoint")
    num_layers = config_int(config, "num_hidden_layers", "n_layers")
    hash_layers = config_int(config, "num_hash_layers", "n_hash_layers")
    original_experts = config_int(config, "n_routed_experts")
    if num_layers != 43 or hash_layers != 3 or original_experts != 256:
        raise ValueError(
            "this implementation is intentionally scoped to V4-Flash "
            f"43 layers / 3 hash layers / 256 experts; found "
            f"{num_layers} / {hash_layers} / {original_experts}"
        )
    active_experts = config_int(
        config, "num_experts_per_tok", "n_activated_experts"
    )
    if not active_experts <= target_experts <= original_experts:
        raise ValueError(
            "target-experts must be between the model's per-token Top-K "
            f"({active_experts}) and {original_experts}; found {target_experts}"
        )
    if len(mask) != num_layers:
        raise ValueError(f"mask has {len(mask)} rows; expected {num_layers}")

    selected: list[tuple[int, ...]] = []
    counts: list[int] = []
    for layer, row in enumerate(mask):
        if len(row) != original_experts:
            raise ValueError(
                f"mask layer {layer} has {len(row)} experts; expected {original_experts}"
            )
        chosen = tuple(index for index, keep in enumerate(row) if keep == 1)
        expected = original_experts if layer < hash_layers else target_experts
        if len(chosen) != expected:
            region = "hash" if layer < hash_layers else "dynamic"
            raise ValueError(
                f"{region} layer {layer} keeps {len(chosen)} experts; expected {expected}"
            )
        if layer < hash_layers and chosen != tuple(range(original_experts)):
            raise ValueError(f"hash layer {layer} must preserve every expert unchanged")
        selected.append(chosen)
        counts.append(len(chosen))

    return V4Layout(
        num_layers=num_layers,
        hash_layers=hash_layers,
        original_experts=original_experts,
        target_dynamic_experts=target_experts,
        counts_by_layer=tuple(counts),
        selected_by_layer=tuple(selected),
    )


def _renamed_expert_key(match: re.Match[str], new_expert: int) -> str:
    return (
        f"{match.group('root')}.{match.group('layer')}.{match.group('block')}."
        f"experts.{new_expert}{match.group('suffix')}"
    )


def validate_source_layout(weight_map: dict[str, str], layout: V4Layout) -> None:
    experts_by_layer: dict[int, set[int]] = {
        layer: set() for layer in range(layout.num_layers)
    }
    suffixes: dict[tuple[int, int], set[str]] = {}
    gate_fields: dict[int, set[str]] = {
        layer: set() for layer in range(layout.num_layers)
    }
    for key in weight_map:
        expert_match = EXPERT_KEY.match(key)
        if expert_match:
            layer = int(expert_match.group("layer"))
            if layer < layout.num_layers:
                expert = int(expert_match.group("expert"))
                experts_by_layer[layer].add(expert)
                suffixes.setdefault((layer, expert), set()).add(
                    expert_match.group("suffix")
                )
            continue
        gate_match = GATE_KEY.match(key)
        if gate_match:
            layer = int(gate_match.group("layer"))
            if layer < layout.num_layers:
                gate_fields[layer].add(gate_match.group("field"))

    expected_ids = set(range(layout.original_experts))
    expected_suffixes: set[str] | None = None
    for layer in range(layout.num_layers):
        if experts_by_layer[layer] != expected_ids:
            missing = sorted(expected_ids - experts_by_layer[layer])
            extra = sorted(experts_by_layer[layer] - expected_ids)
            raise ValueError(
                f"source layer {layer} expert ids are incomplete; "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
        for expert in range(layout.original_experts):
            current = suffixes.get((layer, expert), set())
            if expected_suffixes is None:
                expected_suffixes = current
            if current != expected_suffixes:
                raise ValueError(
                    f"source layer {layer} expert {expert} tensors differ from "
                    f"the common layout: {sorted(current)}"
                )

    for layer in range(layout.num_layers):
        required = {"weight", "tid2eid"} if layer < layout.hash_layers else {"weight", "bias"}
        if not required.issubset(gate_fields[layer]):
            raise ValueError(
                f"source layer {layer} gate is missing {sorted(required - gate_fields[layer])}"
            )


def build_plan(
    index: dict[str, Any], layout: V4Layout, *, provenance: dict[str, str]
) -> PrunePlan:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no non-empty weight_map")
    validate_source_layout(weight_map, layout)

    new_ids = [
        {old: new for new, old in enumerate(selected)}
        for selected in layout.selected_by_layer
    ]
    actions: dict[str, str | None] = {}
    output_map: dict[str, str] = {}
    renamed = 0
    dropped = 0
    for key, shard in weight_map.items():
        output_key: str | None = key
        expert_match = EXPERT_KEY.match(key)
        if expert_match:
            layer = int(expert_match.group("layer"))
            expert = int(expert_match.group("expert"))
            if layer < layout.num_layers:
                if expert not in new_ids[layer]:
                    output_key = None
                elif layer >= layout.hash_layers:
                    output_key = _renamed_expert_key(expert_match, new_ids[layer][expert])
        actions[key] = output_key
        if output_key is None:
            dropped += 1
            continue
        if output_key in output_map:
            raise ValueError(f"renaming collision for output tensor {output_key}")
        output_map[output_key] = str(shard)
        renamed += int(output_key != key)

    fingerprint_payload = {
        "layout": asdict(layout),
        "source_config_sha256": provenance["source_config_sha256"],
        "source_index_sha256": provenance["source_index_sha256"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return PrunePlan(
        layout=layout,
        source_keys=len(weight_map),
        output_keys=len(output_map),
        dropped_keys=dropped,
        renamed_keys=renamed,
        output_weight_map=output_map,
        key_actions=actions,
        fingerprint=fingerprint,
    )


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"invalid safetensors header prefix: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length < 2 or header_length > 512 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length {header_length}: {path}")
        raw = handle.read(header_length)
    if len(raw) != header_length:
        raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw)
    metadata = header.pop("__metadata__", {})
    if not isinstance(header, dict) or not isinstance(metadata, dict):
        raise ValueError(f"invalid safetensors header object: {path}")
    return header, {str(key): str(value) for key, value in metadata.items()}


def tensor_bytes(entry: dict[str, Any]) -> int:
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"invalid safetensors data_offsets: {entry}")
    return int(offsets[1]) - int(offsets[0])


def source_keys_by_shard(index: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for key, shard in index["weight_map"].items():
        result.setdefault(str(shard), set()).add(str(key))
    return result


def estimate_output_bytes(
    input_dir: Path, index: dict[str, Any], plan: PrunePlan
) -> tuple[int, int]:
    total = 0
    largest_shard_output = 0
    for shard, indexed_keys in source_keys_by_shard(index).items():
        source = input_dir / shard
        if not source.is_file():
            raise FileNotFoundError(f"index references missing shard: {source}")
        header, _ = read_safetensors_header(source)
        header_keys = set(header)
        if header_keys != indexed_keys:
            raise ValueError(
                f"{source} header/index keys disagree: "
                f"header_only={sorted(header_keys - indexed_keys)[:5]}, "
                f"index_only={sorted(indexed_keys - header_keys)[:5]}"
            )
        shard_output = 0
        for key, entry in header.items():
            if plan.key_actions[key] is None:
                continue
            size = tensor_bytes(entry)
            shard_output += size
        # Safetensors headers and alignment are small; reserve 1 MiB per shard.
        shard_output += 1024 * 1024
        total += shard_output
        largest_shard_output = max(largest_shard_output, shard_output)
    return total, largest_shard_output


def make_output_config(
    source_config: dict[str, Any], plan: PrunePlan, mask_sha256: str
) -> dict[str, Any]:
    output = dict(source_config)
    output["n_routed_experts"] = plan.layout.original_experts
    selected_sets = [set(row) for row in plan.layout.selected_by_layer]
    output["easyep_expert_mask_by_layer"] = [
        [int(expert in selected_sets[layer]) for expert in range(plan.layout.original_experts)]
        for layer in range(plan.layout.num_layers)
    ]
    output["easyep_pruning"] = {
        "format_version": 1,
        "scope": "dynamic_moe_layers_only",
        "hash_layers_preserved": True,
        "hash_layer_ids": list(range(plan.layout.hash_layers)),
        "dynamic_layer_ids": list(
            range(plan.layout.hash_layers, plan.layout.num_layers)
        ),
        "original_experts_per_layer": plan.layout.original_experts,
        "target_dynamic_experts_per_layer": plan.layout.target_dynamic_experts,
        "dynamic_layer_prune_fraction": plan.layout.dynamic_prune_fraction,
        "main_layer_slot_prune_fraction": plan.layout.main_layer_slot_prune_fraction,
        "mask_sha256": mask_sha256,
        "plan_fingerprint": plan.fingerprint,
        "router_parameters_pruned": False,
        "router_mask_applied_at_runtime": True,
        "mtp_pruned": False,
    }
    return output


def copy_auxiliary_files(input_dir: Path, output_dir: Path) -> None:
    excluded = {"config.json", INDEX_NAME, MANIFEST_NAME, MARKER_NAME, MASK_COPY_NAME}
    for source in input_dir.iterdir():
        if source.name in excluded or source.name == ".cache":
            continue
        if source.is_file() and source.suffix == ".safetensors":
            continue
        target = output_dir / source.name
        if target.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        elif source.is_file():
            shutil.copy2(source, target)


def output_keys_by_shard(plan: PrunePlan) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for key, shard in plan.output_weight_map.items():
        result.setdefault(shard, set()).add(key)
    return result


def verify_output(
    output_dir: Path,
    output_config: dict[str, Any],
    plan: PrunePlan,
    *,
    require_manifest: bool,
) -> int:
    config_path = output_dir / "config.json"
    index_path = output_dir / INDEX_NAME
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"output is incomplete: {output_dir}")
    observed_config = load_json(config_path)
    observed_mask = observed_config.get("easyep_expert_mask_by_layer")
    if not isinstance(observed_mask, list) or len(observed_mask) != plan.layout.num_layers:
        raise ValueError("output config has no valid EASY-EP per-layer mask")
    observed_layout = build_layout(
        observed_config,
        observed_mask,
        plan.layout.target_dynamic_experts,
    )
    if observed_layout.selected_by_layer != plan.layout.selected_by_layer:
        raise ValueError("output config EASY-EP mask does not match this pruning plan")
    if observed_config.get("easyep_pruning") != output_config.get("easyep_pruning"):
        raise ValueError("output config pruning provenance does not match this plan")
    observed_index = load_json(index_path)
    if observed_index.get("weight_map") != plan.output_weight_map:
        raise ValueError("output weight index does not match the pruning plan")

    total_tensor_bytes = 0
    for shard, expected_keys in output_keys_by_shard(plan).items():
        path = output_dir / shard
        if not path.is_file():
            raise FileNotFoundError(f"output index references missing shard: {path}")
        header, _ = read_safetensors_header(path)
        if set(header) != expected_keys:
            raise ValueError(f"output shard keys do not match plan: {path}")
        total_tensor_bytes += sum(tensor_bytes(entry) for entry in header.values())
    metadata_size = int((observed_index.get("metadata") or {}).get("total_size", -1))
    if metadata_size != total_tensor_bytes:
        raise ValueError(
            f"output index total_size={metadata_size}, actual={total_tensor_bytes}"
        )
    if require_manifest:
        manifest = load_json(output_dir / MANIFEST_NAME)
        if manifest.get("plan_fingerprint") != plan.fingerprint:
            raise ValueError("output pruning manifest fingerprint does not match")
    return total_tensor_bytes


def _load_runtime_dependencies():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "physical V4 pruning requires torch and safetensors in V4_PYTHON"
        ) from exc
    return safe_open, save_file


def materialize(
    args: argparse.Namespace,
    source_config: dict[str, Any],
    index: dict[str, Any],
    plan: PrunePlan,
    output_config: dict[str, Any],
    estimated_bytes: int,
    largest_shard_output: int,
) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = output_dir / MARKER_NAME
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.is_file():
        verified = verify_output(
            output_dir, output_config, plan, require_manifest=True
        )
        print(f"checkpoint already complete and verified: {output_dir} ({verified} bytes)")
        return

    existing = [
        item.name
        for item in output_dir.iterdir()
        if item.name not in {MARKER_NAME} and not item.name.endswith(".tmp")
    ]
    if existing and not marker_path.is_file():
        raise FileExistsError(
            f"output directory contains unrelated/incomplete files without {MARKER_NAME}: "
            f"{existing[:8]}"
        )
    if marker_path.is_file():
        marker = load_json(marker_path)
        if marker.get("plan_fingerprint") != plan.fingerprint:
            raise ValueError("existing incomplete output belongs to a different pruning plan")
    else:
        atomic_write_json(
            marker_path,
            {
                "plan_fingerprint": plan.fingerprint,
                "started_at": utc_now(),
                "input_dir": str(args.input_dir.resolve()),
                "mask": str(args.mask_json.resolve()),
            },
        )

    if not args.skip_disk_check:
        reusable_bytes = 0
        for shard, expected_keys in output_keys_by_shard(plan).items():
            destination = output_dir / shard
            if not destination.is_file():
                continue
            header, _ = read_safetensors_header(destination)
            if set(header) != expected_keys:
                raise ValueError(
                    f"existing output shard has unexpected keys: {destination}"
                )
            reusable_bytes += destination.stat().st_size
        free = shutil.disk_usage(output_dir).free
        reserve = largest_shard_output + 2 * 1024**3
        required = max(0, estimated_bytes - reusable_bytes) + reserve
        if free < required:
            raise OSError(
                f"insufficient free disk for {output_dir}: free={free / 2**30:.2f} GiB, "
                f"required≈{required / 2**30:.2f} GiB after reusable shards"
            )

    safe_open, save_file = _load_runtime_dependencies()
    source_by_shard = source_keys_by_shard(index)
    output_by_shard = output_keys_by_shard(plan)
    completed = 0
    total_shards = len(source_by_shard)
    for shard, source_keys in sorted(source_by_shard.items()):
        expected_output_keys = output_by_shard.get(shard, set())
        destination = output_dir / shard
        if destination.is_file():
            header, _ = read_safetensors_header(destination)
            if set(header) == expected_output_keys:
                completed += 1
                print(f"[{completed}/{total_shards}] reuse {shard}", flush=True)
                continue
            raise ValueError(f"existing output shard has unexpected keys: {destination}")

        source = args.input_dir / shard
        output_tensors: dict[str, Any] = {}
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            shard_metadata = handle.metadata()
            for key in source_keys:
                output_key = plan.key_actions[key]
                if output_key is None:
                    continue
                tensor = handle.get_tensor(key)
                output_tensors[output_key] = tensor.contiguous()
        if set(output_tensors) != expected_output_keys:
            raise ValueError(f"planned/output tensor keys disagree for {shard}")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        save_file(output_tensors, str(temporary), metadata=shard_metadata or {})
        written_header, _ = read_safetensors_header(temporary)
        if set(written_header) != expected_output_keys:
            raise ValueError(f"written shard verification failed: {temporary}")
        os.replace(temporary, destination)
        del output_tensors
        completed += 1
        print(f"[{completed}/{total_shards}] wrote {destination}", flush=True)

    copy_auxiliary_files(args.input_dir, output_dir)
    shutil.copy2(args.mask_json, output_dir / MASK_COPY_NAME)
    atomic_write_json(output_dir / "config.json", output_config)

    total_tensor_bytes = 0
    for shard in sorted(output_by_shard):
        header, _ = read_safetensors_header(output_dir / shard)
        total_tensor_bytes += sum(tensor_bytes(entry) for entry in header.values())
    output_index = {
        "metadata": {
            **dict(index.get("metadata") or {}),
            "total_size": total_tensor_bytes,
            "easyep_plan_fingerprint": plan.fingerprint,
        },
        "weight_map": plan.output_weight_map,
    }
    atomic_write_json(output_dir / INDEX_NAME, output_index)
    verify_output(output_dir, output_config, plan, require_manifest=False)

    marker = load_json(marker_path)
    manifest = {
        **marker,
        "completed_at": utc_now(),
        "plan_fingerprint": plan.fingerprint,
        "layout": asdict(plan.layout),
        "source_keys": plan.source_keys,
        "output_keys": plan.output_keys,
        "dropped_keys": plan.dropped_keys,
        "renamed_keys": plan.renamed_keys,
        "router_parameters_pruned": False,
        "router_mask_applied_at_runtime": True,
        "source_checkpoint_bytes": int((index.get("metadata") or {}).get("total_size", 0)),
        "output_checkpoint_bytes": total_tensor_bytes,
        "output_dir": str(output_dir.resolve()),
    }
    atomic_write_json(manifest_path, manifest)
    verify_output(output_dir, output_config, plan, require_manifest=True)
    marker_path.unlink()
    print(
        f"completed {plan.layout.target_dynamic_experts}-expert dynamic-layer "
        f"checkpoint: {output_dir} ({total_tensor_bytes / 2**30:.2f} GiB)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune DeepSeek-V4-Flash layers 3..42 while preserving hash layers 0..2"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-json", type=Path, required=True)
    parser.add_argument(
        "--target-experts",
        type=int,
        required=True,
        help=(
            "uniform retained expert count for dynamic layers; must be at least "
            "the model's per-token Top-K and at most n_routed_experts"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-disk-check", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.mask_json = args.mask_json.expanduser().resolve()
    config_path = args.input_dir / "config.json"
    index_path = args.input_dir / INDEX_NAME
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"input must contain config.json and {INDEX_NAME}: {args.input_dir}"
        )
    if not args.mask_json.is_file():
        raise FileNotFoundError(f"mask does not exist: {args.mask_json}")

    source_config = load_json(config_path)
    index = load_json(index_path)
    mask = load_mask(args.mask_json)
    layout = build_layout(source_config, mask, args.target_experts)
    provenance = {
        "source_config_sha256": sha256_file(config_path),
        "source_index_sha256": sha256_file(index_path),
        "mask_sha256": sha256_file(args.mask_json),
    }
    plan = build_plan(index, layout, provenance=provenance)
    output_config = make_output_config(
        source_config, plan, provenance["mask_sha256"]
    )
    estimated_bytes, largest_shard_output = estimate_output_bytes(
        args.input_dir, index, plan
    )
    print(
        json.dumps(
            {
                "input": str(args.input_dir),
                "output": str(args.output_dir),
                "target_dynamic_experts": layout.target_dynamic_experts,
                "hash_layers_preserved": list(range(layout.hash_layers)),
                "dynamic_layers_pruned": list(range(layout.hash_layers, layout.num_layers)),
                "dynamic_prune_percent": round(layout.dynamic_prune_fraction * 100, 6),
                "main_layer_slot_prune_percent": round(
                    layout.main_layer_slot_prune_fraction * 100, 6
                ),
                "source_keys": plan.source_keys,
                "output_keys": plan.output_keys,
                "estimated_output_gib": round(estimated_bytes / 2**30, 3),
                "plan_fingerprint": plan.fingerprint,
            },
            indent=2,
        )
    )
    if args.dry_run:
        print("dry-run passed; no checkpoint files were written")
        return 0
    if args.verify_only:
        total = verify_output(
            args.output_dir, output_config, plan, require_manifest=True
        )
        print(f"verified output checkpoint: {args.output_dir} ({total} bytes)")
        return 0
    materialize(
        args,
        source_config,
        index,
        plan,
        output_config,
        estimated_bytes,
        largest_shard_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
