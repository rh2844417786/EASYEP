#!/usr/bin/env python3
"""Run a comparable full/prune-25/prune-50 EASY-EP evaluation matrix.

The runner intentionally accepts three materialized checkpoint directories. It
does not pretend that changing a label or mask path physically prunes a model.
For each variant it launches an isolated SGLang server, runs the repository's
three scored math datasets, samples GPU telemetry, and writes resumable raw
outputs plus JSON/CSV/Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CLIENT = REPO_ROOT / "evaluation" / "run_sglang.py"
SMOKE_CLIENT = REPO_ROOT / "scripts" / "smoke_v4_server.py"
GPU_IDLE_CHECK = REPO_ROOT / "scripts" / "check_v4_gpus_idle.sh"
DEFAULT_DATASETS = ("AIME24", "hmmt_feb_2025", "AIME25")
INDEXED_EXPERT_KEY = re.compile(
    r"^(?:model\.)?layers\.(?P<layer>\d+)\.(?:ffn|mlp)\.experts\."
    r"(?P<expert>\d+)\."
)
INDEXED_GATE_KEY = re.compile(
    r"^(?:model\.)?layers\.(?P<layer>\d+)\.(?:ffn|mlp)\.gate\."
    r"(?P<field>weight|bias|tid2eid)$"
)


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    global_routed_experts: int
    routed_experts: int
    hash_layers: int
    hash_routed_experts: int
    expert_counts_by_layer: tuple[int, ...]
    main_layer_slot_prune_fraction: float
    pruning_scope: str | None
    indexed_tensors: int
    shard_count: int
    checkpoint_bytes: int
    shard_sha256: tuple[tuple[str, str], ...]
    fingerprint: str


@dataclass(frozen=True)
class VariantSpec:
    name: str
    prune_fraction: float
    expected_experts: int
    checkpoint: CheckpointInfo


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_parts(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _stat_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def hash_checkpoint_shards(
    paths: list[Path],
    *,
    cache_path: Path | None,
    rehash: bool,
) -> tuple[tuple[str, str], ...]:
    cache: dict[str, Any] = {"format_version": 1, "files": {}}
    if cache_path is not None and cache_path.is_file():
        try:
            loaded = load_json(cache_path)
            if loaded.get("format_version") == 1 and isinstance(
                loaded.get("files"), dict
            ):
                cache = loaded
        except (OSError, ValueError, json.JSONDecodeError):
            cache = {"format_version": 1, "files": {}}

    files = cache["files"]
    changed = False
    result: list[tuple[str, str]] = []
    total_bytes = sum(path.stat().st_size for path in paths)
    if total_bytes >= 1024**3:
        print(
            f"checkpoint payload verification: {len(paths)} shard(s), "
            f"{total_bytes / 2**30:.2f} GiB",
            flush=True,
        )
    for index, path in enumerate(paths, 1):
        key = str(path.resolve())
        signature = _stat_signature(path)
        cached = files.get(key)
        digest: str | None = None
        if not rehash and isinstance(cached, dict):
            candidate = cached.get("sha256")
            if (
                cached.get("signature") == signature
                and isinstance(candidate, str)
                and re.fullmatch(r"[0-9a-f]{64}", candidate)
            ):
                digest = candidate
        if digest is None:
            digest = sha256_file(path)
            if _stat_signature(path) != signature:
                raise RuntimeError(
                    f"checkpoint shard changed while being hashed: {path}"
                )
            files[key] = {"signature": signature, "sha256": digest}
            changed = True
            if total_bytes >= 1024**3:
                print(f"  hashed {index}/{len(paths)}: {path.name}", flush=True)
        result.append((path.name, digest))

    if cache_path is not None and changed:
        cache["updated_at"] = utc_now()
        atomic_write_json(cache_path, cache)
    return tuple(result)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_indexed_expert_ids(
    weight_map: dict[str, Any], expert_counts: list[int], index_path: Path
) -> None:
    observed = [set() for _ in expert_counts]
    for key in weight_map:
        match = INDEXED_EXPERT_KEY.match(str(key))
        if match is None:
            continue
        layer = int(match.group("layer"))
        if 0 <= layer < len(expert_counts):
            observed[layer].add(int(match.group("expert")))

    for layer, expected_count in enumerate(expert_counts):
        expected = set(range(expected_count))
        if observed[layer] != expected:
            missing = sorted(expected - observed[layer])
            extra = sorted(observed[layer] - expected)
            raise ValueError(
                f"{index_path} layer {layer} expert ids do not match config; "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )


def _validate_indexed_router_fields(
    weight_map: dict[str, Any], num_layers: int, hash_layers: int, index_path: Path
) -> None:
    fields = [set() for _ in range(num_layers)]
    for key in weight_map:
        match = INDEXED_GATE_KEY.match(str(key))
        if match is None:
            continue
        layer = int(match.group("layer"))
        if 0 <= layer < num_layers:
            fields[layer].add(match.group("field"))
    for layer in range(num_layers):
        required = {"weight", "tid2eid"} if layer < hash_layers else {"weight", "bias"}
        if not required.issubset(fields[layer]):
            raise ValueError(
                f"{index_path} layer {layer} is missing full router fields "
                f"{sorted(required - fields[layer])}"
            )


def _validate_pruning_provenance(
    path: Path,
    metadata: dict[str, Any],
    expert_counts: list[int],
    global_experts: int,
    hash_layers: int,
) -> None:
    dynamic_experts = expert_counts[hash_layers]
    expected_dynamic_prune = 1.0 - dynamic_experts / global_experts
    expected_main_prune = 1.0 - sum(expert_counts) / (
        len(expert_counts) * global_experts
    )
    exact_fields = {
        "format_version": 1,
        "scope": "dynamic_moe_layers_only",
        "hash_layers_preserved": True,
        "hash_layer_ids": list(range(hash_layers)),
        "dynamic_layer_ids": list(range(hash_layers, len(expert_counts))),
        "original_experts_per_layer": global_experts,
        "target_dynamic_experts_per_layer": dynamic_experts,
        "router_parameters_pruned": False,
        "router_mask_applied_at_runtime": True,
        "mtp_pruned": False,
    }
    for field, expected in exact_fields.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"{path / 'config.json'} easyep_pruning.{field} must be {expected!r}"
            )
    for field, expected in (
        ("dynamic_layer_prune_fraction", expected_dynamic_prune),
        ("main_layer_slot_prune_fraction", expected_main_prune),
    ):
        try:
            observed = float(metadata[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path / 'config.json'} has invalid easyep_pruning.{field}"
            ) from exc
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{path / 'config.json'} easyep_pruning.{field}={observed}, "
                f"expected {expected}"
            )

    plan_fingerprint = metadata.get("plan_fingerprint")
    mask_sha256 = metadata.get("mask_sha256")
    for field, value in (
        ("plan_fingerprint", plan_fingerprint),
        ("mask_sha256", mask_sha256),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(
                f"{path / 'config.json'} easyep_pruning.{field} is not a SHA-256"
            )

    manifest_path = path / "easyep_pruning_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pruned checkpoint manifest is missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("plan_fingerprint") != plan_fingerprint:
        raise ValueError(f"{manifest_path} plan fingerprint does not match config")
    if (
        manifest.get("router_parameters_pruned") is not False
        or manifest.get("router_mask_applied_at_runtime") is not True
    ):
        raise ValueError(f"{manifest_path} does not preserve the full router")
    layout = manifest.get("layout")
    if not isinstance(layout, dict) or layout.get("counts_by_layer") != expert_counts:
        raise ValueError(f"{manifest_path} per-layer expert counts do not match config")


def inspect_checkpoint(
    path: Path,
    *,
    hash_cache_path: Path | None = None,
    rehash: bool = False,
) -> CheckpointInfo:
    path = path.expanduser().resolve()
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"{path} must contain config.json and model.safetensors.index.json"
        )

    config_bytes = config_path.read_bytes()
    index_bytes = index_path.read_bytes()
    config = json.loads(config_bytes)
    index = json.loads(index_bytes)
    global_routed_experts = int(config.get("n_routed_experts", 0) or 0)
    if global_routed_experts < 1:
        raise ValueError(f"{config_path} has no positive n_routed_experts")
    hash_layers = int(
        config.get("num_hash_layers", config.get("n_hash_layers", 0)) or 0
    )
    num_layers = int(config.get("num_hidden_layers", config.get("n_layers", 0)) or 0)
    if num_layers < 1 or not 0 <= hash_layers < num_layers:
        raise ValueError(
            f"{config_path} has invalid layer counts: layers={num_layers}, "
            f"hash_layers={hash_layers}"
        )
    raw_mask = config.get("easyep_expert_mask_by_layer")
    if raw_mask is None:
        expert_counts = [global_routed_experts] * num_layers
    else:
        if not isinstance(raw_mask, list) or len(raw_mask) != num_layers:
            raise ValueError(
                f"{config_path} easyep_expert_mask_by_layer must have {num_layers} rows"
            )
        expert_counts = []
        for layer, row in enumerate(raw_mask):
            if not isinstance(row, list) or len(row) != global_routed_experts:
                raise ValueError(
                    f"{config_path} mask layer {layer} must have "
                    f"{global_routed_experts} entries"
                )
            if any(value not in (0, 1, 0.0, 1.0, False, True) for value in row):
                raise ValueError(f"{config_path} mask layer {layer} is not binary")
            expert_counts.append(sum(int(value) for value in row))
    if any(
        value != global_routed_experts for value in expert_counts[:hash_layers]
    ):
        raise ValueError(
            f"{config_path} prunes a hash layer; layers 0..{hash_layers - 1} "
            f"must retain {global_routed_experts} experts"
        )
    dynamic_counts = set(expert_counts[hash_layers:])
    if len(dynamic_counts) != 1:
        raise ValueError(
            f"{config_path} must use one uniform expert count for dynamic layers; "
            f"found {sorted(dynamic_counts)}"
        )
    routed_experts = next(iter(dynamic_counts))
    pruning_metadata = config.get("easyep_pruning")
    if routed_experts != global_routed_experts:
        if not isinstance(pruning_metadata, dict):
            raise ValueError(
                f"{config_path} has a pruning mask without easyep_pruning provenance"
            )
        if (
            pruning_metadata.get("scope") != "dynamic_moe_layers_only"
            or pruning_metadata.get("hash_layers_preserved") is not True
            or pruning_metadata.get("mtp_pruned") is not False
        ):
            raise ValueError(f"{config_path} has invalid EASY-EP V4 pruning provenance")
        _validate_pruning_provenance(
            path,
            pruning_metadata,
            expert_counts,
            global_routed_experts,
            hash_layers,
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path} has no non-empty weight_map")
    _validate_indexed_expert_ids(weight_map, expert_counts, index_path)
    _validate_indexed_router_fields(weight_map, num_layers, hash_layers, index_path)

    shard_names = sorted(set(str(name) for name in weight_map.values()))
    missing = [name for name in shard_names if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{path} is missing {len(missing)} indexed shard(s): {missing[:5]}"
        )
    shard_paths = [path / name for name in shard_names]
    checkpoint_bytes = sum(item.stat().st_size for item in shard_paths)
    shard_sha256 = hash_checkpoint_shards(
        shard_paths, cache_path=hash_cache_path, rehash=rehash
    )
    fingerprint_parts: list[bytes] = [config_bytes, b"\0", index_bytes, b"\0"]
    for name, digest in shard_sha256:
        fingerprint_parts.extend(
            (name.encode("utf-8"), b"\0", digest.encode("ascii"), b"\0")
        )
    fingerprint = sha256_parts(fingerprint_parts)
    return CheckpointInfo(
        path=str(path),
        global_routed_experts=global_routed_experts,
        routed_experts=routed_experts,
        hash_layers=hash_layers,
        hash_routed_experts=(
            expert_counts[0] if hash_layers else routed_experts
        ),
        expert_counts_by_layer=tuple(expert_counts),
        main_layer_slot_prune_fraction=round(
            1.0 - sum(expert_counts) / (num_layers * global_routed_experts), 10
        ),
        pruning_scope=(
            pruning_metadata.get("scope")
            if isinstance(pruning_metadata, dict)
            else None
        ),
        indexed_tensors=len(weight_map),
        shard_count=len(shard_names),
        checkpoint_bytes=checkpoint_bytes,
        shard_sha256=shard_sha256,
        fingerprint=fingerprint,
    )


def build_variant_specs(
    full_model: Path,
    prune25_model: Path,
    prune50_model: Path,
    *,
    hash_cache_path: Path | None = None,
    rehash: bool = False,
) -> list[VariantSpec]:
    checkpoints = [
        inspect_checkpoint(
            full_model, hash_cache_path=hash_cache_path, rehash=rehash
        ),
        inspect_checkpoint(
            prune25_model, hash_cache_path=hash_cache_path, rehash=rehash
        ),
        inspect_checkpoint(
            prune50_model, hash_cache_path=hash_cache_path, rehash=rehash
        ),
    ]
    original_experts = checkpoints[0].routed_experts
    if any(
        value != checkpoints[0].global_routed_experts
        for value in checkpoints[0].expert_counts_by_layer
    ):
        raise ValueError("full checkpoint is already mask-pruned")
    expected_counts = [
        original_experts,
        original_experts * 3 // 4,
        original_experts // 2,
    ]
    if original_experts % 4:
        raise ValueError(
            f"full checkpoint expert count {original_experts} is not divisible by 4"
        )

    names = ("full", "prune25", "prune50")
    fractions = (0.0, 0.25, 0.50)
    specs: list[VariantSpec] = []
    for name, fraction, expected, checkpoint in zip(
        names, fractions, expected_counts, checkpoints
    ):
        if checkpoint.routed_experts != expected:
            raise ValueError(
                f"{name} must have {expected} experts in dynamic layers 3..42 "
                f"({fraction:.0%} dynamic-layer pruning from {original_experts}), found "
                f"{checkpoint.routed_experts} in {checkpoint.path}"
            )
        if checkpoint.global_routed_experts != checkpoints[0].global_routed_experts:
            raise ValueError(f"{name} changed the global/hash expert count")
        if checkpoint.hash_layers != checkpoints[0].hash_layers:
            raise ValueError(f"{name} changed the hash-layer count")
        if fraction > 0 and checkpoint.pruning_scope != "dynamic_moe_layers_only":
            raise ValueError(f"{name} is not a validated dynamic-layer-only checkpoint")
        specs.append(VariantSpec(name, fraction, expected, checkpoint))

    resolved_paths = [item.checkpoint.path for item in specs]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("full/prune25/prune50 must be three different directories")
    fingerprints = [item.checkpoint.fingerprint for item in specs]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError(
            "two variants have identical config/index fingerprints; refusing to "
            "report the same checkpoint under different pruning labels"
        )
    return specs


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def parse_evaluation_output(
    path: Path,
    wall_seconds: float,
    *,
    expected_dataset: str | None = None,
    expected_model: str | None = None,
    expected_total: int | None = None,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    summary_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if record.get("type") == "sample":
                if summary is not None:
                    raise ValueError(
                        f"evaluation output contains samples after its summary: {path}"
                    )
                samples.append(record)
            elif record.get("type") == "summary":
                summary = record
                summary_count += 1
            else:
                raise ValueError(
                    f"evaluation output has unknown record type at {path}:{line_number}"
                )
    if summary is None:
        raise ValueError(f"evaluation output has no summary record: {path}")
    if summary_count != 1:
        raise ValueError(
            f"evaluation output has {summary_count} summary records, expected one: {path}"
        )
    total = int(summary["total"])
    correct = int(summary["correct"])
    if total < 1:
        raise ValueError(f"evaluation output has no scored samples: {path}")
    if total != len(samples):
        raise ValueError(
            f"evaluation output sample count={len(samples)} but summary total={total}: {path}"
        )
    if not 0 <= correct <= total:
        raise ValueError(f"evaluation output has invalid correct/total: {correct}/{total}")
    job_ids = [sample.get("job_id") for sample in samples]
    if any(not isinstance(value, str) or not value for value in job_ids):
        raise ValueError(f"evaluation output has a sample without a job_id: {path}")
    if len(set(job_ids)) != len(job_ids):
        raise ValueError(f"evaluation output has duplicate job_id values: {path}")
    observed_accuracy = float(summary["accuracy"])
    expected_accuracy = round(correct / total * 100, 2) if total else 0.0
    if not math.isclose(observed_accuracy, expected_accuracy, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"evaluation output accuracy={observed_accuracy}, expected {expected_accuracy}: {path}"
        )
    if expected_total is not None and total != expected_total:
        raise ValueError(
            f"evaluation output total={total}, expected {expected_total}: {path}"
        )
    observed_dataset = summary.get("dataset")
    observed_model = summary.get("model")
    run_fingerprint = summary.get("run_fingerprint")
    if expected_dataset is not None and observed_dataset != expected_dataset:
        raise ValueError(
            f"evaluation output dataset={observed_dataset!r}, "
            f"expected {expected_dataset!r}: {path}"
        )
    if expected_model is not None and observed_model != expected_model:
        raise ValueError(
            f"evaluation output model={observed_model!r}, "
            f"expected {expected_model!r}: {path}"
        )
    if (expected_dataset is not None or expected_model is not None) and (
        not isinstance(run_fingerprint, str) or not run_fingerprint
    ):
        raise ValueError(f"evaluation output has no run fingerprint: {path}")

    latencies = [float(item["latency_seconds"]) for item in samples]
    prompt_tokens = 0
    completion_tokens = 0
    for sample in samples:
        usage = sample.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
    return {
        "accuracy": observed_accuracy,
        "correct": correct,
        "total": total,
        "dataset": observed_dataset,
        "model": observed_model,
        "run_fingerprint": run_fingerprint,
        "evaluator_backend": summary.get("evaluator_backend"),
        "wall_seconds": round(wall_seconds, 6),
        "samples_per_second": round(len(samples) / wall_seconds, 6)
        if wall_seconds > 0
        else None,
        "latency_mean_seconds": round(statistics.fmean(latencies), 6)
        if latencies
        else None,
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "latency_p99_seconds": percentile(latencies, 0.99),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_file": str(path),
        "output_bytes": path.stat().st_size,
        "output_sha256": sha256_file(path),
    }


def summarize_gpu_samples(
    samples: list[dict[str, Any]], baseline_memory_mib: dict[int, float]
) -> dict[str, Any]:
    by_gpu: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_gpu.setdefault(int(sample["gpu_index"]), []).append(sample)
    gpu_rows: list[dict[str, Any]] = []
    for gpu_index, gpu_samples in sorted(by_gpu.items()):
        used = [float(item["memory_used_mib"]) for item in gpu_samples]
        baseline = baseline_memory_mib.get(gpu_index, used[0])
        utilization = [
            float(item["utilization_gpu_percent"]) for item in gpu_samples
        ]
        power = [float(item["power_draw_watts"]) for item in gpu_samples]
        gpu_rows.append(
            {
                "gpu_index": gpu_index,
                "gpu_name": gpu_samples[0]["gpu_name"],
                "samples": len(gpu_samples),
                "baseline_memory_mib": baseline,
                "peak_memory_mib": max(used),
                "peak_memory_delta_mib": max(used) - baseline,
                "mean_memory_mib": round(statistics.fmean(used), 3),
                "peak_utilization_percent": max(utilization),
                "mean_utilization_percent": round(
                    statistics.fmean(utilization), 3
                ),
                "mean_power_watts": round(statistics.fmean(power), 3),
            }
        )
    return {
        "sample_rows": len(samples),
        "gpus": gpu_rows,
        "max_peak_memory_mib": max(
            (row["peak_memory_mib"] for row in gpu_rows), default=None
        ),
        "sum_per_gpu_peak_memory_mib": sum(
            row["peak_memory_mib"] for row in gpu_rows
        ),
    }


class GpuMonitor:
    FIELDS = (
        "timestamp_utc",
        "epoch_seconds",
        "phase",
        "variant",
        "gpu_index",
        "gpu_name",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_percent",
        "power_draw_watts",
    )

    def __init__(self, path: Path, variant: str, gpu_list: str, interval: float):
        self.path = path
        self.variant = variant
        self.gpu_list = gpu_list
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.baseline_memory_mib: dict[int, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def _expected_gpu_indices(self) -> set[int]:
        try:
            return {int(value) for value in self.gpu_list.split(",")}
        except ValueError as exc:
            raise ValueError(f"invalid GPU list: {self.gpu_list!r}") from exc

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        baseline = self._query("baseline")
        self.samples.extend(baseline)
        self.baseline_memory_mib = {
            int(item["gpu_index"]): float(item["memory_used_mib"])
            for item in baseline
        }
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(baseline)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join(timeout=max(35.0, self.interval * 3))
        if self._thread.is_alive():
            raise RuntimeError("GPU telemetry thread did not stop after nvidia-smi timeout")

    def _query(self, phase: str) -> list[dict[str, Any]]:
        query = (
            "index,name,memory.used,memory.total,utilization.gpu,power.draw"
        )
        epoch = time.time()
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                self.gpu_list,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
        samples: list[dict[str, Any]] = []
        for row in csv.reader(completed.stdout.splitlines()):
            if len(row) != 6:
                raise ValueError(f"unexpected nvidia-smi row: {row}")
            samples.append(
                {
                    "timestamp_utc": datetime.fromtimestamp(
                        epoch, timezone.utc
                    ).isoformat(),
                    "epoch_seconds": round(epoch, 6),
                    "phase": phase,
                    "variant": self.variant,
                    "gpu_index": int(row[0].strip()),
                    "gpu_name": row[1].strip(),
                    "memory_used_mib": float(row[2].strip()),
                    "memory_total_mib": float(row[3].strip()),
                    "utilization_gpu_percent": float(row[4].strip()),
                    "power_draw_watts": float(row[5].strip()),
                }
            )
        expected_gpus = self._expected_gpu_indices()
        observed_gpus = {int(item["gpu_index"]) for item in samples}
        if observed_gpus != expected_gpus or len(samples) != len(expected_gpus):
            raise RuntimeError(
                f"{phase} telemetry did not return each requested GPU exactly once; "
                f"expected={sorted(expected_gpus)}, observed={sorted(observed_gpus)}"
            )
        return samples

    def _run(self) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            while not self._stop.is_set():
                try:
                    queried = self._query("runtime")
                    for sample in queried:
                        self.samples.append(sample)
                        writer.writerow(sample)
                    handle.flush()
                except Exception as exc:  # telemetry failure must not kill evaluation
                    self.errors.append(f"{utc_now()}: {exc}")
                self._stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        evidence_errors = list(self.errors)
        expected_gpus = self._expected_gpu_indices()
        runtime_gpus = {
            int(sample["gpu_index"])
            for sample in self.samples
            if sample.get("phase") == "runtime"
        }
        if runtime_gpus != expected_gpus:
            evidence_errors.append(
                "runtime telemetry is incomplete; "
                f"expected={sorted(expected_gpus)}, observed={sorted(runtime_gpus)}"
            )
        result = {
            "trace_file": str(self.path),
            "interval_seconds": self.interval,
            "errors": evidence_errors,
            **summarize_gpu_samples(self.samples, self.baseline_memory_mib),
        }
        if self.path.is_file():
            result.update(
                {
                    "trace_bytes": self.path.stat().st_size,
                    "trace_sha256": sha256_file(self.path),
                }
            )
        return result


def port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def require_idle_gpus(gpu_list: str) -> None:
    env = os.environ.copy()
    env["GPU_LIST"] = gpu_list
    completed = subprocess.run(
        ["bash", str(GPU_IDLE_CHECK)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    output = completed.stdout.strip()
    if output:
        print(output, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "selected GPUs are not exclusive; no evaluation server was started"
        )


def fetch_models(port: int, timeout: float = 3.0) -> dict[str, Any]:
    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(process: subprocess.Popen[Any], port: int, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"SGLang exited before readiness with status {return_code}")
        try:
            payload = fetch_models(port)
        except (OSError, URLError, ValueError, json.JSONDecodeError):
            time.sleep(10)
            continue
        models = payload.get("data") or []
        if models and models[0].get("id"):
            return str(models[0]["id"])
        time.sleep(10)
    raise TimeoutError(f"SGLang was not ready within {timeout} seconds")


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(process_group_id: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_group_exists(process_group_id):
            return True
        time.sleep(0.1)
    return not process_group_exists(process_group_id)


def stop_process_group(process: subprocess.Popen[Any], timeout: float = 30.0) -> int:
    process_group_id = process.pid  # start_new_session=True makes PGID equal the leader PID.
    leader_status = process.poll()
    if process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if not _wait_for_process_group_exit(process_group_id, timeout):
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not _wait_for_process_group_exit(process_group_id, 10.0):
                raise RuntimeError(
                    f"process group {process_group_id} survived SIGKILL"
                )
    if process.poll() is None:
        try:
            leader_status = process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"process-group leader {process.pid} was not reaped"
            ) from exc
    else:
        leader_status = process.returncode
    return int(leader_status or 0)


def tail_text(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def server_command(args: argparse.Namespace, spec: VariantSpec) -> list[str]:
    return [
        str(args.server_python),
        "-m",
        "sglang.launch_server",
        "--trust-remote-code",
        "--model-path",
        spec.checkpoint.path,
        "--tp",
        str(args.tp),
        "--moe-runner-backend",
        args.moe_backend,
        "--reasoning-parser",
        "deepseek-v4",
        "--tool-call-parser",
        "deepseekv4",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--disable-cuda-graph",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--context-length",
        str(args.context_length),
        "--max-running-requests",
        str(args.max_running_requests),
        "--watchdog-timeout",
        str(args.watchdog_timeout),
        "--disable-custom-all-reduce",
        "--disable-shared-experts-fusion",
    ]


def runtime_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": args.gpu_list,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONUNBUFFERED": "1",
            "NCCL_IB_DISABLE": env.get("NCCL_IB_DISABLE", "1"),
            "NCCL_SOCKET_IFNAME": env.get("NCCL_SOCKET_IFNAME", "lo"),
            "GLOO_SOCKET_IFNAME": env.get("GLOO_SOCKET_IFNAME", "lo"),
            "NCCL_CUMEM_HOST_ENABLE": env.get("NCCL_CUMEM_HOST_ENABLE", "0"),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": env.get(
                "TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"
            ),
            "TORCH_NCCL_BLOCKING_WAIT": env.get("TORCH_NCCL_BLOCKING_WAIT", "1"),
        }
    )
    return env


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    timeout: float | None = None,
    label: str = "command",
    stream_output: bool = False,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            text=True,
            bufsize=1,
            stdout=subprocess.PIPE if stream_output else handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        forwarder: threading.Thread | None = None
        if stream_output:
            assert process.stdout is not None

            def forward_output() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    handle.write(line)
                    print(line, end="", flush=True)

            forwarder = threading.Thread(target=forward_output, daemon=True)
            forwarder.start()
        try:
            while True:
                elapsed = time.monotonic() - started
                if timeout is not None and elapsed >= timeout:
                    stop_process_group(process)
                    raise TimeoutError(
                        f"{label} exceeded {timeout:.0f}s; see {log_path}"
                    )
                wait_seconds = 60.0
                if timeout is not None:
                    wait_seconds = max(0.1, min(wait_seconds, timeout - elapsed))
                try:
                    return_code = process.wait(timeout=wait_seconds)
                    break
                except subprocess.TimeoutExpired:
                    # Long generations can legitimately exceed several minutes;
                    # per-question progress is emitted by the evaluation client.
                    continue
        except BaseException:
            stop_process_group(process)
            raise
        finally:
            stop_process_group(process)
            if forwarder is not None:
                forwarder.join(timeout=10)
                if forwarder.is_alive():
                    raise RuntimeError(f"{label} log forwarder did not stop")
    if return_code != 0:
        raise RuntimeError(
            f"command failed with status {return_code}: {' '.join(command)}\n"
            f"last log lines:\n{tail_text(log_path)}"
        )


def _same_scientific_dataset_result(
    stored: dict[str, Any], observed: dict[str, Any]
) -> bool:
    fields = (
        "accuracy",
        "correct",
        "total",
        "dataset",
        "model",
        "run_fingerprint",
        "evaluator_backend",
        "prompt_tokens",
        "completion_tokens",
        "output_bytes",
        "output_sha256",
    )
    return all(stored.get(field) == observed.get(field) for field in fields)


def validated_cached_datasets(
    previous: dict[str, Any],
    args: argparse.Namespace,
    spec: VariantSpec,
    variant_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    if previous.get("protocol_fingerprint") != args.protocol_fingerprint:
        return {}, ["variant protocol fingerprint is missing or stale"]
    stored_checkpoint = previous.get("checkpoint") or {}
    if stored_checkpoint.get("fingerprint") != spec.checkpoint.fingerprint:
        return {}, ["variant checkpoint fingerprint is missing or stale"]
    served_model = previous.get("served_model")
    if not isinstance(served_model, str) or not served_model:
        return {}, ["variant served-model identity is missing"]

    valid: dict[str, Any] = {}
    stored_datasets = previous.get("datasets") or {}
    for dataset in args.datasets:
        stored = stored_datasets.get(dataset)
        if not isinstance(stored, dict):
            reasons.append(f"{dataset}: cached result is missing")
            continue
        output_path = variant_dir / "evaluation" / f"{dataset}.jsonl"
        if not output_path.is_file():
            reasons.append(f"{dataset}: raw JSONL is missing")
            continue
        try:
            wall_seconds = float(stored["wall_seconds"])
            observed = parse_evaluation_output(
                output_path,
                wall_seconds,
                expected_dataset=dataset,
                expected_model=served_model,
                expected_total=args.dataset_totals[dataset] * args.repeats,
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            reasons.append(f"{dataset}: raw JSONL validation failed: {exc}")
            continue
        if not _same_scientific_dataset_result(stored, observed):
            reasons.append(f"{dataset}: cached metrics/hash do not match raw JSONL")
            continue
        valid[dataset] = observed
    return valid, reasons


def validate_completed_variant_artifacts(
    previous: dict[str, Any], variant_dir: Path, gpu_list: str
) -> list[str]:
    reasons: list[str] = []
    gpu = previous.get("gpu") or {}
    if gpu.get("errors"):
        reasons.append("GPU telemetry recorded sampling errors")
    telemetry = variant_dir / "gpu_telemetry.csv"
    if not telemetry.is_file():
        reasons.append("GPU telemetry CSV is missing")
    elif (
        gpu.get("trace_bytes") != telemetry.stat().st_size
        or gpu.get("trace_sha256") != sha256_file(telemetry)
    ):
        reasons.append("GPU telemetry hash/size does not match variant_result.json")
    else:
        try:
            expected_gpus = {int(value) for value in gpu_list.split(",")}
            with telemetry.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != GpuMonitor.FIELDS:
                    raise ValueError("telemetry header does not match the expected schema")
                rows = list(reader)
            observed_gpus = {int(row["gpu_index"]) for row in rows}
            baseline_gpus = {
                int(row["gpu_index"])
                for row in rows
                if row.get("phase") == "baseline"
            }
            runtime_gpus = {
                int(row["gpu_index"])
                for row in rows
                if row.get("phase") == "runtime"
            }
            if (
                observed_gpus != expected_gpus
                or baseline_gpus != expected_gpus
                or runtime_gpus != expected_gpus
            ):
                raise ValueError(
                    "telemetry must contain baseline and runtime rows for every GPU"
                )
            baseline_counts = {
                gpu_index: sum(
                    row.get("phase") == "baseline"
                    and int(row["gpu_index"]) == gpu_index
                    for row in rows
                )
                for gpu_index in expected_gpus
            }
            if any(count != 1 for count in baseline_counts.values()):
                raise ValueError("telemetry must contain exactly one baseline per GPU")
            expected_variant = previous.get("variant")
            if expected_variant and any(
                row.get("variant") != expected_variant for row in rows
            ):
                raise ValueError("telemetry contains rows for another variant")
            parsed_rows: list[dict[str, Any]] = []
            for row in rows:
                parsed_rows.append(
                    {
                        **row,
                        "epoch_seconds": float(row["epoch_seconds"]),
                        "gpu_index": int(row["gpu_index"]),
                        "memory_used_mib": float(row["memory_used_mib"]),
                        "memory_total_mib": float(row["memory_total_mib"]),
                        "utilization_gpu_percent": float(
                            row["utilization_gpu_percent"]
                        ),
                        "power_draw_watts": float(row["power_draw_watts"]),
                    }
                )
            baseline_memory = {
                int(row["gpu_index"]): float(row["memory_used_mib"])
                for row in parsed_rows
                if row["phase"] == "baseline"
            }
            observed_summary = summarize_gpu_samples(parsed_rows, baseline_memory)
            for field in (
                "sample_rows",
                "gpus",
                "max_peak_memory_mib",
                "sum_per_gpu_peak_memory_mib",
            ):
                if gpu.get(field) != observed_summary[field]:
                    raise ValueError(
                        f"stored GPU summary field {field} does not match telemetry"
                    )
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"GPU telemetry CSV is malformed: {exc}")

    stored_artifacts = previous.get("artifacts") or {}
    for name, path in (
        ("server_log", variant_dir / "server.log"),
        ("smoke_log", variant_dir / "smoke.log"),
    ):
        expected = stored_artifacts.get(name)
        if not isinstance(expected, dict) or not path.is_file():
            reasons.append(f"{name} evidence is missing")
            continue
        observed = file_evidence(path)
        if (
            expected.get("bytes") != observed["bytes"]
            or expected.get("sha256") != observed["sha256"]
        ):
            reasons.append(f"{name} hash/size does not match variant_result.json")
    return reasons


def run_variant(
    args: argparse.Namespace,
    spec: VariantSpec,
    run_dir: Path,
) -> dict[str, Any]:
    variant_dir = run_dir / spec.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    result_path = variant_dir / "variant_result.json"
    previous: dict[str, Any] = {}
    if args.resume and result_path.is_file():
        previous = load_json(result_path)
        cached_datasets, cache_reasons = validated_cached_datasets(
            previous, args, spec, variant_dir
        )
        if previous.get("status") == "PASSED":
            cache_reasons.extend(
                validate_completed_variant_artifacts(
                    previous, variant_dir, args.gpu_list
                )
            )
            if not cache_reasons and len(cached_datasets) == len(args.datasets):
                print(
                    f"[{spec.name}] already passed; raw outputs and telemetry verified: "
                    f"{result_path}"
                )
                previous["datasets"] = cached_datasets
                return previous
        if cache_reasons:
            print(
                f"[{spec.name}] cached variant is not fully reusable: "
                + "; ".join(cache_reasons),
                flush=True,
            )
        previous = {"datasets": cached_datasets}

    require_idle_gpus(args.gpu_list)
    if not port_is_free(args.port):
        raise RuntimeError(f"port {args.port} is already in use")

    result: dict[str, Any] = {
        "variant": spec.name,
        "status": "RUNNING",
        "prune_fraction": spec.prune_fraction,
        "expected_experts": spec.expected_experts,
        "checkpoint": asdict(spec.checkpoint),
        "protocol_fingerprint": args.protocol_fingerprint,
        "started_at": utc_now(),
        "datasets": previous.get("datasets", {}),
    }
    atomic_write_json(result_path, result)

    server_log = variant_dir / "server.log"
    monitor = GpuMonitor(
        variant_dir / "gpu_telemetry.csv",
        spec.name,
        args.gpu_list,
        args.monitor_interval,
    )
    command = server_command(args, spec)
    result["server_command"] = command
    atomic_write_json(result_path, result)
    process: subprocess.Popen[Any] | None = None
    variant_started = time.monotonic()
    try:
        monitor.start()
        # Close the gap between the initial exclusivity gate and the actual
        # server launch. This remains read-only and never terminates a process.
        require_idle_gpus(args.gpu_list)
        with server_log.open("w", encoding="utf-8") as server_handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=runtime_environment(args),
                text=True,
                stdout=server_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        startup_started = time.monotonic()
        served_model = wait_for_server(process, args.port, args.startup_timeout)
        result["served_model"] = served_model
        result["startup_seconds"] = round(time.monotonic() - startup_started, 6)
        atomic_write_json(result_path, result)

        smoke_started = time.monotonic()
        run_logged(
            [
                str(args.server_python),
                str(SMOKE_CLIENT),
                "--base-url",
                f"http://127.0.0.1:{args.port}/v1",
                "--timeout",
                str(args.request_timeout),
            ],
            variant_dir / "smoke.log",
            timeout=args.request_timeout + 60,
            label=f"{spec.name} smoke",
        )
        result["smoke_seconds"] = round(time.monotonic() - smoke_started, 6)
        atomic_write_json(result_path, result)

        for dataset in args.datasets:
            output_path = variant_dir / "evaluation" / f"{dataset}.jsonl"
            eval_log = variant_dir / "evaluation" / f"{dataset}.log"
            if (
                args.resume
                and dataset in result["datasets"]
                and output_path.is_file()
            ):
                print(f"[{spec.name}] reusing hash-verified dataset {dataset}")
                continue
            eval_started = time.monotonic()
            eval_command = [
                str(args.eval_python),
                str(EVAL_CLIENT),
                "--data-name",
                dataset,
                "--target-path",
                str(output_path.parent),
                "--output-file",
                str(output_path),
                "--base-url",
                f"http://127.0.0.1:{args.port}/v1",
                "--model",
                served_model,
                "--max-tokens",
                str(args.max_tokens),
                "--workers",
                str(args.workers),
                "--repeats",
                str(args.repeats),
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
                "--timeout",
                str(args.request_timeout),
                "--retries",
                str(args.retries),
                "--thinking",
                "--resume",
            ]
            run_logged(
                eval_command,
                eval_log,
                label=f"{spec.name} {dataset}",
                stream_output=True,
            )
            wall_seconds = time.monotonic() - eval_started
            result["datasets"][dataset] = parse_evaluation_output(
                output_path,
                wall_seconds,
                expected_dataset=dataset,
                expected_model=served_model,
                expected_total=args.dataset_totals[dataset] * args.repeats,
            )
            atomic_write_json(result_path, result)

        dataset_values = list(result["datasets"].values())
        result["aggregate"] = {
            "macro_accuracy": round(
                statistics.fmean(item["accuracy"] for item in dataset_values), 6
            ),
            "weighted_accuracy": round(
                sum(item["correct"] for item in dataset_values)
                / sum(item["total"] for item in dataset_values)
                * 100,
                6,
            ),
            "total_correct": sum(item["correct"] for item in dataset_values),
            "total_samples": sum(item["total"] for item in dataset_values),
            "evaluation_wall_seconds": round(
                sum(item["wall_seconds"] for item in dataset_values), 6
            ),
            "completion_tokens": sum(
                item["completion_tokens"] for item in dataset_values
            ),
        }
        result["status"] = "PASSED"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["server_log_tail"] = tail_text(server_log)
        if isinstance(exc, KeyboardInterrupt):
            result["interrupted"] = True
    finally:
        cleanup_errors: list[str] = []
        if process is not None:
            try:
                result["server_exit_status"] = stop_process_group(process)
            except Exception as exc:
                cleanup_errors.append(f"server process-group cleanup failed: {exc}")
        try:
            monitor.stop()
        except Exception as exc:
            cleanup_errors.append(f"GPU telemetry shutdown failed: {exc}")
        if monitor._thread.is_alive():
            result["gpu"] = {
                "trace_file": str(monitor.path),
                "errors": cleanup_errors,
            }
        else:
            result["gpu"] = monitor.summary()
            if result["gpu"]["errors"]:
                cleanup_errors.append(
                    "GPU telemetry sampling failed: "
                    + "; ".join(result["gpu"]["errors"])
                )
        result["artifacts"] = {}
        for name, path in (
            ("server_log", server_log),
            ("smoke_log", variant_dir / "smoke.log"),
        ):
            if path.is_file():
                result["artifacts"][name] = file_evidence(path)
        if cleanup_errors:
            previous_error = result.get("error")
            result["status"] = "FAILED"
            result["error"] = "; ".join(
                ([str(previous_error)] if previous_error else []) + cleanup_errors
            )
        result["elapsed_seconds"] = round(time.monotonic() - variant_started, 6)
        result["ended_at"] = utc_now()
        atomic_write_json(result_path, result)

    if result["status"] != "PASSED":
        raise RuntimeError(f"{spec.name} failed; see {result_path}: {result['error']}")
    return result


def comparison_rows(results: list[dict[str, Any]], datasets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full = next((item for item in results if item["variant"] == "full"), None)
    full_macro = (full or {}).get("aggregate", {}).get("macro_accuracy")
    for result in results:
        aggregate = result.get("aggregate") or {}
        gpu = result.get("gpu") or {}
        checkpoint = result["checkpoint"]
        row: dict[str, Any] = {
            "variant": result["variant"],
            "status": result["status"],
            "prune_percent": round(float(result["prune_fraction"]) * 100),
            "routed_experts": checkpoint["routed_experts"],
            "hash_routed_experts": checkpoint["hash_routed_experts"],
            "main_layer_slot_prune_percent": round(
                float(checkpoint["main_layer_slot_prune_fraction"]) * 100, 6
            ),
            "checkpoint_gib": round(checkpoint["checkpoint_bytes"] / 2**30, 6),
            "startup_seconds": result.get("startup_seconds"),
            "smoke_seconds": result.get("smoke_seconds"),
            "evaluation_wall_seconds": aggregate.get("evaluation_wall_seconds"),
            "total_elapsed_seconds": result.get("elapsed_seconds"),
            "macro_accuracy": aggregate.get("macro_accuracy"),
            "weighted_accuracy": aggregate.get("weighted_accuracy"),
            "accuracy_delta_vs_full": (
                round(aggregate["macro_accuracy"] - full_macro, 6)
                if full_macro is not None and aggregate.get("macro_accuracy") is not None
                else None
            ),
            "max_peak_memory_mib": gpu.get("max_peak_memory_mib"),
            "sum_per_gpu_peak_memory_mib": gpu.get(
                "sum_per_gpu_peak_memory_mib"
            ),
        }
        for dataset in datasets:
            row[f"{dataset}_accuracy"] = (
                result.get("datasets", {}).get(dataset, {}).get("accuracy")
            )
        rows.append(row)
    return rows


def write_summary(run_dir: Path, manifest: dict[str, Any], results: list[dict[str, Any]]) -> None:
    datasets = list(manifest["settings"]["datasets"])
    rows = comparison_rows(results, datasets)
    atomic_write_json(
        run_dir / "summary.json",
        {"manifest": manifest, "comparison": rows, "results": results},
    )
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["variant"])
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "Variant",
        "Status",
        "Prune",
        "Dynamic experts",
        "Hash experts",
        "Main slots pruned",
        "Checkpoint GiB",
        *datasets,
        "Macro Acc.",
        "Acc. delta",
        "Eval seconds",
        "Peak HBM/GPU MiB",
    ]
    lines = [
        "# EASY-EP reproduction report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- GPUs: `{manifest['settings']['gpu_list']}`; TP={manifest['settings']['tp']}",
        f"- Repeats: {manifest['settings']['repeats']}; max tokens: {manifest['settings']['max_tokens']}",
        "- Results include generation/scoring wall time and sampled NVIDIA-SMI telemetry.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["variant"],
            row["status"],
            f"{row['prune_percent']}%",
            str(row["routed_experts"]),
            str(row["hash_routed_experts"]),
            f"{row['main_layer_slot_prune_percent']}%",
            str(row["checkpoint_gib"]),
            *[str(row.get(f"{name}_accuracy")) for name in datasets],
            str(row.get("macro_accuracy")),
            str(row.get("accuracy_delta_vs_full")),
            str(row.get("evaluation_wall_seconds")),
            str(row.get("max_peak_memory_mib")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This matrix reports accuracy, request latency, wall time, checkpoint bytes, and HBM telemetry.",
            "It does not by itself establish TTFT, TPOT, P99, goodput, or the paper's 8-GPU throughput claim.",
            "V4 hash layers 0..2 keep all 256 experts; only dynamic layers 3..42 are physically pruned.",
            "",
        ]
    )
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def git_tracked_state() -> dict[str, Any]:
    parts: list[bytes] = []
    for command in (
        ["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD", "--"],
        ["git", "-C", str(REPO_ROOT), "diff", "--binary", "--cached", "HEAD", "--"],
    ):
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"cannot fingerprint tracked repository changes: "
                f"{completed.stderr.decode(errors='replace').strip()}"
            )
        parts.append(completed.stdout)
    payload = b"\0".join(parts)
    return {
        "dirty": bool(payload.replace(b"\0", b"")),
        "diff_sha256": hashlib.sha256(payload).hexdigest(),
    }


def python_runtime_identity(python: Path, modules: tuple[str, ...]) -> dict[str, Any]:
    code = r'''
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

result = {"python": platform.python_version(), "executable": sys.executable, "modules": {}}
for name in sys.argv[1:]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        result["modules"][name] = None
        continue
    entry = {"origin": spec.origin}
    try:
        entry["version"] = importlib.metadata.version(name.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError:
        entry["version"] = None
    if spec.origin and Path(spec.origin).is_file():
        entry["origin_sha256"] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    if name == "sglang" and spec.origin:
        root = Path(spec.origin).parent
        patched = {}
        for relative in ("srt/configs/deepseek_v4.py", "srt/models/deepseek_v2.py"):
            path = root / relative
            if not path.is_file():
                raise SystemExit(f"missing required SGLang runtime file: {path}")
            patched[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["critical_files_sha256"] = patched
    result["modules"][name] = entry
print(json.dumps(result, sort_keys=True))
'''
    completed = subprocess.run(
        [str(python), "-c", code, *modules],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot fingerprint Python runtime {python}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"runtime fingerprint returned invalid JSON: {completed.stdout!r}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"runtime fingerprint is not an object: {value!r}")
    return value


def runtime_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "server": python_runtime_identity(args.server_python, ("sglang", "torch")),
        "evaluator": python_runtime_identity(
            args.eval_python, ("symeval", "math_verify")
        ),
        "evaluation_client_sha256": sha256_file(EVAL_CLIENT),
        "smoke_client_sha256": sha256_file(SMOKE_CLIENT),
        "datasets_sha256": {
            name: sha256_file(REPO_ROOT / "evaluation" / "dataset" / f"{name}.jsonl")
            for name in args.datasets
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate full, 25%-pruned, and 50%-pruned checkpoints"
    )
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--prune25-model", type=Path, required=True)
    parser.add_argument("--prune50-model", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "easyep_reproduction",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--server-python", type=Path, default=Path("/opt/sglang-v4/bin/python"))
    parser.add_argument("--eval-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-list", default="4,5,6,7")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--port", type=int, default=60000)
    parser.add_argument("--moe-backend", default="marlin")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--startup-timeout", type=float, default=3600)
    parser.add_argument("--watchdog-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--monitor-interval", type=float, default=2.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rehash-checkpoints",
        action="store_true",
        help=(
            "ignore the local size/mtime/inode hash cache and recompute SHA-256 for "
            "every checkpoint shard"
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "tp": args.tp,
        "port": args.port,
        "repeats": args.repeats,
        "workers": args.workers,
        "max_running_requests": args.max_running_requests,
        "max_tokens": args.max_tokens,
        "context_length": args.context_length,
        "startup_timeout": args.startup_timeout,
        "watchdog_timeout": args.watchdog_timeout,
        "request_timeout": args.request_timeout,
        "monitor_interval": args.monitor_interval,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"these arguments must be positive: {invalid}")
    if args.gpu_list != "4,5,6,7" or args.tp != 4:
        raise ValueError(
            "this evaluation protocol is restricted to physical GPUs 4,5,6,7 "
            "with TP=4"
        )
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("datasets must not contain duplicates")
    if args.retries < 0:
        raise ValueError("retries must be non-negative")
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    if not 0 < args.mem_fraction_static < 1:
        raise ValueError("mem-fraction-static must be in (0, 1)")
    if args.max_tokens >= args.context_length:
        raise ValueError("context-length must exceed max-tokens to leave room for the prompt")
    if not args.server_python.is_file():
        raise FileNotFoundError(f"server Python does not exist: {args.server_python}")
    if not args.eval_python.is_file():
        raise FileNotFoundError(f"evaluation Python does not exist: {args.eval_python}")
    if not os.access(args.server_python, os.X_OK):
        raise PermissionError(f"server Python is not executable: {args.server_python}")
    if not os.access(args.eval_python, os.X_OK):
        raise PermissionError(f"evaluation Python is not executable: {args.eval_python}")
    dataset_totals: dict[str, int] = {}
    for dataset in args.datasets:
        path = REPO_ROOT / "evaluation" / "dataset" / f"{dataset}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"evaluation dataset does not exist: {path}")
        with path.open(encoding="utf-8") as handle:
            total = sum(1 for line in handle if line.strip())
        if total < 1:
            raise ValueError(f"evaluation dataset is empty: {path}")
        dataset_totals[dataset] = total
    args.dataset_totals = dataset_totals
    evaluator_check = subprocess.run(
        [
            str(args.eval_python),
            "-c",
            (
                "import importlib.util as u; import sys; "
                "sys.exit(0 if (u.find_spec('symeval') or u.find_spec('math_verify')) else 1)"
            ),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if evaluator_check.returncode != 0:
        raise RuntimeError(
            f"{args.eval_python} has neither symeval nor math_verify; install "
            "requirements-eval.txt before starting the multi-hour model run"
        )


def build_protocol_fingerprint(
    settings: dict[str, Any], specs: list[VariantSpec], git_revision: str
) -> str:
    non_scientific_settings = {
        "continue_on_error",
        "dry_run",
        "results_root",
        "resume",
        "rehash_checkpoints",
        "run_id",
    }
    payload = {
        "settings": {
            key: value
            for key, value in settings.items()
            if key not in non_scientific_settings
        },
        "variants": [asdict(spec) for spec in specs],
        "git_revision": git_revision,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    results_root = args.results_root.expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    specs = build_variant_specs(
        args.full_model,
        args.prune25_model,
        args.prune50_model,
        hash_cache_path=results_root / ".checkpoint_hash_cache.json",
        rehash=args.rehash_checkpoints,
    )
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in {"full_model", "prune25_model", "prune50_model"}
    }
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in settings.items()
    }
    current_git_commit = git_commit()
    settings["git_tracked_state"] = git_tracked_state()
    settings["runtime_identity"] = runtime_identity(args)
    protocol_fingerprint = build_protocol_fingerprint(
        settings, specs, current_git_commit
    )
    args.protocol_fingerprint = protocol_fingerprint
    manifest_path = run_dir / "manifest.json"
    existing_manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        existing_manifest = load_json(manifest_path)
        previous_fingerprint = existing_manifest.get("protocol_fingerprint")
        if previous_fingerprint != protocol_fingerprint:
            raise ValueError(
                f"RUN_ID={run_id} already contains a different experiment protocol; "
                "use its original settings to resume or choose a new RUN_ID"
            )
    manifest = {
        "run_id": run_id,
        "created_at": (existing_manifest or {}).get("created_at", utc_now()),
        "last_invoked_at": utc_now(),
        "git_commit": current_git_commit,
        "settings": settings,
        "variants": [asdict(spec) for spec in specs],
        "protocol_fingerprint": protocol_fingerprint,
        "downloads_enabled": False,
    }
    atomic_write_json(manifest_path, manifest)
    print(f"run directory: {run_dir}")
    for spec in specs:
        print(
            f"{spec.name}: prune={spec.prune_fraction:.0%}, "
            f"experts={spec.checkpoint.routed_experts}, "
            f"checkpoint={spec.checkpoint.checkpoint_bytes / 2**30:.2f} GiB, "
            f"path={spec.checkpoint.path}"
        )
    if args.dry_run:
        print("dry-run validation passed; no server was launched")
        return 0

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for spec in specs:
        print(f"[{spec.name}] starting")
        try:
            result = run_variant(args, spec, run_dir)
        except Exception as exc:
            failures.append(f"{spec.name}: {exc}")
            result_path = run_dir / spec.name / "variant_result.json"
            if result_path.is_file():
                results.append(load_json(result_path))
            if not args.continue_on_error:
                break
        else:
            results.append(result)
            print(f"[{spec.name}] passed")
        write_summary(run_dir, manifest, results)

    write_summary(run_dir, manifest, results)
    print(f"summary: {run_dir / 'REPORT.md'}")
    if failures:
        for failure in failures:
            print(f"[ERROR] {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
