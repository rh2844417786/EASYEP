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
import threading
import time
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CLIENT = REPO_ROOT / "evaluation" / "run_sglang.py"
SMOKE_CLIENT = REPO_ROOT / "scripts" / "smoke_v4_server.py"
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


def inspect_checkpoint(path: Path) -> CheckpointInfo:
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
    checkpoint_bytes = sum((path / name).stat().st_size for name in shard_names)
    fingerprint = sha256_parts((config_bytes, b"\0", index_bytes))[:20]
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
        fingerprint=fingerprint,
    )


def build_variant_specs(
    full_model: Path,
    prune25_model: Path,
    prune50_model: Path,
) -> list[VariantSpec]:
    checkpoints = [
        inspect_checkpoint(full_model),
        inspect_checkpoint(prune25_model),
        inspect_checkpoint(prune50_model),
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def parse_evaluation_output(path: Path, wall_seconds: float) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if record.get("type") == "sample":
                samples.append(record)
            elif record.get("type") == "summary":
                summary = record
    if summary is None:
        raise ValueError(f"evaluation output has no summary record: {path}")

    latencies = [float(item["latency_seconds"]) for item in samples]
    prompt_tokens = 0
    completion_tokens = 0
    for sample in samples:
        usage = sample.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
    return {
        "accuracy": float(summary["accuracy"]),
        "correct": int(summary["correct"]),
        "total": int(summary["total"]),
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
    }


class GpuMonitor:
    FIELDS = (
        "timestamp_utc",
        "epoch_seconds",
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
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join(timeout=max(10.0, self.interval * 3))

    def _run(self) -> None:
        query = (
            "index,name,memory.used,memory.total,utilization.gpu,power.draw"
        )
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            while not self._stop.is_set():
                epoch = time.time()
                try:
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
                    for row in csv.reader(completed.stdout.splitlines()):
                        if len(row) != 6:
                            raise ValueError(f"unexpected nvidia-smi row: {row}")
                        sample = {
                            "timestamp_utc": datetime.fromtimestamp(
                                epoch, timezone.utc
                            ).isoformat(),
                            "epoch_seconds": round(epoch, 6),
                            "variant": self.variant,
                            "gpu_index": int(row[0].strip()),
                            "gpu_name": row[1].strip(),
                            "memory_used_mib": float(row[2].strip()),
                            "memory_total_mib": float(row[3].strip()),
                            "utilization_gpu_percent": float(row[4].strip()),
                            "power_draw_watts": float(row[5].strip()),
                        }
                        self.samples.append(sample)
                        writer.writerow(sample)
                    handle.flush()
                except Exception as exc:  # telemetry failure must not kill evaluation
                    self.errors.append(f"{utc_now()}: {exc}")
                self._stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        by_gpu: dict[int, list[dict[str, Any]]] = {}
        for sample in self.samples:
            by_gpu.setdefault(int(sample["gpu_index"]), []).append(sample)
        gpu_rows: list[dict[str, Any]] = []
        for gpu_index, samples in sorted(by_gpu.items()):
            used = [float(item["memory_used_mib"]) for item in samples]
            utilization = [
                float(item["utilization_gpu_percent"]) for item in samples
            ]
            power = [float(item["power_draw_watts"]) for item in samples]
            gpu_rows.append(
                {
                    "gpu_index": gpu_index,
                    "gpu_name": samples[0]["gpu_name"],
                    "samples": len(samples),
                    "baseline_memory_mib": used[0],
                    "peak_memory_mib": max(used),
                    "peak_memory_delta_mib": max(used) - used[0],
                    "mean_memory_mib": round(statistics.fmean(used), 3),
                    "peak_utilization_percent": max(utilization),
                    "mean_utilization_percent": round(
                        statistics.fmean(utilization), 3
                    ),
                    "mean_power_watts": round(statistics.fmean(power), 3),
                }
            )
        return {
            "trace_file": str(self.path),
            "interval_seconds": self.interval,
            "sample_rows": len(self.samples),
            "errors": self.errors,
            "gpus": gpu_rows,
            "max_peak_memory_mib": max(
                (row["peak_memory_mib"] for row in gpu_rows), default=None
            ),
            "sum_per_gpu_peak_memory_mib": sum(
                row["peak_memory_mib"] for row in gpu_rows
            ),
        }


def port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def fetch_models(port: int, timeout: float = 3.0) -> dict[str, Any]:
    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(process: subprocess.Popen[Any], port: int, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    next_heartbeat = started + 60
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"SGLang exited before readiness with status {return_code}")
        try:
            payload = fetch_models(port)
        except (OSError, URLError, ValueError, json.JSONDecodeError):
            now = time.monotonic()
            if now >= next_heartbeat:
                print(
                    f"[server] still loading after {now - started:.0f}s; "
                    f"waiting for http://127.0.0.1:{port}/v1/models",
                    flush=True,
                )
                next_heartbeat = now + 60
            time.sleep(10)
            continue
        models = payload.get("data") or []
        if models and models[0].get("id"):
            return str(models[0]["id"])
        time.sleep(10)
    raise TimeoutError(f"SGLang was not ready within {timeout} seconds")


def stop_process_group(process: subprocess.Popen[Any], timeout: float = 30.0) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(process.poll() or 0)
    try:
        return int(process.wait(timeout=timeout))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return int(process.wait(timeout=10))


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
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
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
                    print(
                        f"[{label}] still running after "
                        f"{time.monotonic() - started:.0f}s; log={log_path}",
                        flush=True,
                    )
        except BaseException:
            if process.poll() is None:
                stop_process_group(process)
            raise
    if return_code != 0:
        raise RuntimeError(
            f"command failed with status {return_code}: {' '.join(command)}\n"
            f"last log lines:\n{tail_text(log_path)}"
        )


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
        if previous.get("status") == "PASSED":
            print(f"[{spec.name}] already passed; reusing {result_path}")
            return previous

    if not port_is_free(args.port):
        raise RuntimeError(f"port {args.port} is already in use")

    result: dict[str, Any] = {
        "variant": spec.name,
        "status": "RUNNING",
        "prune_fraction": spec.prune_fraction,
        "expected_experts": spec.expected_experts,
        "checkpoint": asdict(spec.checkpoint),
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
                print(f"[{spec.name}] reusing completed dataset {dataset}")
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
            )
            wall_seconds = time.monotonic() - eval_started
            result["datasets"][dataset] = parse_evaluation_output(
                output_path, wall_seconds
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
        if process is not None:
            result["server_exit_status"] = stop_process_group(process)
        time.sleep(3)
        monitor.stop()
        result["gpu"] = monitor.summary()
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
    if not 0 < args.mem_fraction_static < 1:
        raise ValueError("mem-fraction-static must be in (0, 1)")
    if args.max_tokens >= args.context_length:
        raise ValueError("context-length must exceed max-tokens to leave room for the prompt")
    if not args.server_python.is_file():
        raise FileNotFoundError(f"server Python does not exist: {args.server_python}")
    if not args.eval_python.is_file():
        raise FileNotFoundError(f"evaluation Python does not exist: {args.eval_python}")
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
    specs = build_variant_specs(
        args.full_model,
        args.prune25_model,
        args.prune50_model,
    )
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root.expanduser().resolve() / run_id
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
    protocol_fingerprint = build_protocol_fingerprint(
        settings, specs, current_git_commit
    )
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
