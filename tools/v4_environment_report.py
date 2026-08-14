#!/usr/bin/env python3
"""Collect a redacted DeepSeek-V4/SGLang environment report in Markdown.

The collector intentionally uses only the Python standard library. It is safe
to run before SGLang, PyTorch, CUDA Python packages, or compiler tools are
installed. Missing components are recorded in the report instead of aborting
the collection.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from typing import Iterable


PACKAGE_DISTRIBUTIONS = (
    "sglang",
    "torch",
    "triton",
    "transformers",
    "flashinfer-python",
    "flashinfer-cubin",
    "safetensors",
    "cuda-python",
    "nvidia-nccl-cu12",
    "tilelang",
    "deep-gemm",
    "numpy",
)

MODEL_CONFIG_KEYS = (
    "architectures",
    "model_type",
    "torch_dtype",
    "expert_dtype",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_hash_layers",
    "n_routed_experts",
    "num_experts_per_tok",
    "first_k_dense_replace",
    "moe_layer_freq",
    "scoring_func",
    "topk_method",
    "routed_scaling_factor",
)

SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_REQUIRE_CUDA",
    "MODEL_PATH",
    "PYTHONNOUSERSITE",
    "VIRTUAL_ENV",
    "CONDA_DEFAULT_ENV",
)

REQUIRED_SGLANG_FLAGS = (
    "--model-path",
    "--tp",
    "--moe-runner-backend",
    "--reasoning-parser",
    "--tool-call-parser",
)

TORCH_CUDA_SMOKE = r"""
import json
import sys

result = {"ok": False}
try:
    import torch
    result.update({
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    })
    if not result["cuda_available"]:
        raise RuntimeError("torch.cuda.is_available() is False")
    for logical_index in range(result["device_count"]):
        props = torch.cuda.get_device_properties(logical_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(logical_index)
        x = torch.ones((64, 64), device=logical_index, dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize(logical_index)
        result["devices"].append({
            "logical_index": logical_index,
            "name": props.name,
            "capability": [props.major, props.minor],
            "free_mib": round(free_bytes / 1024**2),
            "total_mib": round(total_bytes / 1024**2),
            "matmul_value": float(y[0, 0].item()),
        })
        del x, y
    result["ok"] = True
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result["ok"] else 1)
"""


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        parts = [part.rstrip() for part in (self.stdout, self.stderr) if part.strip()]
        return "\n".join(parts)


@dataclass
class Finding:
    status: str
    check: str
    detail: str


def sanitize(text: str) -> str:
    """Redact common credentials without dumping or inspecting the environment."""
    patterns = (
        (r"(https?://)[^/@\s:]+:[^/@\s]+@", r"\1<redacted>@"),
        (r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "<redacted-github-token>"),
        (r"\bgh[opusr]_[A-Za-z0-9_]{20,}\b", "<redacted-github-token>"),
        (r"\bhf_[A-Za-z0-9]{20,}\b", "<redacted-hf-token>"),
        (r"(?i)(authorization:\s*(?:bearer|token)\s+)[^\s]+", r"\1<redacted>"),
    )
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


def truncate(text: str, limit: int = 20_000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... <truncated {len(text) - limit} characters> ...\n{text[-half:]}"


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    header_values = [str(item) for item in headers]

    def cell(value: object) -> str:
        return sanitize(str(value)).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(cell(value) for value in header_values) + " |",
        "| " + " | ".join("---" for _ in header_values) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def run_command(
    command: list[str],
    *,
    timeout: int = 60,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=cwd,
            check=False,
        )
        return CommandResult(command, completed.returncode, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        return CommandResult(command, 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(command, 124, stdout, stderr, timed_out=True)


class Report:
    def __init__(self, *, model_path: Path | None, gpu_ids: list[int]) -> None:
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.findings: list[Finding] = []
        self.sections: list[str] = []

    def finding(self, status: str, check: str, detail: str) -> None:
        self.findings.append(Finding(status, check, detail))

    def section(self, title: str, body: str) -> None:
        self.sections.append(f"## {title}\n\n{body.strip()}\n")

    def command_result(self, title: str, result: CommandResult) -> None:
        output = truncate(sanitize(result.combined_output or "<no output>"))
        timeout_note = " (timed out)" if result.timed_out else ""
        body = (
            f"Exit code: `{result.returncode}`{timeout_note}\n\n"
            "````text\n"
            f"$ {sanitize(shlex.join(result.command))}\n"
            f"{output}\n"
            "````"
        )
        self.section(title, body)

    def render(self) -> str:
        generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        model_display = str(self.model_path) if self.model_path else "<not provided>"
        summary = markdown_table(
            ("Status", "Check", "Detail"),
            ((finding.status, finding.check, finding.detail) for finding in self.findings),
        )
        counts = {
            status: sum(finding.status == status for finding in self.findings)
            for status in ("PASS", "WARN", "FAIL", "INFO")
        }
        header = f"""# DeepSeek-V4 / SGLang environment report

- Generated: `{generated}`
- Host: `{sanitize(socket.gethostname())}`
- User: `{sanitize(getpass.getuser())}`
- Requested GPU IDs: `{','.join(map(str, self.gpu_ids))}`
- Model path: `{sanitize(model_display)}`
- Summary: PASS={counts['PASS']}, WARN={counts['WARN']}, FAIL={counts['FAIL']}, INFO={counts['INFO']}

This report contains an allowlisted set of diagnostics. It does not dump the
full process environment, authentication state, Docker inspect output, or model
weights. Common token formats and URL credentials are redacted.

## Check summary

{summary}
"""
        return sanitize(header + "\n" + "\n".join(self.sections)).rstrip() + "\n"


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def parse_gpu_inventory(output: str, fields: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for values in csv.reader(io.StringIO(output)):
        if not values or not any(value.strip() for value in values):
            continue
        if len(values) != len(fields):
            raise ValueError(f"Unexpected nvidia-smi row: {values!r}")
        rows.append(dict(zip(fields, (value.strip() for value in values))))
    return rows


def collect_system(report: Report, repo_root: Path, timeout: int) -> None:
    container_markers = []
    if Path("/.dockerenv").exists():
        container_markers.append("/.dockerenv")
    cgroup = ""
    try:
        cgroup = Path("/proc/1/cgroup").read_text(errors="replace")
    except OSError:
        pass
    if re.search(r"docker|containerd|kubepods", cgroup, re.IGNORECASE):
        container_markers.append("/proc/1/cgroup")

    safe_environment = [(key, os.environ.get(key, "<unset>")) for key in SAFE_ENV_KEYS]
    identity = markdown_table(
        ("Field", "Value"),
        (
            ("Platform", platform.platform()),
            ("Machine", platform.machine()),
            ("Python", sys.version.replace("\n", " ")),
            ("Python executable", sys.executable),
            ("Repository", repo_root),
            ("Container markers", ", ".join(container_markers) or "none detected"),
        ),
    )
    report.section("Runtime identity", identity + "\n\n### Allowlisted environment\n\n" + markdown_table(("Variable", "Value"), safe_environment))
    report.finding("INFO", "Container detection", ", ".join(container_markers) or "no standard marker detected")

    for title, command in (
        ("Kernel", ["uname", "-a"]),
        ("Container base OS", ["bash", "-lc", "test -r /etc/os-release && cat /etc/os-release"]),
        ("CPU and memory", ["bash", "-lc", "command -v lscpu >/dev/null && lscpu; command -v free >/dev/null && free -h"]),
        ("Shared memory", ["df", "-h", "/dev/shm"]),
        ("Resource limits", ["bash", "-lc", "ulimit -a"]),
    ):
        report.command_result(title, run_command(command, timeout=timeout))

    shm_path = Path("/dev/shm")
    if shm_path.exists():
        stats = os.statvfs(shm_path)
        shm_gib = stats.f_frsize * stats.f_blocks / 1024**3
        if shm_gib >= 16:
            report.finding("PASS", "/dev/shm capacity", f"{shm_gib:.1f} GiB")
        else:
            report.finding("WARN", "/dev/shm capacity", f"only {shm_gib:.2f} GiB; use --shm-size 32g")
    else:
        report.finding("WARN", "/dev/shm capacity", "/dev/shm does not exist")


def collect_repository(report: Report, repo_root: Path, timeout: int) -> None:
    for title, command in (
        ("Git revision", ["git", "log", "-1", "--format=%H %s"]),
        ("Git branch and status", ["git", "status", "-sb"]),
    ):
        report.command_result(title, run_command(command, timeout=timeout, cwd=repo_root))


def collect_python_environment(report: Report, repo_root: Path, gpu_ids: list[int], timeout: int) -> bool:
    package_rows = [(name, installed_version(name) or "MISSING") for name in PACKAGE_DISTRIBUTIONS]
    report.section("Selected Python distributions", markdown_table(("Distribution", "Version"), package_rows))

    sglang_version = installed_version("sglang")
    spec = importlib.util.find_spec("sglang")
    sglang_origin = spec.origin if spec and spec.origin else None
    if not sglang_version or not sglang_origin:
        report.finding("FAIL", "SGLang installation", "distribution or importable package is missing")
        sglang_ready = False
    elif str(sglang_origin).startswith(str(repo_root / "sglang")):
        report.finding("FAIL", "SGLang installation", f"resolved legacy repository copy: {sglang_origin}")
        sglang_ready = False
    else:
        report.finding("PASS", "SGLang installation", f"{sglang_version} at {sglang_origin}")
        sglang_ready = True
        parsed_version = version_tuple(sglang_version)
        if parsed_version and parsed_version < (0, 5, 15):
            report.finding("FAIL", "SGLang V4 generation", f"{sglang_version} is older than the repository's V4 baseline")

    pip_check = run_command([sys.executable, "-m", "pip", "check"], timeout=timeout)
    report.command_result("Python dependency consistency", pip_check)
    if pip_check.returncode == 0:
        report.finding("PASS", "pip check", "no broken requirements reported")
    else:
        report.finding("WARN", "pip check", f"exit code {pip_check.returncode}; inspect dependency output")

    if sglang_ready:
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        help_result = run_command(
            [sys.executable, "-m", "sglang.launch_server", "--help"],
            timeout=max(timeout, 90),
            env=child_env,
        )
        help_text = help_result.combined_output
        flag_rows = [(flag, "yes" if flag in help_text else "NO") for flag in REQUIRED_SGLANG_FLAGS]
        report.section(
            "SGLang launch interface",
            f"Command exit code: `{help_result.returncode}`\n\n" + markdown_table(("Required flag", "Present"), flag_rows),
        )
        if help_result.returncode != 0:
            report.command_result("SGLang help failure", help_result)
            report.finding("FAIL", "SGLang launch module", f"--help exited {help_result.returncode}")
        elif all(flag in help_text for flag in REQUIRED_SGLANG_FLAGS):
            report.finding("PASS", "SGLang launch module", "required V4 launch flags are present")
        else:
            missing = [flag for flag in REQUIRED_SGLANG_FLAGS if flag not in help_text]
            report.finding("FAIL", "SGLang launch module", "missing flags: " + ", ".join(missing))
    return sglang_ready


def collect_gpu_environment(report: Report, gpu_ids: list[int], timeout: int, skip_cuda_smoke: bool) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        report.finding("FAIL", "NVIDIA driver visibility", "nvidia-smi is not on PATH")
        report.section("NVIDIA GPU inventory", "`nvidia-smi` was not found.")
        return

    query_options = (
        ["index", "name", "uuid", "driver_version", "compute_cap", "memory.total", "memory.used", "memory.free"],
        ["index", "name", "uuid", "driver_version", "memory.total", "memory.used", "memory.free"],
    )
    inventory: list[dict[str, str]] = []
    query_result: CommandResult | None = None
    for fields in query_options:
        query_result = run_command(
            [nvidia_smi, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            timeout=timeout,
        )
        if query_result.returncode == 0:
            try:
                inventory = parse_gpu_inventory(query_result.stdout, fields)
            except ValueError:
                inventory = []
            if inventory:
                break
    assert query_result is not None
    report.command_result("NVIDIA GPU inventory", query_result)

    if not inventory:
        report.finding("FAIL", "NVIDIA driver visibility", "nvidia-smi query failed or was not parseable")
        return

    report.finding("PASS", "NVIDIA driver visibility", f"nvidia-smi reported {len(inventory)} GPU(s)")
    by_index = {int(row["index"]): row for row in inventory}
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in by_index]
    if missing:
        report.finding("FAIL", "Requested GPU visibility", f"missing nvidia-smi indexes: {missing}")
    else:
        selected = [by_index[gpu_id] for gpu_id in gpu_ids]
        report.finding("PASS", "Requested GPU visibility", f"indexes {gpu_ids} are visible")
        low_memory = [
            row["index"]
            for row in selected
            if int(row.get("memory.free", "0")) < 70_000
        ]
        if low_memory:
            report.finding("WARN", "Requested GPU free memory", f"below 70000 MiB on indexes {low_memory}")
        else:
            report.finding("PASS", "Requested GPU free memory", "all requested GPUs have at least 70000 MiB free")

    report.command_result("GPU topology", run_command([nvidia_smi, "topo", "-m"], timeout=timeout))
    report.command_result(
        "NVIDIA device nodes",
        run_command(["bash", "-lc", "ls -l /dev/nvidia* 2>/dev/null || true"], timeout=timeout),
    )
    report.command_result(
        "CUDA and NCCL shared libraries",
        run_command(
            ["bash", "-lc", "command -v ldconfig >/dev/null && ldconfig -p | grep -E 'lib(cuda|cudart|nccl|nvrtc)\\.so' | head -n 100 || true"],
            timeout=timeout,
        ),
    )
    if shutil.which("nvcc"):
        report.command_result("CUDA compiler", run_command(["nvcc", "--version"], timeout=timeout))
    else:
        report.finding("INFO", "CUDA compiler", "nvcc is not installed; runtime-only images may omit it")

    if skip_cuda_smoke:
        report.finding("INFO", "PyTorch CUDA smoke", "skipped by --skip-cuda-smoke")
        return

    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    smoke = run_command([sys.executable, "-c", TORCH_CUDA_SMOKE], timeout=max(timeout, 120), env=child_env)
    report.command_result("PyTorch CUDA smoke", smoke)
    payload = None
    for line in reversed(smoke.stdout.splitlines()):
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if smoke.returncode == 0 and payload and payload.get("ok"):
        actual_count = payload.get("device_count")
        if actual_count == len(gpu_ids):
            report.finding("PASS", "PyTorch CUDA smoke", f"matmul passed on {actual_count} logical GPU(s)")
        else:
            report.finding("FAIL", "PyTorch CUDA smoke", f"expected {len(gpu_ids)} devices, saw {actual_count}")
    else:
        detail = payload.get("error") if isinstance(payload, dict) else f"exit code {smoke.returncode}"
        report.finding("FAIL", "PyTorch CUDA smoke", str(detail))


def collect_model(report: Report, model_path: Path | None, timeout: int) -> None:
    if model_path is None:
        report.finding("FAIL", "Model path", "MODEL_PATH and --model-path were both omitted")
        report.section("Model checkpoint", "No model path was provided.")
        return
    if not model_path.is_dir():
        report.finding("FAIL", "Model path", f"directory does not exist: {model_path}")
        report.section("Model checkpoint", f"Directory not found: `{sanitize(str(model_path))}`")
        return

    report.finding("PASS", "Model path", f"directory is readable: {model_path}")
    config_path = model_path / "config.json"
    config: dict[str, object] = {}
    config_error = None
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("config.json top level is not an object")
        config = loaded
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        config_error = f"{type(exc).__name__}: {exc}"

    if config_error:
        report.finding("FAIL", "Model config", config_error)
        config_body = config_error
    else:
        rows = [(key, json.dumps(config.get(key), ensure_ascii=False)) for key in MODEL_CONFIG_KEYS]
        config_body = markdown_table(("Key", "Value"), rows)
        report.finding("PASS", "Model config", "config.json parsed successfully")
        if config.get("expert_dtype") is None:
            report.finding("WARN", "Model expert_dtype", "config.json does not declare expert_dtype")
        else:
            report.finding("INFO", "Model expert_dtype", str(config.get("expert_dtype")))

    shard_files = list(model_path.glob("*.safetensors"))
    shard_bytes = 0
    stat_errors = 0
    for shard in shard_files:
        try:
            shard_bytes += shard.stat().st_size
        except OSError:
            stat_errors += 1
    index_path = model_path / "model.safetensors.index.json"
    index_summary = "not found"
    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index_data.get("weight_map", {}) if isinstance(index_data, dict) else {}
        metadata = index_data.get("metadata", {}) if isinstance(index_data, dict) else {}
        index_summary = f"weight_map entries={len(weight_map)}, metadata={metadata}"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        index_summary = f"unreadable: {type(exc).__name__}: {exc}"

    files_body = markdown_table(
        ("Item", "Value"),
        (
            ("Safetensors files", len(shard_files)),
            ("Safetensors total size", f"{shard_bytes / 1024**3:.2f} GiB"),
            ("Shard stat errors", stat_errors),
            ("Index summary", index_summary),
            ("tokenizer_config.json", (model_path / "tokenizer_config.json").is_file()),
        ),
    )
    report.section("Model checkpoint", "### Selected config fields\n\n" + config_body + "\n\n### Files\n\n" + files_body)
    report.command_result("Model filesystem capacity", run_command(["df", "-h", str(model_path)], timeout=timeout))


def collect_port(report: Report, port: int) -> None:
    with socket.socket() as sock:
        sock.settimeout(1)
        listening = sock.connect_ex(("127.0.0.1", port)) == 0
    if listening:
        report.finding("WARN", "Service port", f"127.0.0.1:{port} already accepts connections")
    else:
        report.finding("PASS", "Service port", f"127.0.0.1:{port} is available")


def parse_gpu_ids(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("--gpus must be a comma-separated list of non-negative indexes")
    values = [int(part) for part in parts]
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("--gpus contains duplicate indexes")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a redacted V4/SGLang environment report")
    default_model = Path(os.environ["MODEL_PATH"]) if os.environ.get("MODEL_PATH") else None
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--gpus", type=parse_gpu_ids, default=parse_gpu_ids("4,5,6,7"))
    parser.add_argument("--port", type=int, default=60000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=60, help="per-command timeout in seconds")
    parser.add_argument("--skip-cuda-smoke", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 2 when the report contains FAIL findings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or repo_root / "reports" / f"v4_environment_{timestamp}.md"
    report = Report(model_path=args.model_path, gpu_ids=args.gpus)

    collect_system(report, repo_root, args.timeout)
    collect_repository(report, repo_root, args.timeout)
    collect_python_environment(report, repo_root, args.gpus, args.timeout)
    collect_gpu_environment(report, args.gpus, args.timeout, args.skip_cuda_smoke)
    collect_model(report, args.model_path, args.timeout)
    collect_port(report, args.port)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.render(), encoding="utf-8")
    fail_count = sum(finding.status == "FAIL" for finding in report.findings)
    print(f"Report written to: {output}")
    print(f"FAIL findings: {fail_count}")
    return 2 if args.strict and fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
