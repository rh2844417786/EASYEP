#!/usr/bin/env python3
"""Smoke-test full/prune25/prune50 V4 checkpoints with the same fast server profile.

Each model is started alone on physical GPUs 4..7, captures the TP=4 decode
CUDA graph (full, batch 32), then receives one OpenAI-compatible request.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "scripts" / "smoke_v4_server.py"
BENCHMARKS = REPO_ROOT / "evaluation" / "run_v4_benchmarks.py"


def server_command(args: argparse.Namespace, model: Path) -> list[str]:
    return [
        str(args.server_python), "-m", "sglang.launch_server", "--trust-remote-code",
        "--model-path", str(model), "--tp", "4", "--moe-runner-backend", "marlin",
        "--reasoning-parser", "deepseek-v4", "--tool-call-parser", "deepseekv4",
        "--host", "127.0.0.1", "--port", str(args.port), "--context-length", "65536",
        "--mem-fraction-static", "0.80", "--chunked-prefill-size", "8192",
        "--max-running-requests", "32", "--cuda-graph-backend-decode", "full",
        "--cuda-graph-max-bs-decode", "32", "--watchdog-timeout", "1800",
        "--disable-custom-all-reduce", "--disable-shared-experts-fusion",
        "--decode-log-interval", "10", "--enable-metrics",
    ]


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        "PYTHONUNBUFFERED": "1", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo", "NCCL_CUMEM_HOST_ENABLE": "0",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "TORCH_NCCL_BLOCKING_WAIT": "1",
    })
    return env


def ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-model", type=Path, default=Path("/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash"))
    parser.add_argument("--prune25-model", type=Path, default=Path("/mnt/docker_data/v4-converted/v4-prune25-keep192"))
    parser.add_argument("--prune50-model", type=Path, default=Path("/mnt/docker_data/v4-converted/v4-prune50-keep128"))
    parser.add_argument("--server-python", type=Path, default=Path("/opt/sglang-v4/bin/python"))
    parser.add_argument("--port", type=int, default=60000)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--smoke-timeout", type=float, default=600)
    parser.add_argument("--eval-max-tokens", type=int, default=128,
                        help="Token cap for the one-question benchmark adapter checks")
    parser.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    args = parser.parse_args()
    if not args.server_python.is_file():
        raise FileNotFoundError(args.server_python)
    if ready(args.port):
        raise RuntimeError(
            f"port {args.port} already serves SGLang; stop that service before running the model matrix"
        )
    variants = {"full": args.full_model, "prune25": args.prune25_model, "prune50": args.prune50_model}
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, model in variants.items():
        if not model.is_dir():
            raise FileNotFoundError(model)
        server_log = args.logs_dir / f"v4_{name}_fast_smoke_{stamp}.server.log"
        smoke_dir = args.logs_dir / f"v4_{name}_fast_smoke_{stamp}"
        smoke_log = smoke_dir / "smoke.log"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] starting fast CUDA-graph server: {model}", flush=True)
        with server_log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(server_command(args, model), cwd=REPO_ROOT, env=runtime_env(),
                text=True, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"[{name}] server exited {process.returncode}; log={server_log}")
                if ready(args.port):
                    break
                time.sleep(2)
            else:
                raise TimeoutError(f"[{name}] startup timeout; log={server_log}")
            command = [str(args.server_python), str(SMOKE), "--base-url", f"http://127.0.0.1:{args.port}/v1", "--timeout", str(args.smoke_timeout)]
            with smoke_log.open("w", encoding="utf-8") as handle:
                subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=handle, stderr=subprocess.STDOUT,
                    timeout=args.smoke_timeout + 30, check=True)
                for data_name in ("agent_os", "gpqa", "kuvecodebench"):
                    print(f"[{name}] adapter smoke: {data_name}", file=handle, flush=True)
                    evaluation = [
                        str(args.server_python), "-u", str(BENCHMARKS), "--data-name", data_name,
                        "--target-path", str(smoke_dir / "evaluation"),
                        "--output-file", str(smoke_dir / "evaluation" / f"{data_name}.jsonl"),
                        "--base-url", f"http://127.0.0.1:{args.port}/v1", "--model", str(model),
                        "--max-tokens", str(args.eval_max_tokens), "--workers", "1", "--repeats", "1",
                        "--temperature", "1.0", "--top-p", "1.0", "--timeout", str(args.smoke_timeout),
                        "--retries", "0", "--limit", "1", "--no-resume",
                    ]
                    subprocess.run(evaluation, cwd=REPO_ROOT, text=True, stdout=handle, stderr=subprocess.STDOUT,
                        timeout=args.smoke_timeout + 30, check=True)
            captured = "Capture target decode CUDA graph end" in server_log.read_text(encoding="utf-8", errors="replace")
            if not captured:
                raise RuntimeError(f"[{name}] smoke passed but decode CUDA graph was not captured; log={server_log}")
            print(f"[{name}] PASS fast CUDA-graph smoke; log={smoke_log}", flush=True)
        finally:
            stop(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
