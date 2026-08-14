#!/usr/bin/env python3
"""Fail-fast checks for serving DeepSeek-V4-Flash with SGLang.

This script intentionally has no third-party dependencies so it can run before
the CUDA/Python environment is fully installed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


EXPECTED_V4_FLASH = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_hash_layers": 3,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "expert_dtype": "fp4",
}


def _config_value(config: dict[str, Any], key: str) -> Any:
    aliases = {
        "hidden_size": ("hidden_size", "dim"),
        "num_hidden_layers": ("num_hidden_layers", "n_layers"),
        "num_hash_layers": ("num_hash_layers", "n_hash_layers"),
        "num_experts_per_tok": ("num_experts_per_tok", "n_activated_experts"),
        "scoring_func": ("scoring_func", "score_func"),
    }
    for candidate in aliases.get(key, (key,)):
        if candidate in config:
            return config[candidate]
    return None


def load_config(model_path: Path, config_path: Path | None) -> tuple[Path, dict[str, Any]]:
    path = config_path or model_path / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型配置: {path}")
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config.json 顶层必须是对象: {path}")
    return path, config


def query_gpus() -> tuple[list[dict[str, Any]], str | None]:
    queries = (
        "index,name,memory.total,memory.free,compute_cap",
        "index,name,memory.total,memory.free",
    )
    last_error: str | None = None
    for query in queries:
        command = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            last_error = str(exc)
            continue

        fields = query.split(",")
        gpus: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            values = [value.strip() for value in line.split(",")]
            if len(values) != len(fields):
                last_error = f"无法解析 nvidia-smi 输出: {line!r}"
                gpus = []
                break
            gpu = dict(zip(fields, values))
            gpu["index"] = int(gpu["index"])
            gpu["memory.total"] = int(gpu["memory.total"])
            gpu["memory.free"] = int(gpu["memory.free"])
            gpus.append(gpu)
        if gpus:
            return gpus, None
    return [], last_error


def visible_gpus(gpus: list[dict[str, Any]], cuda_visible_devices: str | None) -> list[dict[str, Any]]:
    if cuda_visible_devices is None or not cuda_visible_devices.strip():
        return gpus
    entries = [entry.strip() for entry in cuda_visible_devices.split(",") if entry.strip()]
    if entries == ["-1"]:
        return []
    if not all(entry.isdigit() for entry in entries):
        # UUID-based CUDA_VISIBLE_DEVICES cannot be joined reliably with the
        # index-only nvidia-smi query. The count is still useful for TP checks.
        return [{"index": entry, "name": "CUDA-visible GPU", "memory.total": 0, "memory.free": 0} for entry in entries]
    by_index = {gpu["index"]: gpu for gpu in gpus}
    return [by_index[int(entry)] for entry in entries if int(entry) in by_index]


def parse_version(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    facts: list[str] = []

    try:
        config_path, config = load_config(args.model_path, args.config)
        facts.append(f"配置文件: {config_path}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [], "facts": []}

    model_type = _config_value(config, "model_type")
    architectures = config.get("architectures", [])
    if model_type not in (None, "deepseek_v4") and not any("V4" in str(item) for item in architectures):
        errors.append(f"模型类型不是 DeepSeek-V4: model_type={model_type!r}, architectures={architectures!r}")

    for key, expected in EXPECTED_V4_FLASH.items():
        actual = _config_value(config, key)
        if actual is None:
            if key == "model_type":
                warnings.append("配置未声明 model_type；将按官方 inference 配置继续检查")
            else:
                warnings.append(f"配置缺少 {key}，无法验证官方 V4-Flash 结构")
        elif actual != expected:
            errors.append(f"{key}={actual!r}，预期官方 V4-Flash 为 {expected!r}")

    expert_dtype = _config_value(config, "expert_dtype")
    facts.append(
        "结构: "
        f"layers={_config_value(config, 'num_hidden_layers')}, "
        f"hash_layers={_config_value(config, 'num_hash_layers')}, "
        f"experts={_config_value(config, 'n_routed_experts')}, "
        f"topk={_config_value(config, 'num_experts_per_tok')}, "
        f"expert_dtype={expert_dtype}"
    )

    sglang_version = installed_version("sglang")
    if sglang_version is None:
        warnings.append("当前 Python 环境未安装 sglang；请在实际服务容器中再次运行 preflight")
    else:
        facts.append(f"SGLang: {sglang_version}")
        if parse_version(sglang_version) < (0, 5, 15):
            errors.append(
                f"SGLang {sglang_version} 过旧；仓库原 requirements 的 0.4.3 不支持 DeepSeek-V4，"
                "请使用当前 lmsysorg/sglang:latest 或更新版本"
            )

    gpus, gpu_error = query_gpus()
    selected = visible_gpus(gpus, os.getenv("CUDA_VISIBLE_DEVICES"))
    if not gpus:
        message = f"无法查询 NVIDIA GPU ({gpu_error or 'unknown error'})"
        if args.allow_no_gpu:
            warnings.append(message)
        else:
            errors.append(message)
    else:
        visible_setting = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible_setting and visible_setting.strip() and visible_setting.strip() != "-1" and not selected:
            errors.append(
                f"CUDA_VISIBLE_DEVICES={visible_setting!r} 与 nvidia-smi GPU index 不匹配"
            )
        facts.append(
            "可见 GPU: "
            + "; ".join(
                f"{gpu['index']}={gpu['name']} free={gpu['memory.free']}MiB/{gpu['memory.total']}MiB"
                for gpu in selected
            )
        )
        if len(selected) != args.tp:
            errors.append(
                f"CUDA_VISIBLE_DEVICES 实际可见 {len(selected)} 张卡，但 --tp={args.tp}；"
                "两者必须一致"
            )

        names = " ".join(str(gpu["name"]).lower() for gpu in selected)
        if "h100" in names and expert_dtype == "fp4":
            if args.tp != 8:
                errors.append("官方 H100 + V4-Flash FP4 单机路径要求 TP=8；不要使用原报告中的 TP=4")
            if args.backend != "marlin":
                errors.append("H100 上原始 FP4 专家应使用 --moe-runner-backend marlin")
        if any(gpu["memory.free"] and gpu["memory.free"] < args.min_free_mib for gpu in selected):
            errors.append(
                f"至少一张可见 GPU 的空闲显存低于 {args.min_free_mib} MiB；"
                "请先清理残留进程再启动"
            )

    if args.backend == "triton" and expert_dtype == "fp4":
        errors.append("V4-Flash 原始 FP4 专家不能沿用旧 Triton fused_moe 路径；这会复现 hidden-size mismatch")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "facts": facts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash + SGLang 启动前检查")
    parser.add_argument("--model-path", type=Path, required=True, help="原始 Hugging Face 模型目录")
    parser.add_argument("--config", type=Path, help="可选的 config.json；默认读取 MODEL_PATH/config.json")
    parser.add_argument("--tp", type=int, default=8, help="SGLang tensor parallel size")
    parser.add_argument("--backend", default="marlin", help="SGLang MoE runner backend")
    parser.add_argument("--min-free-mib", type=int, default=70_000, help="每卡最低空闲显存，仅作启动保护")
    parser.add_argument("--allow-no-gpu", action="store_true", help="本地/CI 静态检查时允许没有 NVIDIA GPU")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_checks(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for fact in result["facts"]:
            print(f"[INFO] {fact}")
        for warning in result["warnings"]:
            print(f"[WARN] {warning}")
        for error in result["errors"]:
            print(f"[ERROR] {error}", file=sys.stderr)
        print("[PASS] preflight 通过" if result["ok"] else "[FAIL] preflight 未通过")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
