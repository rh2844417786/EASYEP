#!/usr/bin/env python3
"""Generate any number of V4 EASY-EP masks from one calibration probe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from expert_selection import aggregate_scores, build_mask, load_samples
from v4_pruning_targets import (
    actual_prune_percent,
    decimal_text,
    mask_filename,
    model_directory_name,
    resolve_targets,
    validate_safe_prefix,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_mask(path: Path, value: list[list[int]]) -> None:
    """Preserve the historical mask byte format and therefore its SHA-256."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_save_scores(path: Path, scores: torch.Tensor) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(scores, temporary)
    os.replace(temporary, path)


def v4_mask(scores: torch.Tensor, target_experts: int, hash_layers: int) -> list[list[int]]:
    mask = build_mask(scores, target_experts).to(dtype=torch.int8).tolist()
    for layer in range(hash_layers):
        mask[layer] = [1] * scores.shape[1]
    return mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate one EASY-EP V4 calibration probe once and generate one or "
            "more per-layer Top-K masks"
        )
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scores-file", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mask-prefix", default="aime_v4")
    parser.add_argument(
        "--target-experts",
        type=int,
        action="append",
        default=[],
        help="retained dynamic-layer expert count; repeat for multiple masks",
    )
    parser.add_argument(
        "--prune-percent",
        action="append",
        default=[],
        help=(
            "requested dynamic-layer prune percentage in [0,100); repeat as needed. "
            "Retained counts round upward so actual pruning never exceeds the request"
        ),
    )
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=43)
    parser.add_argument("--hash-layers", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument(
        "--sample-strategy", choices=("longest", "random", "first"), default="longest"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input_file.is_file():
        raise FileNotFoundError(f"calibration statistics do not exist: {args.input_file}")
    if args.num_experts <= 0:
        raise ValueError("num-experts must be positive")
    if not 0 <= args.hash_layers <= args.num_layers:
        raise ValueError("hash-layers must be in [0, num-layers]")
    prefix = validate_safe_prefix(args.mask_prefix)
    targets = resolve_targets(
        args.target_experts, args.prune_percent, args.num_experts
    )

    samples = load_samples(
        args.input_file, args.num_samples, args.sample_strategy, args.seed
    )
    scores = aggregate_scores(samples, args.num_experts)
    if scores.shape != (args.num_layers, args.num_experts):
        raise ValueError(
            f"score shape must be ({args.num_layers}, {args.num_experts}), "
            f"found {tuple(scores.shape)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_file = args.scores_file or args.output_dir / f"{prefix}_scores.pt"
    manifest_path = args.manifest or args.output_dir / f"{prefix}_mask_manifest.json"
    scores_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_scores(scores_file, scores)

    records: list[dict[str, Any]] = []
    for target in targets:
        mask = v4_mask(scores, target, args.hash_layers)
        counts = [sum(row) for row in mask]
        expected = [args.num_experts] * args.hash_layers + [target] * (
            args.num_layers - args.hash_layers
        )
        if counts != expected:
            raise RuntimeError(
                f"generated mask counts disagree for target {target}: {counts}"
            )
        mask_path = args.output_dir / mask_filename(
            prefix, target, args.num_experts
        )
        atomic_write_mask(mask_path, mask)
        dynamic_percent = actual_prune_percent(target, args.num_experts)
        main_slot_percent = (
            (args.num_layers - args.hash_layers)
            * (args.num_experts - target)
            * 100
            / (args.num_layers * args.num_experts)
        )
        records.append(
            {
                "path": str(mask_path.resolve()),
                "sha256": sha256_file(mask_path),
                "dynamic_layer_experts": target,
                "dynamic_layer_prune_percent": decimal_text(dynamic_percent),
                "main_layer_slot_prune_percent": main_slot_percent,
                "model_directory_name": model_directory_name(
                    target, args.num_experts
                ),
                "hash_layers_preserved": list(range(args.hash_layers)),
                "dynamic_layers_pruned": list(
                    range(args.hash_layers, args.num_layers)
                ),
            }
        )

    manifest = {
        "type": "easyep_mask_manifest",
        "format_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "token_statistics": str(args.input_file.resolve()),
        "token_statistics_sha256": sha256_file(args.input_file),
        "scores": str(scores_file.resolve()),
        "scores_sha256": sha256_file(scores_file),
        "score_shape": list(scores.shape),
        "score_calibration": {
            "num_samples": args.num_samples,
            "sample_strategy": args.sample_strategy,
            "seed": args.seed,
        },
        "num_layers": args.num_layers,
        "hash_layers": args.hash_layers,
        "num_experts": args.num_experts,
        "masks": records,
        "physical_pruned_checkpoints_created": False,
        "warning": (
            "Masks are not checkpoints. Hash-routed layers remain complete; "
            "physical checkpoints require the matching EASY-EP runtime patch."
        ),
    }
    atomic_write_json(manifest_path, manifest)
    print(f"saved scores={tuple(scores.shape)} to {scores_file}")
    for record in records:
        print(
            "generated "
            f"keep={record['dynamic_layer_experts']} "
            f"dynamic_prune={record['dynamic_layer_prune_percent']}% "
            f"mask={record['path']}"
        )
    print(f"mask manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
