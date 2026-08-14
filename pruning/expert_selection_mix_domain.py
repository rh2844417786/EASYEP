"""Combine normalized per-domain EASY-EP scores into one expert mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def safe_load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{path} 必须包含二维 score tensor")
    value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{path} 包含 NaN/Inf")
    if (value < 0).any():
        raise ValueError(f"{path} 包含负分数")
    row_sums = value.sum(dim=-1, keepdim=True)
    if (row_sums <= 0).any():
        bad_layers = torch.where(row_sums.squeeze(-1) <= 0)[0].tolist()
        raise ValueError(f"{path} 的这些层总分为 0: {bad_layers}")
    return value / row_sums


def combine_scores(paths: list[Path]) -> torch.Tensor:
    if not paths:
        raise ValueError("目录中没有 .pt 专家分数文件")
    tensors = [safe_load_tensor(path) for path in paths]
    expected_shape = tensors[0].shape
    for path, tensor in zip(paths[1:], tensors[1:]):
        if tensor.shape != expected_shape:
            raise ValueError(f"{path} shape={tuple(tensor.shape)}，预期 {tuple(expected_shape)}")
    return torch.stack(tensors).mean(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge normalized expert scores across domains")
    parser.add_argument(
        "--expert-info-dir", "--expert_info_dir", dest="expert_info_dir", type=Path,
        default=Path("expert_statistics/expert_information"),
    )
    parser.add_argument("--target-number", "--target_number", dest="target_number", type=int, default=128)
    parser.add_argument("--expert-mask", "--expert_mask", dest="expert_mask", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(path for path in args.expert_info_dir.iterdir() if path.is_file() and path.suffix == ".pt")
    combined = combine_scores(paths)
    if not 1 <= args.target_number <= combined.shape[1]:
        raise ValueError(f"target_number 必须在 [1, {combined.shape[1]}]，实际为 {args.target_number}")

    topk_experts = torch.topk(combined, k=args.target_number, dim=-1).indices
    mask = torch.zeros_like(combined)
    mask.scatter_(1, topk_experts, 1)
    args.expert_mask.parent.mkdir(parents=True, exist_ok=True)
    with args.expert_mask.open("w", encoding="utf-8") as handle:
        json.dump(mask.tolist(), handle)
    print(
        f"merged {len(paths)} domains with score shape={tuple(combined.shape)}; "
        f"kept {args.target_number} experts/layer in {args.expert_mask}"
    )


if __name__ == "__main__":
    main()
