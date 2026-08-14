"""Aggregate EASY-EP token statistics into per-layer expert masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import torch
from tqdm import tqdm


REQUIRED_FIELDS = ("idxs", "weights", "norms", "simibr")


def load_samples(path: Path, num_samples: int, strategy: str, seed: int) -> list[dict[str, Any]]:
    records: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                records.append((line_number, line))
    if not records:
        raise ValueError(f"统计文件为空: {path}")

    if strategy == "longest":
        # Randomize ties while retaining the original preference for long probes.
        random.Random(seed).shuffle(records)
        records.sort(key=lambda item: len(item[1]), reverse=True)
    elif strategy == "random":
        random.Random(seed).shuffle(records)
    elif strategy != "first":
        raise ValueError(f"未知样本选择策略: {strategy}")

    if num_samples > 0:
        records = records[:num_samples]

    samples: list[dict[str, Any]] = []
    for line_number, line in records:
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON: {exc}") from exc
        missing = [field for field in REQUIRED_FIELDS if field not in sample]
        if missing:
            raise ValueError(f"{path}:{line_number} 缺少字段: {', '.join(missing)}")
        samples.append(sample)
    return samples


def _token_similarities(value: Any, layer: int) -> list[float]:
    similarities = value[layer]
    # Collector output is normally [batch=1, tokens]. Accept a flat token list
    # as well, but reject batches > 1 because the other arrays are flattened.
    if similarities and isinstance(similarities[0], list):
        if len(similarities) != 1:
            raise ValueError(
                f"layer {layer}: simibr has batch size {len(similarities)}; "
                "the selection format requires batch size 1"
            )
        similarities = similarities[0]
    return similarities


def infer_shape(samples: list[dict[str, Any]], num_experts: int | None) -> tuple[int, int]:
    num_layers = len(samples[0]["idxs"])
    if num_layers == 0:
        raise ValueError("idxs 没有任何 MoE 层")
    max_expert = -1
    for sample_index, sample in enumerate(samples):
        for field in REQUIRED_FIELDS:
            if len(sample[field]) != num_layers:
                raise ValueError(
                    f"sample {sample_index}: {field} 有 {len(sample[field])} 层，预期 {num_layers}"
                )
        for layer_routes in sample["idxs"]:
            for token_routes in layer_routes:
                if token_routes:
                    max_expert = max(max_expert, max(int(index) for index in token_routes))
    inferred = max_expert + 1
    if num_experts is None:
        if inferred <= 0:
            raise ValueError("无法从空路由中推断专家数量，请传 --num-experts")
        num_experts = inferred
    elif inferred > num_experts:
        raise ValueError(f"观测到 expert id {max_expert}，超过 --num-experts={num_experts}")
    return num_layers, num_experts


def aggregate_scores(samples: list[dict[str, Any]], num_experts: int | None = None) -> torch.Tensor:
    num_layers, num_experts = infer_shape(samples, num_experts)
    scores = torch.zeros((num_layers, num_experts), dtype=torch.float32)

    for sample_index, sample in enumerate(tqdm(samples, desc="aggregate expert scores")):
        for layer in range(num_layers):
            routes = sample["idxs"][layer]
            weights = sample["weights"][layer]
            norms = sample["norms"][layer]
            similarities = _token_similarities(sample["simibr"], layer)
            token_count = len(routes)
            if not (len(weights) == len(norms) == len(similarities) == token_count):
                raise ValueError(
                    f"sample {sample_index}, layer {layer}: token dimensions disagree: "
                    f"idxs={token_count}, weights={len(weights)}, norms={len(norms)}, "
                    f"simibr={len(similarities)}"
                )

            try:
                route_tensor = torch.tensor(routes, dtype=torch.long)
                weight_tensor = torch.tensor(weights, dtype=torch.float32)
                norm_tensor = torch.tensor(norms, dtype=torch.float32)
                similarity_tensor = torch.tensor(similarities, dtype=torch.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sample {sample_index}, layer {layer}: ragged or non-numeric top-k data"
                ) from exc
            if route_tensor.ndim != 2 or route_tensor.shape != weight_tensor.shape or route_tensor.shape != norm_tensor.shape:
                raise ValueError(
                    f"sample {sample_index}, layer {layer}: expected matching [tokens, topk] arrays; "
                    f"idxs={tuple(route_tensor.shape)}, weights={tuple(weight_tensor.shape)}, "
                    f"norms={tuple(norm_tensor.shape)}"
                )
            if similarity_tensor.shape != (token_count,):
                raise ValueError(
                    f"sample {sample_index}, layer {layer}: simibr shape={tuple(similarity_tensor.shape)}, "
                    f"expected ({token_count},)"
                )
            if route_tensor.numel() and (
                route_tensor.min().item() < 0 or route_tensor.max().item() >= num_experts
            ):
                raise ValueError(
                    f"sample {sample_index}, layer {layer}: expert ids outside [0, {num_experts})"
                )
            contribution = (1.0 - similarity_tensor).clamp_min_(0.0).unsqueeze(1)
            values = weight_tensor * norm_tensor * contribution
            scores[layer].scatter_add_(0, route_tensor.flatten(), values.flatten())

    if not torch.isfinite(scores).all():
        raise ValueError("聚合分数包含 NaN/Inf，请检查采集输出")
    return scores


def build_mask(scores: torch.Tensor, target_number: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"scores 必须是二维张量，实际 shape={tuple(scores.shape)}")
    if not 1 <= target_number <= scores.shape[1]:
        raise ValueError(f"target_number 必须在 [1, {scores.shape[1]}]，实际为 {target_number}")
    topk_experts = torch.topk(scores, target_number, dim=-1).indices
    mask = torch.zeros_like(scores)
    mask.scatter_(1, topk_experts, 1)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate EASY-EP probe JSONL and write an expert mask")
    parser.add_argument("--input-file", "--input_file", dest="input_file", type=Path, required=True)
    parser.add_argument("--output-file", "--output_file", dest="output_file", type=Path, required=True)
    parser.add_argument("--target-number", "--target_number", dest="target_number", type=int, default=128)
    parser.add_argument("--expert-mask", "--expert_mask", dest="expert_mask", type=Path, required=True)
    parser.add_argument(
        "--num-experts", type=int, default=256,
        help="模型的 routed expert 数；默认 256，避免短 calibration 未覆盖高编号专家时低估维度",
    )
    parser.add_argument("--num-samples", type=int, default=25, help="0 表示使用全部样本")
    parser.add_argument("--sample-strategy", choices=("longest", "random", "first"), default="longest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_samples(args.input_file, args.num_samples, args.sample_strategy, args.seed)
    scores = aggregate_scores(samples, args.num_experts)
    mask = build_mask(scores, args.target_number)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.expert_mask.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scores, args.output_file)
    with args.expert_mask.open("w", encoding="utf-8") as handle:
        json.dump(mask.tolist(), handle)
    print(
        f"saved scores={tuple(scores.shape)} to {args.output_file}; "
        f"kept {args.target_number} experts/layer in {args.expert_mask}"
    )


if __name__ == "__main__":
    main()
