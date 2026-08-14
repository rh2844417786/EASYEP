#!/usr/bin/env python3
"""Collect EASY-EP statistics from DeepSeek-V4's official inference model.

The original ``inf_new.py`` embeds a DeepSeek-V3/R1 implementation and cannot
load V4. This collector imports the official V4 ``inference/model.py`` at
runtime and instruments its MoE/Hyper-Connection blocks without modifying the
upstream checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from safetensors.torch import load_model
from tqdm import tqdm


class ProbeRecorder:
    def __init__(self, rank: int, expected_layers: int, sync_debug: bool = False):
        self.rank = rank
        self.expected_layers = expected_layers
        self.sync_debug = sync_debug
        self.trace = False
        self.experts: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.similarities: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def enabled(self) -> bool:
        return self.rank == 0

    def begin_sample(self, trace: bool = False) -> None:
        self.experts.clear()
        self.similarities.clear()
        self.trace = trace

    def stage(self, layer: int, name: str) -> None:
        if self.sync_debug:
            torch.cuda.synchronize()
        if self.enabled and self.trace:
            print(f"[v4-probe] layer={layer:02d} stage={name}", flush=True)

    def record_experts(
        self,
        layer: int,
        weights: torch.Tensor,
        indices: torch.Tensor,
        norms: torch.Tensor,
    ) -> None:
        if self.enabled:
            self.experts[layer] = (
                weights.detach().float().cpu(),
                indices.detach().cpu(),
                norms.detach().float().cpu(),
            )

    def record_similarities(
        self,
        layer: int,
        before_to_routed: torch.Tensor,
        shared_to_full: torch.Tensor,
        routed_to_full: torch.Tensor,
    ) -> None:
        if self.enabled:
            self.similarities[layer] = (
                before_to_routed.detach().float().cpu(),
                shared_to_full.detach().float().cpu(),
                routed_to_full.detach().float().cpu(),
            )

    def finish_sample(self) -> dict[str, Any]:
        expected = list(range(self.expected_layers))
        if sorted(self.experts) != expected or sorted(self.similarities) != expected:
            raise RuntimeError(
                "probe did not observe every V4 layer: "
                f"expert_layers={sorted(self.experts)}, similarity_layers={sorted(self.similarities)}, "
                f"expected={expected}"
            )
        weights, indices, norms = [], [], []
        simibr, simisf, simirf = [], [], []
        for layer in expected:
            layer_weights, layer_indices, layer_norms = self.experts[layer]
            layer_simibr, layer_simisf, layer_simirf = self.similarities[layer]
            weights.append(layer_weights.tolist())
            indices.append(layer_indices.tolist())
            norms.append(layer_norms.tolist())
            simibr.append(layer_simibr.tolist())
            simisf.append(layer_simisf.tolist())
            simirf.append(layer_simirf.tolist())
        return {
            "format_version": 2,
            "architecture": "deepseek_v4",
            "layer_ids": expected,
            "idxs": indices,
            "weights": weights,
            "norms": norms,
            "simibr": simibr,
            "simisf": simisf,
            "simirf": simirf,
        }


def load_official_model_module(inference_dir: Path) -> ModuleType:
    model_file = inference_dir / "model.py"
    kernel_file = inference_dir / "kernel.py"
    if not model_file.is_file() or not kernel_file.is_file():
        raise FileNotFoundError(
            f"{inference_dir} must contain the official DeepSeek-V4 model.py and kernel.py"
        )
    sys.path.insert(0, str(inference_dir))
    spec = importlib.util.spec_from_file_location("easyep_deepseek_v4_official_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official model from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("ModelArgs", "Transformer", "MoE", "Block"):
        if not hasattr(module, name):
            raise RuntimeError(f"official model.py is missing expected symbol {name}")
    return module


def patch_official_model(module: ModuleType, recorder: ProbeRecorder) -> None:
    """Replace V4 MoE/Block forwards with numerically equivalent probe versions."""

    def moe_forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids is None:
            raise RuntimeError("DeepSeek-V4 MoE probing requires input_ids for hash routing")
        shape = x.size()
        flat_x = x.view(-1, self.dim)
        weights, indices = self.gate(flat_x, input_ids.flatten())
        routed = torch.zeros_like(flat_x, dtype=torch.float32)
        num_tokens = flat_x.size(0)
        local_norms = torch.zeros(
            (self.n_local_experts, num_tokens), device=flat_x.device, dtype=torch.float32
        )
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for expert_id in range(self.experts_start_idx, self.experts_end_idx):
            if counts[expert_id] == 0:
                continue
            token_indices, topk_slots = torch.where(indices == expert_id)
            route_weights = weights[token_indices, topk_slots, None]
            # Preserve the official V4 forward exactly (weights are applied
            # before the quantized w2 GEMM), then run the expert once more
            # without weights to obtain EASY-EP's ||E_i(x)|| output norm.
            weighted_output = self.experts[expert_id](flat_x[token_indices], route_weights)
            routed[token_indices] += weighted_output.float()
            expert_output = self.experts[expert_id](flat_x[token_indices])
            local_norms[expert_id - self.experts_start_idx, token_indices] = torch.linalg.vector_norm(
                expert_output.float(), ord=2, dim=-1
            )

        if module.world_size > 1:
            gathered_norms = [torch.zeros_like(local_norms) for _ in range(module.world_size)]
            dist.all_gather(gathered_norms, local_norms)
            all_norms = torch.cat(gathered_norms, dim=0).transpose(0, 1)
            dist.all_reduce(routed)
        else:
            all_norms = local_norms.transpose(0, 1)
        selected_norms = all_norms.gather(1, indices)
        recorder.record_experts(self.layer_id, weights, indices, selected_norms)

        shared = self.shared_experts(flat_x).float()
        if recorder.enabled:
            self._easyep_routed = routed.type_as(flat_x).view(shape)
            self._easyep_shared = shared.type_as(flat_x).view(shape)
        return (routed + shared).type_as(flat_x).view(shape)

    def block_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        recorder.stage(self.layer_id, "attention:start")
        residual = x
        attention_input, post, combination = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        attention_output = self.attn(self.attn_norm(attention_input), start_pos)
        x = self.hc_post(attention_output, residual, post, combination)
        recorder.stage(self.layer_id, "attention:end")

        residual = x
        ffn_input, post, combination = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        recorder.stage(self.layer_id, "moe:start")
        ffn_output = self.ffn(self.ffn_norm(ffn_input), input_ids)
        recorder.stage(self.layer_id, "moe:end")
        x = self.hc_post(ffn_output, residual, post, combination)

        if recorder.enabled:
            routed_component = self.ffn._easyep_routed
            shared_component = self.ffn._easyep_shared
            del self.ffn._easyep_routed
            del self.ffn._easyep_shared
            zero = torch.zeros_like(routed_component)
            baseline = self.hc_post(zero, residual, post, combination)
            routed_only = self.hc_post(routed_component, residual, post, combination)
            shared_only = self.hc_post(shared_component, residual, post, combination)
            # Flatten the HC copies so each token has one contribution score.
            baseline_flat = baseline.flatten(-2).float()
            routed_flat = routed_only.flatten(-2).float()
            shared_flat = shared_only.flatten(-2).float()
            full_flat = x.flatten(-2).float()
            recorder.record_similarities(
                self.layer_id,
                F.cosine_similarity(baseline_flat, routed_flat, dim=-1),
                F.cosine_similarity(shared_flat, full_flat, dim=-1),
                F.cosine_similarity(routed_flat, full_flat, dim=-1),
            )
        recorder.stage(self.layer_id, "block:end")
        return x

    module.MoE.forward = moe_forward
    module.Block.forward = block_forward


def count_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_probe_dataset(path: Path, split: str | None):
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split is None:
            if len(dataset) != 1:
                raise ValueError(f"dataset has splits {list(dataset)}; pass --split")
            split = next(iter(dataset))
        dataset = dataset[split]
    if "input_ids" not in dataset.column_names:
        raise ValueError(f"dataset {path} has no input_ids column")
    return dataset


@torch.inference_mode()
def collect(args: argparse.Namespace) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)

    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("score_func") != "sqrtsoftplus":
        raise ValueError("V4 probe config must use score_func=sqrtsoftplus")
    if config.get("n_hash_layers") != 3:
        raise ValueError("V4-Flash probe config must declare n_hash_layers=3")
    config["max_batch_size"] = 1
    config["max_seq_len"] = args.max_input_tokens

    module = load_official_model_module(args.inference_dir)
    model_args = module.ModelArgs(**config)
    recorder = ProbeRecorder(rank, model_args.n_layers, args.sync_debug)
    patch_official_model(module, recorder)

    with torch.device("cuda"):
        model = module.Transformer(model_args)
    checkpoint = args.ckpt_path / f"model{rank}-mp{world_size}.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"missing {checkpoint}; converted checkpoint MP={world_size} must match torchrun world size"
        )
    missing, unexpected = load_model(model, str(checkpoint), strict=False)
    allowed_missing_prefixes = ("mtp.",)
    disallowed_missing = [
        key for key in missing if not key.startswith(allowed_missing_prefixes)
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "converted checkpoint/model mismatch: "
            f"missing={disallowed_missing[:10]}, unexpected={unexpected[:10]}"
        )
    if rank == 0 and missing:
        print(f"[WARN] allowed missing MTP keys: {missing[:10]}", flush=True)
    torch.set_default_device("cuda")

    dataset = load_probe_dataset(args.input_file, args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    resume_count = count_records(args.output) if rank == 0 and args.resume else 0
    if world_size > 1:
        value = [resume_count]
        dist.broadcast_object_list(value, src=0)
        resume_count = int(value[0])
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            args.output.write_text("", encoding="utf-8")
        print(
            f"V4 probe: samples={len(dataset)}, resume={resume_count}, mp={world_size}, "
            f"expert_dtype={model_args.expert_dtype}, max_input_tokens={args.max_input_tokens}",
            flush=True,
        )
    if resume_count > len(dataset):
        raise ValueError(
            f"resume file has {resume_count} records but selected dataset has only {len(dataset)} samples"
        )

    iterator = tqdm(range(resume_count, len(dataset)), disable=rank != 0, desc="V4 EASY-EP probe")
    processed_count = resume_count
    for sample_index in iterator:
        token_ids = dataset[sample_index]["input_ids"]
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"sample {sample_index} has invalid input_ids")
        if len(token_ids) > args.max_input_tokens:
            raise ValueError(
                "V4 probe does not silently skip overlong samples because JSONL resume is "
                f"record-count based. sample={sample_index}, tokens={len(token_ids)}, "
                f"limit={args.max_input_tokens}; rebuild/filter the dataset or raise the limit."
            )
        if min(token_ids) < 0 or max(token_ids) >= model_args.vocab_size:
            raise ValueError(
                f"sample {sample_index} token id outside V4 vocabulary [0, {model_args.vocab_size})"
            )
        recorder.begin_sample(trace=args.trace_first_sample and sample_index == resume_count)
        tokens = torch.tensor([token_ids], device="cuda", dtype=torch.long)
        model.forward(tokens, 0)
        if rank == 0:
            record = recorder.finish_sample()
            record["sample_index"] = sample_index
            record["num_tokens"] = len(token_ids)
            record["num_routed_experts"] = model_args.n_routed_experts
            record["num_activated_experts"] = model_args.n_activated_experts
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            processed_count += 1

    if rank == 0 and processed_count != len(dataset):
        raise RuntimeError(
            f"probe wrote {processed_count} records for {len(dataset)} selected samples"
        )

    if world_size > 1:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect EASY-EP statistics from official DeepSeek-V4 inference")
    parser.add_argument("--inference-dir", type=Path, required=True, help="official V4 inference/ directory")
    parser.add_argument("--ckpt-path", type=Path, required=True, help="official convert.py output directory")
    parser.add_argument("--config", type=Path, required=True, help="official inference-format V4 config")
    parser.add_argument("--input-file", type=Path, required=True, help="datasets.save_to_disk calibration data")
    parser.add_argument("--split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=13000)
    parser.add_argument("--limit", type=int, help="only probe the first N records")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-first-sample", action="store_true")
    parser.add_argument("--sync-debug", action="store_true", help="CUDA synchronize at every layer stage")
    parser.add_argument("--seed", type=int, default=965)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        collect(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
