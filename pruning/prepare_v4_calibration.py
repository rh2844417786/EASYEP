#!/usr/bin/env python3
"""Re-tokenize an existing EASY-EP calibration dataset with DeepSeek-V4."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer


def load_encoder(inference_dir: Path):
    encoding_dir = inference_dir.parent / "encoding"
    encoder_file = encoding_dir / "encoding_dsv4.py"
    if not encoder_file.is_file():
        raise FileNotFoundError(
            f"cannot find {encoder_file}; use the official V4 repository containing inference/ and encoding/"
        )
    sys.path.insert(0, str(encoding_dir))
    spec = importlib.util.spec_from_file_location("easyep_encoding_dsv4", encoder_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {encoder_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare tokenizer-correct DeepSeek-V4 calibration data")
    parser.add_argument("--source", type=Path, required=True, help="existing datasets.save_to_disk directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="original Hugging Face V4 model directory")
    parser.add_argument("--inference-dir", type=Path, required=True, help="official V4 inference/ directory")
    parser.add_argument("--text-field", default="input")
    parser.add_argument("--thinking-mode", default="chat")
    parser.add_argument("--num-proc", type=int, default=1)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {args.output}")
    dataset = load_from_disk(str(args.source))
    sample_dataset = next(iter(dataset.values())) if isinstance(dataset, DatasetDict) else dataset
    if args.text_field not in sample_dataset.column_names:
        raise ValueError(f"{args.source} has no text field {args.text_field!r}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    encode_messages = load_encoder(args.inference_dir)

    def tokenize(batch):
        input_ids = []
        for text in batch[args.text_field]:
            encoded_prompt = encode_messages(
                [{"role": "user", "content": text}], thinking_mode=args.thinking_mode
            )
            input_ids.append(tokenizer.encode(encoded_prompt))
        return {
            "input_ids": input_ids,
            "easyep_tokenizer_source": [str(args.model_path)] * len(input_ids),
        }

    dataset = dataset.map(tokenize, batched=True, num_proc=args.num_proc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output))
    print(f"saved V4-tokenized calibration data to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
