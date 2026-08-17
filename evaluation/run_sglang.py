#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible SGLang server with resumable requests.

This client deliberately does not import SGLang. The server and evaluation
environment can therefore use different SGLang versions, which is required for
DeepSeek-V4 while the original EASY-EP quick-mask code remains on SGLang 0.4.3.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

from evaluator.MATH_evaluator_list import MATHEvaluator


DATASETS = {
    "AIME24": SCRIPT_DIR / "dataset" / "AIME24.jsonl",
    "AIME25": SCRIPT_DIR / "dataset" / "AIME25.jsonl",
    "hmmt_feb_2025": SCRIPT_DIR / "dataset" / "hmmt_feb_2025.jsonl",
}


def normalize_base_url(value: str) -> str:
    value = value.rstrip("/")
    return value if value.endswith("/v1") else f"{value}/v1"


def request_json(
    url: str,
    payload: dict[str, Any] | None,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc.reason}") from exc


def discover_model(base_url: str, api_key: str | None, timeout: float) -> str:
    response = request_json(f"{base_url}/models", None, api_key, timeout)
    models = response.get("data") or []
    if not models or "id" not in models[0]:
        raise RuntimeError(f"/models returned no usable model: {response}")
    return str(models[0]["id"])


def load_dataset(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("problem"), str) or not isinstance(row.get("solution"), str):
                raise ValueError(f"{path}:{line_number} requires string problem and solution fields")
            rows.append({"problem": row["problem"], "solution": row["solution"]})
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def build_run_fingerprint(args: argparse.Namespace, dataset_path: Path) -> str:
    dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    settings = {
        "model": args.model,
        "dataset": dataset_digest,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "thinking": args.thinking,
        "system_prompt": args.system_prompt,
    }
    encoded = json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_messages(problem: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.lower() != "none":
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": problem})
    return messages


def run_one(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": build_messages(job["problem"], args.system_prompt),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.thinking:
        payload["chat_template_kwargs"] = {"thinking": True}

    error: Exception | None = None
    for attempt in range(args.retries + 1):
        started = time.monotonic()
        try:
            response = request_json(
                f"{args.base_url}/chat/completions",
                payload,
                args.api_key,
                args.timeout,
            )
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError(f"response has no choices: {response}")
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            if not content and not reasoning:
                raise RuntimeError(f"response contains neither content nor reasoning_content: {response}")
            return {
                "type": "sample",
                "job_id": job["job_id"],
                "dataset_index": job["dataset_index"],
                "repeat": job["repeat"],
                "input": job["problem"],
                "output": job["solution"],
                "prediction": [{
                    # With the V4 reasoning parser, an empty content field means
                    # no final answer was produced (usually max-token truncation).
                    # Do not grade the private reasoning trace as the answer.
                    "solution": content,
                    "reasoning_content": reasoning,
                }],
                "latency_seconds": round(time.monotonic() - started, 6),
                "usage": response.get("usage"),
                "finish_reason": choices[0].get("finish_reason"),
            }
        except Exception as exc:  # retries must include malformed server responses
            error = exc
            if attempt < args.retries:
                time.sleep(min(2 ** attempt, 8) + random.random() * 0.25)
    raise RuntimeError(f"job {job['job_id']} failed after {args.retries + 1} attempts: {error}")


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt resume file {path}:{line_number}: {exc}") from exc
            if record.get("type") == "sample" and record.get("job_id"):
                completed[str(record["job_id"])] = record
    return completed


def score_results(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = [record["prediction"][0]["solution"] for record in records]
    references = [record["output"] for record in records]
    evaluator = MATHEvaluator()
    correctness = evaluator.score(predictions, references)
    if len(correctness) != len(records):
        raise RuntimeError(
            f"evaluator returned {len(correctness)} scores for {len(records)} records"
        )
    total_correct = 0
    for record, correct in zip(records, correctness):
        value = bool(correct)
        record["prediction"][0]["correctness"] = value
        total_correct += int(value)
    total = len(records)
    summary = {
        "type": "summary",
        "accuracy": round(total_correct / total * 100, 2) if total else 0.0,
        "correct": total_correct,
        "total": total,
        "evaluator_backend": getattr(evaluator, "backend", type(evaluator).__name__),
    }
    return records, summary


def score_record(record: dict[str, Any], evaluator: Any) -> bool:
    correctness = evaluator.score(
        [record["prediction"][0]["solution"]], [record["output"]]
    )
    if len(correctness) != 1:
        raise RuntimeError(
            f"evaluator returned {len(correctness)} scores for one record"
        )
    value = bool(correctness[0])
    record["prediction"][0]["correctness"] = value
    return value


def completion_tokens_per_second(record: dict[str, Any]) -> str:
    """Return the request-average completion speed for progress logging."""
    usage = record.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    latency_seconds = record.get("latency_seconds")
    if (
        isinstance(completion_tokens, (int, float))
        and not isinstance(completion_tokens, bool)
        and isinstance(latency_seconds, (int, float))
        and not isinstance(latency_seconds, bool)
        and latency_seconds > 0
    ):
        return f"{completion_tokens / latency_seconds:.2f}"
    return "unknown"


def write_final(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a DeepSeek/SGLang OpenAI-compatible endpoint")
    parser.add_argument("--data-name", "--data_name", dest="data_name", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset-path", type=Path, help="Override the built-in dataset JSONL")
    parser.add_argument("--target-path", "--target_path", dest="target_path", type=Path, required=True)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:60000/v1")
    parser.add_argument("--model", "--model_name_or_path", dest="model")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", "--number", dest="repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--system-prompt", "--system_prompt", dest="system_prompt", default="none")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.repeats < 1 or args.max_tokens < 1:
        raise ValueError("workers, repeats and max_tokens must be positive")
    args.base_url = normalize_base_url(args.base_url)
    if not args.model:
        args.model = discover_model(args.base_url, args.api_key, min(args.timeout, 60.0))
    dataset_path = args.dataset_path or DATASETS[args.data_name]
    dataset = load_dataset(dataset_path)
    run_fingerprint = build_run_fingerprint(args, dataset_path)

    args.target_path.mkdir(parents=True, exist_ok=True)
    output_path = args.output_file or (
        args.target_path / f"{args.data_name}-v4-L_{args.max_tokens}-n_{args.repeats}.jsonl"
    )
    partial_path = output_path.with_suffix(".partial.jsonl")
    if not args.resume:
        partial_path.unlink(missing_ok=True)
    completed = read_completed(partial_path)

    jobs = [
        {
            "job_id": f"{run_fingerprint}:{repeat}:{index}",
            "dataset_index": index,
            "repeat": repeat,
            "problem": row["problem"],
            "solution": row["solution"],
        }
        for repeat in range(args.repeats)
        for index, row in enumerate(dataset)
    ]
    pending = [job for job in jobs if job["job_id"] not in completed]
    resumed_jobs = len(jobs) - len(pending)
    print(
        f"dataset={args.data_name} model={args.model} total={len(jobs)} "
        f"resumed={resumed_jobs} pending={len(pending)} workers={args.workers} "
        f"run={run_fingerprint}"
    )

    failures: list[str] = []
    if pending:
        evaluator = MATHEvaluator()
        with partial_path.open("a", encoding="utf-8") as checkpoint, ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = {pool.submit(run_one, job, args): job for job in pending}
            finished = 0
            for future in as_completed(futures):
                job = futures[future]
                record: dict[str, Any] | None = None
                try:
                    record = future.result()
                except Exception as exc:
                    failures.append(str(exc))
                    print(
                        f"question={job['dataset_index'] + 1}/{len(dataset)} "
                        f"repeat={job['repeat'] + 1}/{args.repeats} "
                        f"status=error error={exc}",
                        flush=True,
                    )
                else:
                    assert record is not None
                    correct = score_record(record, evaluator)
                    completed[record["job_id"]] = record
                    checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                finished += 1
                if record is not None:
                    usage = record.get("usage") or {}
                    completion_tokens = usage.get("completion_tokens", "unknown")
                    average_completion_tokens_per_second = completion_tokens_per_second(record)
                    print(
                        f"question={record['dataset_index'] + 1}/{len(dataset)} "
                        f"repeat={record['repeat'] + 1}/{args.repeats} "
                        f"correct={'yes' if correct else 'no'} "
                        f"latency={record['latency_seconds']:.2f}s "
                        f"completion_tokens={completion_tokens} "
                        f"token_ps="
                        f"{average_completion_tokens_per_second} "
                        f"progress={resumed_jobs + finished}/{len(jobs)} "
                        f"failures={len(failures)}",
                        flush=True,
                    )

    if failures:
        for failure in failures:
            print(f"[ERROR] {failure}")
        print(f"Partial results retained at {partial_path}; rerun the same command to resume")
        return 2

    missing = [job["job_id"] for job in jobs if job["job_id"] not in completed]
    if missing:
        raise RuntimeError(f"missing {len(missing)} jobs after evaluation: {missing[:5]}")
    records = [completed[job["job_id"]] for job in jobs]
    records, summary = score_results(records)
    summary.update({
        "dataset": args.data_name,
        "model": args.model,
        "run_fingerprint": run_fingerprint,
        "output_file": str(output_path),
    })
    write_final(output_path, records, summary)
    print(
        f"{args.data_name}: {summary['accuracy']}% "
        f"({summary['correct']} / {summary['total']}); output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
