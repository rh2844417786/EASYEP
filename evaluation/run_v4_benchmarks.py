#!/usr/bin/env python3
"""Run local Agent-OS, GPQA, and LiveCodeBench data against a V4 SGLang API.

The source Arrow files are local EASY-EP calibration/evaluation artifacts, not
the official interactive harnesses.  Each scorer therefore records its exact
scope: GPQA is multiple-choice accuracy, Agent-OS is next-action matching from
an offline trajectory, and LiveCodeBench is Python sample-I/O pass rate.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"
DATASETS = {
    "agent_os": DATASET_ROOT / "agent_os_full_ids" / "data-00000-of-00001.arrow",
    "gpqa": DATASET_ROOT / "gpqa_past_full" / "data-00000-of-00001.arrow",
    # Keep the requested public name while using the repository's local data.
    "kuvecodebench": DATASET_ROOT / "livecodebench_v3_full_ids" / "data-00000-of-00001.arrow",
}
SCORING_VERSIONS = {
    # Bump this when sample extraction changes so stale false/unknown records
    # are not reused by --resume.
    "kuvecodebench": "python_samples_v3_leetcode_method",
}

BOXED_OPTION_RE = re.compile(r"\\boxed\{\s*([ABCD])\s*\}", re.I)
OPTION_RE = re.compile(r"(?:answer|option)(?:\s+is)?\s*[:=]?\s*([ABCD])\b", re.I)
STANDALONE_OPTION_RE = re.compile(r"^\s*([ABCD])\s*$", re.I | re.M)
ACTION_RE = re.compile(r"Act:\s*(bash|finish|answer)\b", re.I)
FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.I | re.S)
SAMPLE_RE = re.compile(
    r"Sample Input(?:\s+\d+)?\s*\n(.*?)\n\s*Sample Output(?:\s+\d+)?\s*\n(.*?)(?=\n\s*Sample Input|\Z)",
    re.I | re.S,
)
LEETCODE_INPUT_RE = re.compile(r"^\s*Input:\s*(.+?)\s*$", re.I | re.M)
LEETCODE_OUTPUT_RE = re.compile(r"^\s*Output:\s*(.+?)\s*$", re.I | re.M)


@dataclass(frozen=True)
class Sample:
    prompt: str
    reference: str | None
    metadata: dict[str, Any]


def normalize_base_url(value: str) -> str:
    value = value.rstrip("/")
    return value if value.endswith("/v1") else f"{value}/v1"


def request_json(url: str, payload: dict[str, Any] | None, api_key: str | None, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=None if payload is None else json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}: {exc.read().decode('utf-8', errors='replace')}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc.reason}") from exc


def discover_model(base_url: str, api_key: str | None, timeout: float) -> str:
    models = request_json(f"{base_url}/models", None, api_key, timeout).get("data") or []
    if not models or not models[0].get("id"):
        raise RuntimeError("/v1/models returned no model id")
    return str(models[0]["id"])


def load_arrow_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for local V4 benchmark Arrow files") from exc
    with pa.memory_map(str(path), "r") as source:
        try:
            table = ipc.RecordBatchStreamReader(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            table = ipc.RecordBatchFileReader(source).read_all()
    return table.to_pylist()


def last_option(value: str) -> str | None:
    matches = [(match.start(1), match.group(1)) for match in BOXED_OPTION_RE.finditer(value)]
    matches.extend((match.start(1), match.group(1)) for match in OPTION_RE.finditer(value))
    matches.extend((match.start(1), match.group(1)) for match in STANDALONE_OPTION_RE.finditer(value))
    return max(matches, default=(-1, ""))[1].upper() or None


def last_action(value: str) -> str | None:
    matches = ACTION_RE.findall(value)
    return matches[-1].lower() if matches else None


def extract_code(value: str) -> str:
    matches = FENCED_CODE_RE.findall(value)
    return (matches[-1] if matches else value).strip()


def normalize_output(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().replace("\r\n", "\n").splitlines()).strip()


def extract_samples(question: str) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for input_text, output_block in SAMPLE_RE.findall(question):
        # LiveCodeBench includes explanations immediately after a blank line
        # following each sample output. Keep only the first paragraph, which is
        # the actual judge output; otherwise valid short answers become false.
        output_text = re.split(r"\n\s*\n", output_block.replace("\r\n", "\n"), maxsplit=1)[0]
        samples.append((input_text.strip() + "\n", normalize_output(output_text)))
    return samples


def _split_assignments(value: str) -> list[str]:
    """Split comma-separated assignments without splitting nested literals."""
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in "'\"":
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def _literal(value: str) -> Any:
    normalized = re.sub(r"\btrue\b", "True", value, flags=re.I)
    normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.I)
    normalized = re.sub(r"\bnull\b", "None", normalized, flags=re.I)
    return ast.literal_eval(normalized.strip())


def _leetcode_method(starter_code: str) -> tuple[str | None, list[str]]:
    try:
        tree = ast.parse(starter_code)
    except SyntaxError:
        # The local artifact stores the starter method with an empty body.
        # Add a synthetic pass solely for signature discovery.
        try:
            tree = ast.parse(starter_code.rstrip() + "\n        pass\n")
        except SyntaxError:
            return None, []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    arguments = [argument.arg for argument in member.args.args if argument.arg != "self"]
                    return member.name, arguments
    return None, []


def extract_leetcode_samples(question: str, starter_code: str) -> tuple[str | None, list[tuple[list[Any], Any]]]:
    method, parameter_names = _leetcode_method(starter_code)
    if not method:
        return None, []
    inputs = LEETCODE_INPUT_RE.findall(question)
    outputs = LEETCODE_OUTPUT_RE.findall(question)
    cases: list[tuple[list[Any], Any]] = []
    for input_text, output_text in zip(inputs, outputs):
        assignments: dict[str, Any] = {}
        try:
            for assignment in _split_assignments(input_text):
                name, literal = assignment.split("=", 1)
                assignments[name.strip()] = _literal(literal)
            arguments = [assignments[name] for name in parameter_names]
            cases.append((arguments, _literal(output_text)))
        except (ValueError, SyntaxError, KeyError):
            continue
    return method, cases


def build_samples(data_name: str, rows: list[dict[str, Any]]) -> list[Sample]:
    samples: list[Sample] = []
    for row in rows:
        if data_name == "gpqa":
            prompt = f"{row['input']}\n\nReply with only the correct option letter: A, B, C, or D."
            samples.append(Sample(prompt, last_option(str(row["output"])), {"scoring": "multiple_choice"}))
        elif data_name == "agent_os":
            transcript = str(row["text"])
            if "<｜Assistant｜>" not in transcript:
                raise ValueError("Agent-OS row lacks an assistant-turn marker")
            prefix, expected = transcript.rsplit("<｜Assistant｜>", 1)
            prompt = (
                "Continue the following Agent-OS transcript. Produce exactly one next action "
                "using the required Think/Act format. Do not describe the benchmark.\n\n"
                f"{prefix}<｜Assistant｜>"
            )
            samples.append(Sample(prompt, last_action(expected), {"scoring": "next_action"}))
        else:
            question = str(row["question_content"])
            starter = str(row.get("starter_code") or "")
            method, leetcode_cases = extract_leetcode_samples(question, starter)
            prompt = (
                "Solve this programming problem in Python 3. Return only executable code in a "
                "single markdown code fence.\n\n"
                f"Title: {row['question_title']}\n\n{question}"
                + (f"\n\nStarter code:\n```python\n{starter}\n```" if starter else "")
            )
            samples.append(Sample(prompt, None, {
                "scoring": "python_samples",
                "platform": str(row.get("platform") or ""),
                "samples": extract_samples(question),
                "method": method,
                "leetcode_cases": leetcode_cases,
            }))
    return samples


def score_sample(data_name: str, sample: Sample, response: str, python: str, timeout: float) -> tuple[bool | None, dict[str, Any]]:
    if data_name == "gpqa":
        predicted = last_option(response)
        return predicted == sample.reference, {"reference": sample.reference, "prediction": predicted}
    if data_name == "agent_os":
        predicted = last_action(response)
        return predicted == sample.reference, {"reference_action": sample.reference, "predicted_action": predicted}

    if sample.metadata["platform"] == "leetcode":
        cases: list[tuple[list[Any], Any]] = sample.metadata.get("leetcode_cases", [])
        method = sample.metadata.get("method")
        if not method or not cases:
            return None, {"reason": "no_runnable_leetcode_examples"}
        code = extract_code(response)
        if not code:
            return False, {"reason": "no_code"}
        return _score_leetcode_method(code, method, cases, python, timeout)

    cases: list[tuple[str, str]] = sample.metadata["samples"]
    if sample.metadata["platform"] != "atcoder" or not cases:
        return None, {"reason": "no_runnable_python_sample_cases"}
    code = extract_code(response)
    if not code:
        return False, {"reason": "no_code"}
    outcomes: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="easyep_lcb_") as directory:
            program = Path(directory) / "solution.py"
            program.write_text(code + "\n", encoding="utf-8")
            for input_text, expected in cases:
                completed = subprocess.run(
                    [python, str(program)], input=input_text, text=True, capture_output=True,
                    timeout=timeout, cwd=directory, check=False,
                )
                actual = normalize_output(completed.stdout)
                outcomes.append({"returncode": completed.returncode, "expected": expected, "actual": actual})
                if completed.returncode != 0 or actual != expected:
                    return False, {"samples": outcomes}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, {"reason": f"execution_error:{type(exc).__name__}", "samples": outcomes}
    return True, {"samples": outcomes}


def _score_leetcode_method(
    code: str, method: str, cases: list[tuple[list[Any], Any]], python: str, timeout: float,
) -> tuple[bool, dict[str, Any]]:
    runner = r'''import contextlib
import io
import json
import math
import sys
import typing
from bisect import *
from collections import *
from functools import *
from heapq import *
from itertools import *
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
method_name = sys.argv[2]
cases = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
namespace = {"__name__": "__easyep_solution__"}
namespace.update(vars(typing))
namespace.update({
    "typing": typing, "math": math, "List": typing.List, "Optional": typing.Optional,
    "Deque": typing.Deque, "Dict": typing.Dict, "Set": typing.Set, "Tuple": typing.Tuple,
})
exec(compile(source, str(sys.argv[1]), "exec"), namespace, namespace)
solution_type = namespace.get("Solution")
if solution_type is None:
    raise RuntimeError("generated code does not define class Solution")
results = []
for arguments in cases:
    instance = solution_type()
    with contextlib.redirect_stdout(io.StringIO()):
        result = getattr(instance, method_name)(*arguments)
    results.append(result)
print(json.dumps(results, ensure_ascii=False))
'''
    outcomes: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="easyep_leetcode_") as directory:
        root = Path(directory)
        solution_path = root / "solution.py"
        runner_path = root / "runner.py"
        cases_path = root / "cases.json"
        solution_path.write_text(code + "\n", encoding="utf-8")
        runner_path.write_text(runner, encoding="utf-8")
        cases_path.write_text(json.dumps([arguments for arguments, _ in cases], ensure_ascii=False), encoding="utf-8")
        try:
            completed = subprocess.run(
                [python, str(runner_path), str(solution_path), method, str(cases_path)],
                text=True, capture_output=True, timeout=timeout, cwd=directory, check=False,
            )
            if completed.returncode != 0:
                return False, {"reason": "execution_error", "stderr": completed.stderr[-2000:]}
            actual_values = json.loads(completed.stdout)
            if len(actual_values) != len(cases):
                return False, {"reason": "invalid_result_count"}
            for actual, (_, expected) in zip(actual_values, cases):
                outcome = {"expected": expected, "actual": actual}
                outcomes.append(outcome)
                if actual != expected:
                    return False, {"examples": outcomes}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, TypeError) as exc:
            return False, {"reason": f"execution_error:{type(exc).__name__}", "examples": outcomes}
    return True, {"examples": outcomes}


def fingerprint(args: argparse.Namespace, data_path: Path) -> str:
    payload = {
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset": args.data_name,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "thinking": args.thinking,
        "code_timeout": args.code_timeout,
    }
    # Keep the historical fingerprint for Agent-OS and GPQA so their valid
    # checkpoints remain resumable; only the changed KuveCodeBench scorer gets
    # a new identity.
    if args.data_name in SCORING_VERSIONS:
        payload["scoring_version"] = SCORING_VERSIONS[args.data_name]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt checkpoint {path}:{number}") from exc
        if record.get("type") == "sample" and record.get("job_id"):
            completed[str(record["job_id"])] = record
    return completed


def token_ps(record: dict[str, Any]) -> str:
    usage = record.get("usage") or {}
    tokens, latency = usage.get("completion_tokens"), record.get("latency_seconds")
    if isinstance(tokens, (int, float)) and isinstance(latency, (int, float)) and latency > 0:
        return f"{tokens / latency:.2f}"
    return "unknown"


def run_one(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": job["sample"].prompt}],
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
            response = request_json(f"{args.base_url}/chat/completions", payload, args.api_key, args.timeout)
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError("response has no choices")
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "")
            reasoning = str(message.get("reasoning_content") or "")
            if not content and not reasoning:
                raise RuntimeError("response has neither content nor reasoning_content")
            correct, details = score_sample(args.data_name, job["sample"], content, args.code_python, args.code_timeout)
            return {
                "type": "sample", "job_id": job["job_id"], "dataset_index": job["dataset_index"], "repeat": job["repeat"],
                "input": job["sample"].prompt, "output": job["sample"].reference,
                "prediction": [{"solution": content, "reasoning_content": reasoning, "correctness": correct}],
                "scoring": details, "latency_seconds": round(time.monotonic() - started, 6),
                "usage": response.get("usage"), "finish_reason": choices[0].get("finish_reason"),
            }
        except Exception as exc:
            error = exc
            if attempt < args.retries:
                time.sleep(min(2 ** attempt, 8) + random.random() * 0.25)
    raise RuntimeError(f"job {job['job_id']} failed after {args.retries + 1} attempts: {error}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Evaluate local Agent-OS, GPQA, and KuveCodeBench V4 data")
    value.add_argument("--data-name", choices=sorted(DATASETS), required=True)
    value.add_argument("--dataset-path", type=Path)
    value.add_argument("--target-path", type=Path, required=True)
    value.add_argument("--output-file", type=Path)
    value.add_argument("--base-url", default="http://127.0.0.1:60000/v1")
    value.add_argument("--model")
    value.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    value.add_argument("--max-tokens", type=int, default=32768)
    value.add_argument("--workers", type=int, default=1)
    value.add_argument("--repeats", type=int, default=1)
    value.add_argument("--temperature", type=float, default=1.0)
    value.add_argument("--top-p", type=float, default=1.0)
    value.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--timeout", type=float, default=3600.0)
    value.add_argument("--retries", type=int, default=2)
    value.add_argument("--code-python", default=sys.executable)
    value.add_argument("--code-timeout", type=float, default=3.0)
    value.add_argument("--limit", type=int)
    value.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if min(args.max_tokens, args.workers, args.repeats) < 1 or args.code_timeout <= 0:
        raise ValueError("max-tokens, workers, repeats, and code-timeout must be positive")
    args.base_url = normalize_base_url(args.base_url)
    if not args.model:
        args.model = discover_model(args.base_url, args.api_key, min(args.timeout, 60.0))
    data_path = args.dataset_path or DATASETS[args.data_name]
    samples = build_samples(args.data_name, load_arrow_rows(data_path))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:args.limit]
    run_id = fingerprint(args, data_path)
    args.target_path.mkdir(parents=True, exist_ok=True)
    output = args.output_file or args.target_path / f"{args.data_name}.jsonl"
    partial = output.with_suffix(".partial.jsonl")
    if not args.resume:
        partial.unlink(missing_ok=True)
    completed = read_completed(partial)
    jobs = [{"job_id": f"{run_id}:{repeat}:{index}", "dataset_index": index, "repeat": repeat, "sample": sample}
            for repeat in range(args.repeats) for index, sample in enumerate(samples)]
    pending = [job for job in jobs if job["job_id"] not in completed]
    resumed = len(jobs) - len(pending)
    print(f"dataset={args.data_name} model={args.model} total={len(jobs)} resumed={resumed} pending={len(pending)} workers={args.workers} run={run_id}", flush=True)
    failures: list[str] = []
    with partial.open("a", encoding="utf-8") as checkpoint, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, job, args): job for job in pending}
        finished = 0
        for future in as_completed(futures):
            job = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                failures.append(str(exc))
                print(f"question={job['dataset_index'] + 1}/{len(samples)} repeat={job['repeat'] + 1}/{args.repeats} status=error error={exc}", flush=True)
            else:
                completed[record["job_id"]] = record
                checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                checkpoint.flush()
                correctness = record["prediction"][0]["correctness"]
                label = "yes" if correctness is True else "no" if correctness is False else "unknown"
                usage = record.get("usage") or {}
                print(
                    f"question={record['dataset_index'] + 1}/{len(samples)} repeat={record['repeat'] + 1}/{args.repeats} "
                    f"correct={label} latency={record['latency_seconds']:.2f}s completion_tokens={usage.get('completion_tokens', 'unknown')} "
                    f"token_ps={token_ps(record)} progress={resumed + finished + 1}/{len(jobs)} failures={len(failures)}",
                    flush=True,
                )
            finished += 1
    if failures:
        for failure in failures:
            print(f"[ERROR] {failure}", flush=True)
        return 2
    records = [completed[job["job_id"]] for job in jobs]
    scored = [item for item in records if item["prediction"][0]["correctness"] is not None]
    correct = sum(item["prediction"][0]["correctness"] is True for item in scored)
    summary = {
        "type": "summary", "dataset": args.data_name, "model": args.model, "run_fingerprint": run_id,
        "correct": correct, "scored_total": len(scored), "unscored": len(records) - len(scored), "total": len(records),
        "accuracy": round(correct / len(scored) * 100, 2) if scored else None,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(f"{args.data_name}: {summary['accuracy']}% ({correct} / {len(scored)} scored; {summary['unscored']} unscored); output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
