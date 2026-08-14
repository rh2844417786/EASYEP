#!/usr/bin/env python3
"""Small OpenAI-compatible smoke request with no external dependency."""

from __future__ import annotations

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, payload: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc.reason}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:60000/v1")
    parser.add_argument("--model", help="API model name; defaults to the first /models entry")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = args.model
    if not model:
        models = request_json(f"{base_url}/models", timeout=args.timeout)
        entries = models.get("data") or []
        if not entries:
            raise RuntimeError("/v1/models returned no model")
        model = entries[0]["id"]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Return only: READY"}],
        "temperature": 0,
        "max_tokens": 16,
    }
    started = time.monotonic()
    response = request_json(f"{base_url}/chat/completions", payload, args.timeout)
    elapsed = time.monotonic() - started
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in response: {response}")
    message = choices[0].get("message") or {}
    print(json.dumps({
        "ok": True,
        "model": model,
        "elapsed_seconds": round(elapsed, 3),
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "usage": response.get("usage"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
