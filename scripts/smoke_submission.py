#!/usr/bin/env python3
"""Public, GPU-side smoke test for a running submission server.

This verifies the visible API contract. It does not reproduce the instructor's
hidden quality, canary, latency, or speed measurements.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def check(base_url: str, timeout: float) -> None:
    models = get_json(f"{base_url}/v1/models", timeout)
    names = {item.get("id") for item in models.get("data", [])}
    if "submission" not in names:
        raise ValueError("/v1/models does not advertise model 'submission'")

    payload = {
        "model": "submission",
        "messages": [{"role": "user", "content": "Explain TTFT in one sentence."}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    text_parts = []
    saw_done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise ValueError("chat completion did not return a streaming response")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(data)
            text_parts.append(chunk["choices"][0].get("delta", {}).get("content") or "")
    if not saw_done or not "".join(text_parts).strip():
        raise ValueError("stream must contain text and end with [DONE]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    try:
        check(args.base_url.rstrip("/"), args.timeout)
    except (ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as error:
        print(f"smoke check failed: {error}", file=sys.stderr)
        return 1
    print("smoke check passed: model metadata and streaming chat endpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
