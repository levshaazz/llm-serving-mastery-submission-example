#!/usr/bin/env python3
"""CPU-only structure/arithmetic check, not a proof that a benchmark is genuine."""
import argparse
import json
import math
from pathlib import Path

MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}
CASES = [(1, 128, 32), (1, 4096, 32), (1, 128, 128), (1, 4096, 128), (4, 128, 32)]


def validate(data):
    if (data.get("protocol"), data.get("status"), data.get("scope"), data.get("trials")) != (
            "topic02-phase-v2", "complete", "core", 5):
        raise ValueError("Expected a completed core protocol-v2 run with five trials; smoke/partial is not complete")
    if data.get("models") != MODELS or sorted(data.get("cases", [])) != sorted(map(list, CASES)):
        raise ValueError("Pinned model revisions or required cases differ")
    env = data.get("environment", {})
    if any(env.get(k) != v for k, v in {"dtype": "float16", "transformers": "4.51.3",
                                        "attention_requested": "sdpa", "logits_to_keep": 1}.items()):
        raise ValueError("Protocol runtime settings differ")
    if not all(env.get(k) for k in ("python", "torch", "cuda", "gpu")):
        raise ValueError("Missing environment provenance")
    expected = {(model, *case, repeat) for model in MODELS for case in CASES for repeat in range(5)}
    seen = set()
    for row in data.get("results", []):
        key = tuple(row[k] for k in ("model", "batch", "input_tokens", "output_tokens", "repeat"))
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate model/shape/trial")
        seen.add(key)
        model, batch, prompt, output, repeat = key
        for metric in ("prefill_wall_ms", "prefill_event_ms", "decode_wall_ms", "decode_event_ms",
                       "decode_aggregate_tokens_per_s", "decode_mean_step_ms", "peak_allocated_bytes",
                       "peak_reserved_bytes", "prefill_peak_allocated_bytes"):
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{metric} must be a finite positive measurement")
        if (row.get("decode_steps_per_sequence"), row.get("generated_tokens"),
            row.get("retained_tokens_per_sequence")) != (output - 1, batch * output, prompt + output - 1):
            raise ValueError("Output/step/KV accounting mismatch")
        if not math.isclose(row["decode_aggregate_tokens_per_s"], batch * (output - 1) * 1000 / row["decode_wall_ms"], rel_tol=1e-7):
            raise ValueError("Aggregate decode rate uses the wrong numerator or clock")
        if not math.isclose(row["decode_mean_step_ms"], row["decode_wall_ms"] / (output - 1), rel_tol=1e-7):
            raise ValueError("Mean step interval uses the wrong denominator")
        if row["peak_reserved_bytes"] < row["peak_allocated_bytes"] or row["peak_allocated_bytes"] < row["prefill_peak_allocated_bytes"]:
            raise ValueError("Impossible allocated/reserved peak ordering")
    if seen != expected:
        raise ValueError(f"Missing trials: found {len(seen)} of {len(expected)}")
    return len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    count = validate(json.loads(args.artifact.read_text(encoding="utf-8")))
    print(f"Topic 02: {count} raw trials, pinned settings and arithmetic passed; authenticity needs review")
