#!/usr/bin/env python3
"""Fail CI when public course notebooks disappear or link to private source."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "00-live-demo-pipeline-vs-vllm.ipynb",
    "01-serving-system-baseline.ipynb",
    "02-gpu-profiling-and-bottlenecks.ipynb",
)


def main() -> None:
    for name in NOTEBOOKS:
        path = ROOT / "seminars" / name
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("nbformat") != 4 or not data.get("cells"):
            raise ValueError(f"{name}: invalid or empty notebook")
        source = "".join("".join(cell.get("source", [])) for cell in data["cells"])
        if "github.com/levshaazz/llm-serving-mastery/" in source:
            raise ValueError(f"{name}: private course repository link in student notebook")
        if any(cell.get("outputs") for cell in data["cells"]):
            raise ValueError(f"{name}: generated outputs must not be published")
        print(f"{name}: {len(data['cells'])} cells passed")


if __name__ == "__main__":
    main()
