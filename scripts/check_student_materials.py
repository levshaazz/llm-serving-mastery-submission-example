#!/usr/bin/env python3
"""Fail CI when public course notebooks disappear or link to private source."""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "00-live-demo-pipeline-vs-vllm.ipynb",
    "01-serving-system-baseline.ipynb",
    "02-gpu-profiling-and-bottlenecks.ipynb",
    "03-quantization-tradeoffs.ipynb",
    "04-vllm-openai-serving.ipynb",
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
        if name.startswith(("02-", "03-", "04-")):
            if not all(term in source for term in ("LSM_MODEL_CACHE", "snapshot_download", "local_files_only=True")):
                raise ValueError(f"{name}: persistent pinned local model cache contract missing")
            if "local_dir=MODEL_DIR" in source or not re.search(r"[0-9a-f]{40}", source):
                raise ValueError(f"{name}: immutable model revision/cache integrity contract missing")
        code_cells = 0
        for index, cell in enumerate(data["cells"], 1):
            if cell.get("outputs") or cell.get("execution_count") is not None:
                raise ValueError(f"{name} cell {index}: generated execution state")
            if cell.get("cell_type") != "code":
                continue
            code_cells += 1
            code = "".join(cell.get("source", []))
            if not code.lstrip().startswith(("%", "!")):
                ast.parse(code, filename=f"{name}:cell-{index}")
        if not code_cells:
            raise ValueError(f"{name}: no executable cells")
        print(f"{name}: {len(data['cells'])} cells, {code_cells} code cells passed")


if __name__ == "__main__":
    main()
