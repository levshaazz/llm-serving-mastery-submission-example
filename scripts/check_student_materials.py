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
        if name.startswith(("01-", "02-", "03-", "04-")):
            if not all(term in source for term in ("LSM_MODEL_CACHE", "snapshot_download", "local_files_only=True")):
                raise ValueError(f"{name}: persistent pinned local model cache contract missing")
            if "local_dir=MODEL_DIR" in source or not re.search(r"[0-9a-f]{40}", source):
                raise ValueError(f"{name}: immutable model revision/cache integrity contract missing")
            if not all(term in source for term in ("PackageNotFoundError", "version(name)" if not name.startswith("04-") else "version('vllm')", "if not installed")):
                raise ValueError(f"{name}: repeated package installation is not guarded")
        if name.startswith("01-") and not all(term in source for term in ("worker.join(timeout=30)", "TextIteratorStreamer", "timeout=30")):
            raise ValueError(f"{name}: streaming worker needs a bounded timeout")
        if name.startswith("03-") and not all(term in source for term in ("quality_sample", "timing_trials", "force_length=True", "natural_eos")):
            raise ValueError(f"{name}: quality and timing must have separate stopping rules")
        if name.startswith("04-") and not all(term in source for term in ("IS_WSL", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_FLASHINFER_SAMPLER", "rendered['input_ids']", "if rendered_prompt_tokens > 2048: break")):
            raise ValueError(f"{name}: WSL2 fallback or actual over-context token count missing")
        code_cells = 0
        cell_ids = set()
        for index, cell in enumerate(data["cells"], 1):
            cell_id = cell.get("id")
            if not isinstance(cell_id, str) or not cell_id or cell_id in cell_ids:
                raise ValueError(f"{name} cell {index}: missing or duplicate cell ID")
            cell_ids.add(cell_id)
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
