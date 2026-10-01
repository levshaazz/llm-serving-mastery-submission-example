#!/usr/bin/env python3
"""Static, GPU-free gate for student-facing notebooks."""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "seminars").glob("[0-9][0-9]-*.ipynb"))


def main() -> None:
    if len(NOTEBOOKS) < 5:
        raise SystemExit("expected Topics 00–04 notebooks")
    for path in NOTEBOOKS:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("nbformat") != 4 or not data.get("cells"):
            raise ValueError(f"{path.name}: invalid or empty notebook")
        all_source = "".join("".join(cell.get("source", [])) for cell in data["cells"])
        if path.name.startswith(("01-", "02-", "03-", "04-")):
            if not all(term in all_source for term in ("LSM_MODEL_CACHE", "snapshot_download", "local_files_only=True")):
                raise ValueError(f"{path.name}: persistent pinned local model cache contract missing")
            if "local_dir=MODEL_DIR" in all_source or not re.search(r"[0-9a-f]{40}", all_source):
                raise ValueError(f"{path.name}: immutable model revision/cache integrity contract missing")
            install_guard = "if not matches" if path.name.startswith(("03-", "04-")) else "if not installed"
            if not all(term in all_source for term in ("PackageNotFoundError", "version(name)" if not path.name.startswith("04-") else "version('vllm')", install_guard)):
                raise ValueError(f"{path.name}: pinned package installation must be conditional")
        if path.name.startswith("01-") and not all(
            term in all_source for term in ("worker.join(timeout=30)", "TextIteratorStreamer", "timeout=30")
        ):
            raise ValueError(f"{path.name}: streaming thread needs a bounded timeout")
        if path.name.startswith("03-") and not all(
            term in all_source for term in ("topic-03-paired-smoke-v3", "quality_sample", "timing_trials", "min_new_tokens", "natural_eos", "fixed_output_length", "cleanup_verified", "checkpoint(output_path,data)", "validate_artifact", "RECORDED_QUANTIZATION", "toy_namespace")
        ):
            raise ValueError(f"{path.name}: canonical paired v3 clock/quality/checkpoint/CPU contract missing")
        if path.name.startswith("04-") and not all(
            term in all_source for term in ("topic-04-api-smoke-v3", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_FLASHINFER_SAMPLER", "env=env", "normalize_input_ids", "2048<len(rendered)<8192", "ChatStream", "stop_owned_process_group", "max_retries=0", "validate_artifact", "RECORDED_SERVICE", "parser_namespace")
        ):
            raise ValueError(f"{path.name}: canonical v3 WSL/context/byte-replay/owned-teardown contract missing")
        code_cells = 0
        cell_ids = set()
        for index, cell in enumerate(data["cells"], 1):
            cell_id = cell.get("id")
            if not isinstance(cell_id, str) or not cell_id or cell_id in cell_ids:
                raise ValueError(f"{path.name} cell {index}: missing or duplicate nbformat cell ID")
            cell_ids.add(cell_id)
            if cell.get("outputs") or cell.get("execution_count") is not None:
                raise ValueError(f"{path.name} cell {index}: published output/execution state")
            source = "".join(cell.get("source", []))
            if "github.com/levshaazz/llm-serving-mastery/" in source:
                raise ValueError(f"{path.name} cell {index}: private source link")
            if cell.get("cell_type") != "code":
                continue
            code_cells += 1
            if source.lstrip().startswith(("%", "!")):
                continue  # notebook runtime magic/shell cell
            ast.parse(source, filename=f"{path.name}:cell-{index}")
        if not code_cells:
            raise ValueError(f"{path.name}: no executable code")
        print(f"{path.name}: {len(data['cells'])} cells, {code_cells} code cells passed")


if __name__ == "__main__":
    main()
