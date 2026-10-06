#!/usr/bin/env python3
"""Static, GPU-free gate for student-facing notebooks."""

import ast
import json
import re
import subprocess
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
        pilot = path.name == "03-quantization-pilot.ipynb"
        if pilot:
            # This independent teaching protocol has student TODOs and an opt-in
            # runner. Do not accidentally impose the archived v3 runner's schema
            # on every file beginning with 03-, or silently skip the new contract.
            # Portable mirror gate: never depend on private course JS modules.
            roles = [c.get("metadata", {}).get("pilot_role") for c in data["cells"]]
            for role in ("student-scale", "student-groups", "methods-explanation", "real-load",
                         "real-calibration", "real-selection", "real-clipping", "real-nf4", "real-freeze",
                         "real-distributions", "real-held-out", "layer-explanation", "layer-gpu-bridge",
                         "quality-rule", "readiness-check", "gpu-optin"):
                if roles.count(role) != 1:
                    raise ValueError(f"{path.name}: missing/duplicate pilot role {role}")
            if not roles.index("real-calibration") < roles.index("real-held-out") < roles.index("gpu-optin"):
                raise ValueError("calibration, held-out, GPU order changed")
            if not roles.index("real-freeze") < roles.index("real-held-out") < roles.index("quality-rule") < roles.index("readiness-check") < roles.index("gpu-optin"):
                raise ValueError("frozen selection and pre-run readiness order changed")
            for term in ("def derive_scale(", "def quantize_groups(", "raise NotImplementedError",
                         "REAL_SIZES = [None]", "CHOSEN_GROUP = None", "output_error = None",
                         "RUN_GPU = False", "allow_pickle=False", "4.30", "topic03-layer-decision.json",
                         "CLIP_CANDIDATES = (100, 99, 99.5)", "CHOSEN_CLIP = None",
                         "NF4_RESTORED = nf4_qdq(W_REAL)", "FROZEN_CHOICE = json.dumps",
                         "check_gpu_readiness()", "FROZEN_QUALITY_RULE", "topic03-preflight.json",
                         "topic03-screening.json", "topic03-weights-center.png"):
                if term not in all_source:
                    raise ValueError(f"{path.name}: student contract missing {term}")
            import hashlib
            for name, key in (("topic03_pilot.py", "runner_sha256"),):
                local = ROOT / "teaching" / name
                if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() != data["metadata"]["topic03_pilot"][key]:
                    raise ValueError(f"{path.name}: mismatched runner")
        if not pilot and path.name.startswith(("01-", "02-", "03-", "04-")):
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
        if not pilot and path.name.startswith("03-") and not all(
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
