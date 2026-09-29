#!/usr/bin/env bash
# Networked bootstrap phase. The scored server starts later on an internal Docker network.
set -euo pipefail
cd "$(dirname "$0")"
: "${UV_PROJECT_ENVIRONMENT:=/runtime/.venv}"
export UV_PROJECT_ENVIRONMENT
uv sync --frozen
uv run --frozen python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
)
PY
