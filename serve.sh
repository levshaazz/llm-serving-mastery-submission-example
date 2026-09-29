#!/usr/bin/env bash
# The judge runs exactly this from a clean checkout of your tag.
# Contract: OpenAI-compatible server on 0.0.0.0:8000, model name "submission", streaming chat completions.
set -euo pipefail
SOURCE_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SOURCE_DIR"
if [[ -x /runtime/.venv/bin/vllm ]]; then
  VLLM=(/runtime/.venv/bin/vllm)
else
  uv sync --frozen
  VLLM=(uv run --frozen vllm)
fi

MODEL_ID=Qwen/Qwen2.5-3B-Instruct
MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
MODEL_KEY=models--Qwen--Qwen2.5-3B-Instruct
SNAPSHOT="${HF_HOME:-$HOME/.cache/huggingface}/hub/$MODEL_KEY/snapshots/$MODEL_REVISION"

# The judge prefetches the immutable snapshot while egress is available, then starts this script on
# an internal network. Make the Hub id resolve as a local relative path so vLLM keeps the declared
# model root in /v1/models without attempting a Hub API lookup during offline measurement.
if [[ -d "$SNAPSHOT" ]]; then
  OFFLINE_ROOT=/tmp/lsm-models
  mkdir -p "$OFFLINE_ROOT/$(dirname "$MODEL_ID")"
  ln -sfn "$SNAPSHOT" "$OFFLINE_ROOT/$MODEL_ID"
  cd "$OFFLINE_ROOT"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

# Model Runner V2 requires UVA-backed host buffers. CUDA on WSL2 does not expose that facility, so
# use vLLM's supported V1 runner there; native Linux keeps the v0.29 default.
if [[ -n "${WSL_INTEROP:-}" || "$(uname -r)" == *[Mm]icrosoft* ]]; then
  export VLLM_USE_V2_MODEL_RUNNER=0
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi

exec "${VLLM[@]}" serve "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --served-model-name submission \
  --host 0.0.0.0 --port 8000 \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
