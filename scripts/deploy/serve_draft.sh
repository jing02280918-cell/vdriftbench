#!/usr/bin/env bash
# Starts the local draft (O-T-R free-generation) model with vLLM, per
# HANDOFF.md 4.1 / section 6 step 0a-2. Occupies ~0.45 of the single GPU's
# memory (~21GB on a 48GB card).
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"

DRAFT_MODEL="${DRAFT_MODEL:-ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ}"
PORT="${DRAFT_PORT:-8001}"
GPU_UTIL="${DRAFT_GPU_UTIL:-0.48}"
# Model's native max_seq_len (32768) needs ~8GiB of KV cache (32B model, more
# layers/heads than the judge), which doesn't fit alongside the judge
# model's share of the GPU; 8192 covers a long multi-turn draft conversation
# and needs only ~2GiB.
MAX_MODEL_LEN="${DRAFT_MAX_MODEL_LEN:-8192}"

source /hy-tmp/venv/bin/activate

# --enforce-eager skips torch.compile / CUDA-graph capture (very slow /
# unreliable to warm up on first run in this environment) — fine for the
# draft/free-generation workload here.
exec vllm serve "$DRAFT_MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --quantization awq \
    --served-model-name draft-model \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --download-dir "$HF_HOME"
