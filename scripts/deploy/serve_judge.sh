#!/usr/bin/env bash
# Starts the local judge (5-dim scoring / Layer2 classifier / 9.2 RWI review)
# model with vLLM, per HANDOFF.md 4.1 / section 6 step 0a-2. Occupies ~0.35 of
# the single GPU's memory (~16GB on a 48GB card).
#
# NOTE: HANDOFF.md originally selected meta-llama/Llama-3.1-8B-Instruct for
# this role, but that repo is HF-gated and no approved token was available at
# deployment time. This script defaults to mistralai/Mistral-7B-Instruct-v0.3
# instead (ungated, similar size, different model family from the Qwen-based
# draft model and from common remote target models). See
# scripts/deploy/download_models.py docstring for details, or override
# JUDGE_MODEL below once you have an HF_TOKEN with Llama-3.1 access.
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"

JUDGE_MODEL="${JUDGE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
PORT="${JUDGE_PORT:-8002}"
GPU_UTIL="${JUDGE_GPU_UTIL:-0.40}"
# Model's native max_seq_len (32768) needs ~4GiB of KV cache, which doesn't
# fit alongside the draft model's share of the GPU; 16384 is plenty for
# scoring a multi-turn conversation transcript and needs only ~2GiB.
MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"

source /hy-tmp/venv/bin/activate

# --enforce-eager skips torch.compile / CUDA-graph capture (which was very
# slow / unreliable to warm up on first run in this environment) — fine for
# judge scoring workloads where per-request latency isn't the bottleneck.
exec vllm serve "$JUDGE_MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --served-model-name judge-model \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --download-dir "$HF_HOME"
