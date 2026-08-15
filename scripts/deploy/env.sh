#!/usr/bin/env bash
# Source this file to set up the vdriftbench runtime environment:
#   source scripts/deploy/env.sh
#
# It activates the venv (installed on the large data disk /hy-tmp, not the
# small root disk) and sets the HF mirror / cache locations plus the
# OpenAI-compatible endpoint variables consumed by vdriftbench/llm_client.py.

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
# NOTE: HF_HUB_ENABLE_HF_TRANSFER is deprecated/unused by the installed
# huggingface_hub version and the optional hf_transfer package was never
# installed in the venv — leaving this =1 makes some code paths (e.g. vLLM's
# internal weight downloader) crash with `ModuleNotFoundError: No module
# named 'hf_transfer'`. Do not set it.

source /hy-tmp/venv/bin/activate

# --- remote target model (real, non-mock runs) ---
# Prefer a `.env` file instead of exporting secrets in this shared script:
# `main.py` auto-loads `./.env` (or repo-root `.env`) via
# vdriftbench/env_loader.py, without needing python-dotenv installed. e.g.
# create /root/vdriftbench/.env with:
#   DEEPSEEK_API_KEY=sk-xxxx
# and --target-model deepseek-v4-flash will pick it up automatically (base
# url defaults to https://api.deepseek.com for any "deepseek*" model name).
# The two lines below are the manual/global equivalent if you'd rather export
# them by hand (works for any OpenAI-compatible endpoint, not just DeepSeek):
# export VDB_API_KEY="sk-xxxx"
# export VDB_BASE_URL="https://api.deepseek.com"   # or the target model's endpoint

# --- per-role overrides (llm_client.build_llm_client) ---
# Each persona (target/judge/classifier/enrich) falls back to VDB_API_KEY /
# VDB_BASE_URL above when its own VDB_<PERSONA>_* pair is unset, so a single
# global endpoint (the v2 behavior) still works if you only set the two
# variables above. To run entirely on the local vLLM servers instead
# (see serve_draft.sh / serve_judge.sh), e.g. for an offline smoke test:
# export VDB_TARGET_API_KEY="EMPTY"
# export VDB_TARGET_BASE_URL="http://127.0.0.1:8001/v1"   # draft-model
# export VDB_JUDGE_API_KEY="EMPTY"
# export VDB_JUDGE_BASE_URL="http://127.0.0.1:8002/v1"    # judge-model
# (classifier defaults to the judge endpoint automatically if unset)

echo "vdriftbench venv activated: $(which python)"
echo "HF cache dir: $HF_HOME (via $HF_ENDPOINT)"
