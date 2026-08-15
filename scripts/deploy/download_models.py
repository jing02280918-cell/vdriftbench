#!/usr/bin/env python3
"""Download the local model weights needed for HANDOFF.md 4.1 (draft / judge /
embedding roles) from the HF-Mirror endpoint into a cache directory on the
large data disk (default: /hy-tmp/hf_cache).

Usage:
    python download_models.py --all
    python download_models.py --only embedding,judge
    python download_models.py --only draft

Notes
-----
- HANDOFF.md 4.1 specifies the draft model as
  ``huihui-ai/Qwen2.5-32B-Instruct-abliterated`` "4bit AWQ/GPTQ, ~21GB", but
  that HF repo itself only hosts the full bf16 checkpoint (~65GB, too big for
  the 48GB card alongside the judge model). The default draft repo here is
  instead ``ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ``, the
  most widely-used (15k+ downloads) community AWQ quantization of that same
  huihui-ai abliterated checkpoint (~19.3GB, matches the ~21GB budget).
- meta-llama/Llama-3.1-8B-Instruct (the judge model originally selected in
  HANDOFF.md 4.1) is a *gated* HF repo: downloading its weight files requires
  an authenticated HF token that has been granted access. No such token was
  available at deployment time, so the default judge model here is
  ``mistralai/Mistral-7B-Instruct-v0.3`` instead — ungated, similar size
  (~7B, bf16 ~15GB), a different model family from both the draft model
  (Qwen-based) and the common remote target models (GPT/DeepSeek/Qwen), which
  preserves the "judge != target family" constraint from HANDOFF.md section 4.
  If you later obtain an approved HF token, re-run with
  ``--judge-repo meta-llama/Llama-3.1-8B-Instruct`` and ``HF_TOKEN=...`` set.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# The Xet fast-transfer backend talks to cas-server.xethub.hf.co directly,
# which is not proxied by the HF mirror and 401s from this network; fall back
# to plain HTTP downloads through the mirror instead.
os.environ["HF_HUB_DISABLE_XET"] = "1"

DEFAULT_CACHE_DIR = "/hy-tmp/hf_cache"

MODELS = {
    "embedding": "BAAI/bge-m3",
    "draft": "ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ",
    "judge": "mistralai/Mistral-7B-Instruct-v0.3",
}


# Skip repo assets we don't need (images/docs/alternate export formats) and
# that occasionally 403 through the HF mirror (e.g. dotfiles under imgs/).
IGNORE_PATTERNS = [
    "*.md", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.DS_Store",
    "imgs/*", "images/*", "*.onnx", "onnx/*", "*.msgpack", "*.h5",
    "*.ot", "*.tflite", "coreml/*", "openvino/*",
    # some repos ship a duplicate single-file "consolidated" checkpoint
    # (for e.g. the mistral-inference client) alongside the sharded
    # safetensors that transformers/vLLM actually load — skip it, it just
    # doubles the download/disk footprint.
    "consolidated*", "*.pth",
]


def download(repo_id: str, cache_dir: str, token: str | None, max_attempts: int = 15) -> None:
    from huggingface_hub import snapshot_download

    print(f"\n=== downloading {repo_id} -> {cache_dir} ===", flush=True)
    # The HF mirror occasionally returns transient 502/503/timeout errors on
    # individual files; already-downloaded blobs are content-addressed and
    # skipped on retry, so simply re-invoking snapshot_download resumes.
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                token=token,
                max_workers=4,
                ignore_patterns=IGNORE_PATTERNS,
            )
            print(f"=== done: {repo_id} -> {path} ===", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait_s = min(5 * attempt, 60)
            print(f"[retry {attempt}/{max_attempts}] {repo_id} failed: {exc}\n  -> retrying in {wait_s}s", flush=True)
            time.sleep(wait_s)
    raise RuntimeError(f"giving up on {repo_id} after {max_attempts} attempts") from last_err


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--only", default=None, help="comma-separated subset of: embedding,draft,judge")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--draft-repo", default=MODELS["draft"])
    parser.add_argument("--judge-repo", default=MODELS["judge"])
    parser.add_argument("--embedding-repo", default=MODELS["embedding"])
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    repos = {
        "embedding": args.embedding_repo,
        "draft": args.draft_repo,
        "judge": args.judge_repo,
    }

    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
    elif args.all or True:
        wanted = list(repos.keys())

    os.makedirs(args.cache_dir, exist_ok=True)
    for key in wanted:
        if key not in repos:
            print(f"unknown model key: {key}", file=sys.stderr)
            sys.exit(1)
        download(repos[key], args.cache_dir, args.hf_token)

    print("\nAll requested models downloaded.")
    print(f"HF cache dir: {args.cache_dir}")
    print("Set HF_HOME to this directory (or export HF_HUB_CACHE) so vLLM / sentence-transformers reuse it:")
    print(f'  export HF_HOME="{args.cache_dir}"')


if __name__ == "__main__":
    main()
