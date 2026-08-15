"""Calibrate tau_high / tau_low (geometry fast-path) and theta (bandit reward
threshold) from a dev-set run, instead of hand-picking them.

Runs the dev set once with the `no_geometry` ablation preset so every round
is forced through the Layer 2 LLM classifier and gets a ground-truth
`state_resolved` label to calibrate the geometry fast-path against.

Usage (offline smoke test):
    python scripts/calibrate_thresholds.py --dev-data data/dev.jsonl --mock

Usage (real models):
    python scripts/calibrate_thresholds.py --dev-data data/dev.jsonl \
        --target-model deepseek-v4-flash --judge-model judge-model
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench.ablation import get_preset
from vdriftbench.bandit import ThompsonSamplingScheduler
from vdriftbench.calibration import suggest_geometry_thresholds, suggest_reward_theta
from vdriftbench.config import Config
from vdriftbench.embedding_client import build_embedding_client
from vdriftbench.enrich import load_enriched_jsonl
from vdriftbench.llm_client import build_llm_client
from vdriftbench.pipeline import run_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate tau_high/tau_low/theta on the dev set")
    parser.add_argument("--dev-data", default="data/dev.jsonl")
    parser.add_argument("--config", default=None, help="YAML file overriding Config defaults")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    config = config.apply_overrides(
        target_model=args.target_model, judge_model=args.judge_model, embedding_model=args.embedding_model
    )

    samples = load_enriched_jsonl(args.dev_data)
    if args.limit:
        samples = samples[: args.limit]

    target_client = build_llm_client(config.target_model, mock=args.mock, persona="target")
    judge_client = build_llm_client(config.judge_model, mock=args.mock, persona="judge")
    classifier_client = build_llm_client(config.judge_model, mock=args.mock, persona="classifier")
    embed_client = build_embedding_client(config.embedding_model, mock=args.mock)
    scheduler = ThompsonSamplingScheduler()

    ablation = get_preset("no_geometry")
    results = run_dataset(
        samples, target_client, judge_client, classifier_client, embed_client, scheduler, config, ablation
    )

    per_sample_rounds = [r.rounds for r in results]
    geo = suggest_geometry_thresholds(per_sample_rounds)

    risk_sequences = [[rr.scores.risk_score for rr in r.rounds if rr.scores] for r in results]
    theta = suggest_reward_theta(risk_sequences)

    print("几何阈值建议 (写入 config.py 的 tau_high / tau_low):")
    print(json.dumps(geo.__dict__, ensure_ascii=False, indent=2))
    print("\nBandit奖励阈值建议 (写入 config.py 的 reward_theta):")
    print(json.dumps(theta.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
