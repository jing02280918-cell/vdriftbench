"""Experiment 1: pilot run of the full APS-v2 method on a small, evenly
sampled subset (default ~5 samples per macro category), against a single
target model. Verifies the state cascade fires sensibly and the pipeline is
healthy before committing budget to the main experiment.

Usage:
    python scripts/run_pilot.py --data data/dev.jsonl --mock
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench.bandit import ThompsonSamplingScheduler
from vdriftbench.config import Config
from vdriftbench.embedding_client import build_embedding_client
from vdriftbench.enrich import load_enriched_jsonl
from vdriftbench.io_utils import save_results_jsonl
from vdriftbench.llm_client import build_llm_client
from vdriftbench.pipeline import run_dataset, summarize


def sample_pilot_set(samples, per_category: int, seed: int):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for s in samples:
        buckets[s.category_macro].append(s)
    pilot = []
    for bucket in buckets.values():
        shuffled = bucket[:]
        rng.shuffle(shuffled)
        pilot.extend(shuffled[:per_category])
    return pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: pilot run")
    parser.add_argument("--data", default="data/dev.jsonl")
    parser.add_argument("--config", default=None, help="YAML file overriding Config defaults")
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--out", default="pilot_results.jsonl")
    parser.add_argument("--bandit-state", default="pilot_bandit_posterior.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    config = config.apply_overrides(
        target_model=args.target_model, judge_model=args.judge_model, embedding_model=args.embedding_model
    )

    samples = load_enriched_jsonl(args.data)
    pilot_samples = sample_pilot_set(samples, per_category=args.per_category, seed=args.seed)

    target_client = build_llm_client(config.target_model, mock=args.mock, persona="target")
    judge_client = build_llm_client(config.judge_model, mock=args.mock, persona="judge")
    classifier_client = build_llm_client(config.judge_model, mock=args.mock, persona="classifier")
    embed_client = build_embedding_client(config.embedding_model, mock=args.mock)
    scheduler = ThompsonSamplingScheduler.load_or_create(args.bandit_state, seed=args.seed)

    results = run_dataset(
        pilot_samples, target_client, judge_client, classifier_client, embed_client, scheduler, config
    )

    save_results_jsonl(results, args.out)
    scheduler.save(args.bandit_state)

    resolved_by_counts = defaultdict(int)
    state_counts = defaultdict(int)
    for r in results:
        for rd in r.rounds:
            if rd.resolved_by:
                resolved_by_counts[rd.resolved_by] += 1
            if rd.state_resolved:
                state_counts[rd.state_resolved] += 1

    print(f"pilot集大小: {len(pilot_samples)}（每大类约{args.per_category}条）")
    print("总体指标:", json.dumps(summarize(results, config), ensure_ascii=False, indent=2))
    print("状态解析层级分布 (layer0/layer1/layer2):", dict(resolved_by_counts))
    print("状态分类分布:", dict(state_counts))
    print(f"逐轮记录 -> {args.out}")


if __name__ == "__main__":
    main()
