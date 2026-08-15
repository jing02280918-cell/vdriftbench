"""Experiment 2: the core Group A / B / C comparison across one or more
target models — the main experiment described in the remaining-work plan.

Group A: single-turn baseline (raw prompt, one shot)
Group B: batched instruction chain (fixed default path, one shot)
Group C: interactive multi-turn with the full v3.1 method

Usage:
    python scripts/run_main_experiment.py --data data/test.jsonl \
        --target-models deepseek-v4-flash --mock
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench.bandit import ThompsonSamplingScheduler
from vdriftbench.config import Config
from vdriftbench.embedding_client import build_embedding_client
from vdriftbench.enrich import load_enriched_jsonl
from vdriftbench.groups import run_group_a, run_group_b
from vdriftbench.io_utils import save_results_jsonl
from vdriftbench.llm_client import build_llm_client
from vdriftbench.pipeline import run_dataset, summarize

GROUPS = ("A", "B", "C")


def run_for_model(samples, model_name: str, config: Config, args, group: str) -> list:
    target_client = build_llm_client(model_name, mock=args.mock, persona="target")
    judge_client = build_llm_client(config.judge_model, mock=args.mock, persona="judge")
    embed_client = build_embedding_client(config.embedding_model, mock=args.mock)

    if group == "A":
        return [run_group_a(s, target_client, judge_client, embed_client, config) for s in samples]
    if group == "B":
        return [run_group_b(s, target_client, judge_client, embed_client, config) for s in samples]

    classifier_client = build_llm_client(config.judge_model, mock=args.mock, persona="classifier")
    bandit_state_path = Path(args.out_dir) / f"bandit_posterior_{model_name.replace('/', '_')}.json"
    scheduler = ThompsonSamplingScheduler.load_or_create(str(bandit_state_path), seed=args.seed)
    results = run_dataset(samples, target_client, judge_client, classifier_client, embed_client, scheduler, config)
    scheduler.save(str(bandit_state_path))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: Group A/B/C main experiment")
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--config", default=None, help="YAML file overriding Config defaults")
    parser.add_argument("--target-models", required=True, help="comma-separated list of target model names")
    parser.add_argument("--groups", default="A,B,C", help="comma-separated subset of A,B,C")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="experiment2_results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    config = config.apply_overrides(judge_model=args.judge_model, embedding_model=args.embedding_model)

    samples = load_enriched_jsonl(args.data)
    if args.limit:
        samples = samples[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in args.target_models.split(",") if m.strip()]
    groups = [g.strip().upper() for g in args.groups.split(",") if g.strip()]

    summary_table = {}
    for model_name in model_names:
        summary_table[model_name] = {}
        for group in groups:
            if group not in GROUPS:
                raise SystemExit(f"unknown group {group!r}, must be one of {GROUPS}")
            results = run_for_model(samples, model_name, config, args, group)
            safe_model = model_name.replace("/", "_")
            out_path = out_dir / f"group{group}_{safe_model}.jsonl"
            save_results_jsonl(results, str(out_path))
            summary_table[model_name][group] = summarize(results, config)
            print(f"[{model_name} / Group {group}] -> {out_path}")
            print(json.dumps(summary_table[model_name][group], ensure_ascii=False, indent=2))

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, ensure_ascii=False, indent=2)
    print(f"\n汇总表 -> {summary_path}")


if __name__ == "__main__":
    main()
