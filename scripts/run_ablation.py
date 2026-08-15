"""Ablations 1-4 (remaining-work plan, section three). All ablations run
inside Group C (interactive multi-turn) — only the mechanism under test
changes, per the design: adaptivity itself (1), the geometry fast-path (2),
the bandit scheduler (3), and the bandit's context granularity (4).

Usage:
    python scripts/run_ablation.py --data data/test.jsonl --mock \
        --presets full,fixed_path,no_geometry,no_bandit
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench.ablation import PRESETS, get_preset
from vdriftbench.bandit import ThompsonSamplingScheduler
from vdriftbench.config import Config
from vdriftbench.embedding_client import build_embedding_client
from vdriftbench.enrich import load_enriched_jsonl
from vdriftbench.io_utils import save_results_jsonl
from vdriftbench.llm_client import build_llm_client
from vdriftbench.pipeline import run_dataset, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablations 1-4 on Group C")
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument(
        "--presets",
        default="full,fixed_path,no_geometry,no_bandit,context_state_only,context_category_only",
        help=f"comma-separated subset of {sorted(PRESETS)}",
    )
    parser.add_argument("--config", default=None, help="YAML file overriding Config defaults")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="ablation_results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    config = config.apply_overrides(
        target_model=args.target_model, judge_model=args.judge_model, embedding_model=args.embedding_model
    )

    samples = load_enriched_jsonl(args.data)
    if args.limit:
        samples = samples[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preset_names = [p.strip() for p in args.presets.split(",") if p.strip()]
    summary_table = {}

    for preset_name in preset_names:
        ablation = get_preset(preset_name)

        target_client = build_llm_client(config.target_model, mock=args.mock, persona="target")
        judge_client = build_llm_client(config.judge_model, mock=args.mock, persona="judge")
        classifier_client = build_llm_client(config.judge_model, mock=args.mock, persona="classifier")
        embed_client = build_embedding_client(config.embedding_model, mock=args.mock)
        # Each ablation gets its own fresh scheduler: cross-sample learning
        # should not leak between "full" and "no_bandit" runs, for example.
        scheduler = ThompsonSamplingScheduler(seed=args.seed)

        results = run_dataset(
            samples, target_client, judge_client, classifier_client, embed_client, scheduler, config, ablation
        )

        out_path = out_dir / f"{preset_name}.jsonl"
        save_results_jsonl(results, str(out_path))
        scheduler.save(str(out_dir / f"{preset_name}_bandit_posterior.json"))

        resolved_by_counts = defaultdict(int)
        for r in results:
            for rd in r.rounds:
                if rd.resolved_by:
                    resolved_by_counts[rd.resolved_by] += 1

        summary = summarize(results, config)
        summary["resolved_by_distribution"] = dict(resolved_by_counts)
        summary_table[preset_name] = summary

        print(f"[{preset_name}] -> {out_path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, ensure_ascii=False, indent=2)
    print(f"\n消融对比汇总 -> {summary_path}")


if __name__ == "__main__":
    main()
