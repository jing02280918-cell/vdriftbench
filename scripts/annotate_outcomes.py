"""Label every sample in a dataset with its experiment outcome (成功 / 部分成功
/ 失败 / 未运行) based on a `main.py`-produced results.jsonl, and split out the
unsuccessful ones into their own file for review.

"未成功" here uses the strict definition: only "成功" counts as success (i.e.
`SampleResult.is_success()`), so both "部分成功" and "失败" samples are marked
unsuccessful — matching risk_success_threshold/risk_partial_threshold in
Config (override via --config if you calibrated your own thresholds).

Safe to run against a still-in-progress (or --resume'd) results.jsonl: rows
not yet present are labeled "未运行" and excluded from the unsuccessful file
rather than being counted as failures.

Usage:
    python scripts/annotate_outcomes.py \
        --data data/dataset_100.enriched.jsonl --results results.jsonl \
        --annotated-out data/dataset_100.annotated.jsonl \
        --unsuccessful-out data/dataset_100.unsuccessful.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench.config import Config
from vdriftbench.enrich import load_enriched_jsonl
from vdriftbench.io_utils import load_sample_results_jsonl
from vdriftbench.schema import SampleResult

NOT_RUN = "未运行"


def outcome_for_sample(result: SampleResult, success_threshold: float, partial_threshold: float) -> str:
    """The best outcome achieved by any build-up round, matching the same
    per-round `JudgeScores.outcome()` rule `SampleResult.is_success()` uses
    for its all-or-nothing check — just surfaced as three tiers instead of a
    bool."""

    best = "失败"
    for r in result.rounds:
        if r.scores is None:
            continue
        o = r.scores.outcome(success_threshold, partial_threshold)
        if o == "成功":
            return "成功"
        if o == "部分成功":
            best = "部分成功"
    return best


def annotate(samples: list, results_by_id: dict, success_threshold: float, partial_threshold: float) -> list[dict]:
    annotated = []
    for sample in samples:
        row = asdict(sample)
        result = results_by_id.get(sample.sample_id)
        if result is None:
            row.update(experiment_outcome=NOT_RUN, is_success=None, max_risk_score=None,
                       first_success_round=None, terminated_early=None)
        else:
            outcome = outcome_for_sample(result, success_threshold, partial_threshold)
            risk_scores = [r.scores.risk_score for r in result.rounds if r.scores is not None]
            row.update(
                experiment_outcome=outcome,
                is_success=(outcome == "成功"),
                max_risk_score=max(risk_scores) if risk_scores else None,
                first_success_round=result.first_drift_round(success_threshold),
                terminated_early=result.terminated_early,
            )
        annotated.append(row)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Label dataset rows with their experiment outcome")
    parser.add_argument("--data", required=True, help="enriched dataset jsonl (the one --data was run on)")
    parser.add_argument("--results", required=True, help="results.jsonl produced by main.py")
    parser.add_argument("--config", default=None, help="YAML overriding Config defaults (for calibrated thresholds)")
    parser.add_argument("--annotated-out", default=None, help="default: <data> with .annotated.jsonl suffix")
    parser.add_argument("--unsuccessful-out", default=None, help="default: <data> with .unsuccessful.jsonl suffix")
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()

    samples = load_enriched_jsonl(args.data)
    results = load_sample_results_jsonl(args.results)
    results_by_id = {r.sample.sample_id: r for r in results}

    annotated = annotate(samples, results_by_id, config.risk_success_threshold, config.risk_partial_threshold)

    annotated_out = args.annotated_out or args.data.rsplit(".jsonl", 1)[0] + ".annotated.jsonl"
    unsuccessful_out = args.unsuccessful_out or args.data.rsplit(".jsonl", 1)[0] + ".unsuccessful.jsonl"

    with open(annotated_out, "w", encoding="utf-8") as f:
        for row in annotated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    unsuccessful = [row for row in annotated if row["is_success"] is False]
    with open(unsuccessful_out, "w", encoding="utf-8") as f:
        for row in unsuccessful:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in annotated:
        counts[row["experiment_outcome"]] = counts.get(row["experiment_outcome"], 0) + 1

    print(json.dumps({
        "n_total": len(annotated),
        "by_outcome": counts,
        "n_unsuccessful (部分成功+失败, 不含未运行)": len(unsuccessful),
        "success_threshold": config.risk_success_threshold,
        "partial_threshold": config.risk_partial_threshold,
    }, ensure_ascii=False, indent=2))
    print(f"annotated dataset -> {annotated_out}")
    print(f"unsuccessful subset -> {unsuccessful_out}")


if __name__ == "__main__":
    main()
