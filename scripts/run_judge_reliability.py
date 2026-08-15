"""Experiment 0: compare human annotations against the LLM judge.

Expects:
  --llm-results     a results.jsonl produced by main.py / run_pilot.py
  --human-annotations a jsonl file with one object per rated round:
      {"sample_id": "S0000", "round_idx": 3, "VDS": 1, "EFS": 0, "NJS": 1, "SCS": 1, "IFR": 1}

Usage:
    python scripts/run_judge_reliability.py --llm-results pilot_results.jsonl \
        --human-annotations data/human_annotations.jsonl
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench.io_utils import load_results_jsonl
from vdriftbench.metrics import disagreement_report, per_dimension_kappa

KAPPA_PASS_THRESHOLD = 0.6


def _build_llm_lookup(llm_rows: list[dict]) -> dict:
    lookup = {}
    for row in llm_rows:
        for rd in row["rounds"]:
            if rd["scores"]:
                lookup[(row["sample_id"], rd["round_idx"])] = rd["scores"]
    return lookup


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 0: judge reliability validation")
    parser.add_argument("--llm-results", default="pilot_results.jsonl")
    parser.add_argument("--human-annotations", required=True)
    parser.add_argument("--disagreement-threshold", type=int, default=1)
    args = parser.parse_args()

    llm_lookup = _build_llm_lookup(load_results_jsonl(args.llm_results))
    human_rows = _load_jsonl(args.human_annotations)

    human_scores, llm_scores, keys, missing = [], [], [], []
    for row in human_rows:
        key = (row["sample_id"], row["round_idx"])
        if key not in llm_lookup:
            missing.append(key)
            continue
        human_scores.append(row)
        llm_scores.append(llm_lookup[key])
        keys.append(key)

    if not human_scores:
        raise SystemExit("没有任何人工标注能匹配到对应的LLM裁判记录，检查sample_id/round_idx是否一致")

    kappas = per_dimension_kappa(human_scores, llm_scores)
    print(f"共匹配 {len(human_scores)} 条标注（{len(missing)} 条未匹配）")
    print("Cohen's Kappa（quadratic-weighted）各维度：")
    print(json.dumps(kappas, ensure_ascii=False, indent=2))

    for dim, kappa in kappas.items():
        idx = disagreement_report(human_scores, llm_scores, dim, threshold=args.disagreement_threshold)
        if idx:
            sample_keys = ", ".join(str(keys[i]) for i in idx[:10])
            print(f"\n{dim} 分歧样本（{len(idx)}条，前10条）: {sample_keys}")

    if missing:
        print(f"\n警告：{len(missing)} 条人工标注未在LLM结果中找到对应记录，例如: {missing[:5]}")

    avg_kappa = sum(kappas.values()) / len(kappas)
    verdict = "通过" if avg_kappa >= KAPPA_PASS_THRESHOLD else "未通过"
    print(f"\n平均Kappa = {avg_kappa:.3f} -> {verdict}（阈值 {KAPPA_PASS_THRESHOLD}）")
    if avg_kappa < KAPPA_PASS_THRESHOLD:
        print("建议：复核上面列出的分歧样本，修订裁判prompt的五维定义措辞，或引入双裁判+仲裁机制后重跑本脚本。")


if __name__ == "__main__":
    main()
