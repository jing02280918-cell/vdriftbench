"""Single-shot A/B validation: does P4_value lift NJS vs P4_rhetoric on the
same frozen S_engaged context?

Build-up path (matches v3.9 single-arm S_engaged behaviour):
    round1 = P1
    round2 = P4_rhetoric
    round3 = P4_rhetoric
then FORK round4 on the identical frozen history [r1,r2,r3]:
    arm A = P4_rhetoric   (control — what single-arm bandit would keep doing)
    arm B = P4_value      (treatment — what NJS escalation forces)

Both arms get the same dimension feedback hint (derived from round-3 scores)
and the same accumulated [O] hypothesis; only the principle token differs.
We then judge both arms and compare NJS (and full RiskScore).

This uses the REAL endpoints (draft@8001, judge@8002, target=deepseek), so it
is a genuine — but cheap (N samples, no embedding/bandit) — signal on whether
the escalation is worth running at full scale.

Usage:
    /hy-tmp/venv/bin/python scripts/validate_p4_value.py --n 6
"""

from __future__ import annotations

import argparse
import json

from vdriftbench.env_loader import load_dotenv_if_present
from vdriftbench.llm_client import build_llm_client
from vdriftbench.principles import PRINCIPLES, _build_score_feedback, draft_utterance
from vdriftbench.judge import judge_response
from vdriftbench.enrich import load_enriched_jsonl
from vdriftbench.schema import RoundRecord


def _run_round(principle, sample, history, round_idx, draft_client, target_client,
               judge_client, prev_observation, dimension_hint):
    draft = draft_utterance(
        principle=principle,
        sample=sample,
        history=history,
        round_idx=round_idx,
        draft_client=draft_client,
        prev_observation=prev_observation,
        dimension_hint=dimension_hint,
        category_macro=sample.category_macro,
    )
    prompt_text = draft.response
    messages = []
    for r in history:
        messages.append({"role": "user", "content": r.prompt})
        messages.append({"role": "assistant", "content": r.response})
    messages.append({"role": "user", "content": prompt_text})
    response_text = target_client.chat(messages)
    history_text = "\n".join(
        f"[第{r.round_idx}轮-提问] {r.prompt}\n[第{r.round_idx}轮-回复] {r.response}"
        for r in history
    )
    scores = judge_response(
        target_claim=sample.target_claim,
        expected_safe_behavior=sample.expected_safe_behavior,
        history_text=history_text,
        latest_response=response_text,
        judge_client=judge_client,
    )
    record = RoundRecord(
        round_idx=round_idx,
        state_resolved=None,
        resolved_by=None,
        principle_used=principle,
        bandit_context=None,
        bandit_posterior_snapshot=None,
        prompt=prompt_text,
        response=response_text,
        scores=scores,
        embed_drift_norm=None,
        draft_observation=draft.observation,
        draft_thought=draft.thought,
        fidelity_label=draft.fidelity_label,
    )
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--data", default="data/dataset_100.enriched.jsonl")
    args = ap.parse_args()

    load_dotenv_if_present()

    draft_client = build_llm_client("draft-model", mock=False, persona="draft")
    target_client = build_llm_client("deepseek-v4-flash", mock=False, persona="target")
    judge_client = build_llm_client("judge-model", mock=False, persona="judge")

    samples = load_enriched_jsonl(args.data)[: args.n]

    rows = []
    for sample in samples:
        history = []

        # round 1: P1
        r1 = _run_round("P1", sample, history, 1, draft_client, target_client,
                        judge_client, None, None)
        history.append(r1)

        # round 2: P4_rhetoric (single-arm engaged path)
        r2 = _run_round("P4_rhetoric", sample, history, 2, draft_client, target_client,
                        judge_client, r1.draft_observation, _build_score_feedback(r1.scores))
        history.append(r2)

        # round 3: P4_rhetoric (build-up continues while NJS <= 1)
        r3 = _run_round("P4_rhetoric", sample, history, 3, draft_client, target_client,
                        judge_client, r2.draft_observation, _build_score_feedback(r2.scores))
        history.append(r3)

        # FORK on the identical frozen history [r1, r2, r3]
        hint = _build_score_feedback(r3.scores)
        obs = r3.draft_observation

        arm_a = _run_round("P4_rhetoric", sample, history, 4, draft_client, target_client,
                           judge_client, obs, hint)
        arm_b = _run_round("P4_value", sample, history, 4, draft_client, target_client,
                           judge_client, obs, hint)

        rows.append({
            "sample_id": sample.sample_id,
            "r3": {
                "VDS": r3.scores.VDS, "EFS": r3.scores.EFS, "NJS": r3.scores.NJS,
                "SCS": r3.scores.SCS, "IFR": r3.scores.IFR, "RS": r3.scores.risk_score,
            },
            "arm_A_P4_rhetoric": {
                "VDS": arm_a.scores.VDS, "EFS": arm_a.scores.EFS, "NJS": arm_a.scores.NJS,
                "SCS": arm_a.scores.SCS, "IFR": arm_a.scores.IFR, "RS": arm_a.scores.risk_score,
            },
            "arm_B_P4_value": {
                "VDS": arm_b.scores.VDS, "EFS": arm_b.scores.EFS, "NJS": arm_b.scores.NJS,
                "SCS": arm_b.scores.SCS, "IFR": arm_b.scores.IFR, "RS": arm_b.scores.risk_score,
            },
        })
        print(f"[{sample.sample_id}] r3 RS={r3.scores.risk_score} NJS={r3.scores.NJS} | "
              f"A(P4_rhetoric) NJS={arm_a.scores.NJS} RS={arm_a.scores.risk_score} | "
              f"B(P4_value) NJS={arm_b.scores.NJS} RS={arm_b.scores.risk_score}", flush=True)

    print("\n===== 汇总 =====")
    njs_win_b = 0
    njs_tie = 0
    njs_win_a = 0
    for row in rows:
        a = row["arm_A_P4_rhetoric"]["NJS"]
        b = row["arm_B_P4_value"]["NJS"]
        if b > a:
            njs_win_b += 1
        elif b == a:
            njs_tie += 1
        else:
            njs_win_a += 1
    print(f"NJS: B(P4_value) 更高 {njs_win_b} 次 / 持平 {njs_tie} 次 / A 更高 {njs_win_a} 次 (n={len(rows)})")

    avg_a_njs = sum(r["arm_A_P4_rhetoric"]["NJS"] for r in rows) / len(rows)
    avg_b_njs = sum(r["arm_B_P4_value"]["NJS"] for r in rows) / len(rows)
    avg_a_rs = sum(r["arm_A_P4_rhetoric"]["RS"] for r in rows) / len(rows)
    avg_b_rs = sum(r["arm_B_P4_value"]["RS"] for r in rows) / len(rows)
    print(f"平均 NJS: A={avg_a_njs:.2f}  B={avg_b_njs:.2f}")
    print(f"平均 RS:  A={avg_a_rs:.2f}  B={avg_b_rs:.2f}")

    out = "data/p4_value_ab_validation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"详细结果 -> {out}")


if __name__ == "__main__":
    main()
