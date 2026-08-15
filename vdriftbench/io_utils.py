"""Shared (de)serialization helpers for `SampleResult`, used by `main.py` and
every script under `scripts/` so the on-disk record format stays identical
across the single-run CLI, the pilot experiment, the main experiment, and
every ablation run — this is what makes their outputs directly comparable.
"""

from __future__ import annotations

import json
from typing import Iterable

from .schema import JudgeScores, RoundRecord, RWIScores, Sample, SampleResult


def round_to_dict(r: RoundRecord) -> dict:
    return {
        "round_idx": r.round_idx,
        "state_resolved": r.state_resolved,
        "resolved_by": r.resolved_by,
        "principle_used": r.principle_used,
        "bandit_context": r.bandit_context,
        "bandit_posterior_snapshot": r.bandit_posterior_snapshot,
        "prompt": r.prompt,
        "response": r.response,
        "scores": r.scores.to_dict() if r.scores else None,
        "embed_drift_norm": r.embed_drift_norm,
        "reward_applied_to_previous": r.reward_applied_to_previous,
        # --- v3 4.3/4.6节 ---
        "draft_observation": r.draft_observation,
        "draft_thought": r.draft_thought,
        "fidelity_label": r.fidelity_label,
        # --- v3 5.2/5.3节 ---
        "r_token": r.r_token,
        "trajectory": r.trajectory,
        # --- v3 6.5节 ---
        "resist_archetype_raw": r.resist_archetype_raw,
        "resist_archetype_id": r.resist_archetype_id,
        "resist_archetype_name": r.resist_archetype_name,
    }


def sample_result_to_dict(result: SampleResult, extra: dict | None = None) -> dict:
    payload = {
        "sample_id": result.sample.sample_id,
        "category": result.sample.category,
        "category_macro": result.sample.category_macro,
        "target_claim": result.sample.target_claim,
        "terminated_early": result.terminated_early,
        "terminated_at_round": result.terminated_at_round,
        "rounds": [round_to_dict(r) for r in result.rounds],
        "recovery_prompt": result.recovery_prompt,
        "recovery_response": result.recovery_response,
        "recovery_scs": result.recovery_scs,
        # --- v3 9.2节 ---
        "rwi_scores": result.rwi_scores.to_dict() if result.rwi_scores else None,
        "rwi_reviewed_round_idx": result.rwi_reviewed_round_idx,
    }
    if extra:
        payload.update(extra)
    return payload


def save_results_jsonl(results: Iterable[SampleResult], path: str, extra: dict | None = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(sample_result_to_dict(r, extra=extra), ensure_ascii=False) + "\n")


def load_results_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _judge_scores_from_dict(d: dict | None) -> JudgeScores | None:
    if d is None:
        return None
    return JudgeScores(VDS=d["VDS"], EFS=d["EFS"], NJS=d["NJS"], SCS=d["SCS"], IFR=d["IFR"])


def _rwi_scores_from_dict(d: dict | None) -> RWIScores | None:
    if d is None:
        return None
    return RWIScores(BEL=d["BEL"], PER=d["PER"], SHA=d["SHA"])


def _round_from_dict(d: dict) -> RoundRecord:
    return RoundRecord(
        round_idx=d["round_idx"],
        state_resolved=d.get("state_resolved"),
        resolved_by=d.get("resolved_by"),
        principle_used=d.get("principle_used", ""),
        bandit_context=d.get("bandit_context"),
        bandit_posterior_snapshot=d.get("bandit_posterior_snapshot"),
        prompt=d.get("prompt", ""),
        response=d.get("response", ""),
        scores=_judge_scores_from_dict(d.get("scores")),
        embed_drift_norm=d.get("embed_drift_norm"),
        reward_applied_to_previous=d.get("reward_applied_to_previous"),
        draft_observation=d.get("draft_observation"),
        draft_thought=d.get("draft_thought"),
        fidelity_label=d.get("fidelity_label"),
        r_token=d.get("r_token"),
        trajectory=d.get("trajectory"),
        resist_archetype_raw=d.get("resist_archetype_raw"),
        resist_archetype_id=d.get("resist_archetype_id"),
        resist_archetype_name=d.get("resist_archetype_name"),
    )


def sample_result_from_dict(d: dict) -> SampleResult:
    """Inverse of `sample_result_to_dict`. `sample_result_to_dict` only keeps a
    few `Sample` fields (sample_id/category/category_macro/target_claim) for
    the on-disk record — the rest (`prompt`/`value_axis`/`risk_type`/
    `expected_safe_behavior`) are not round-trippable from a results file
    alone and are left blank here. Good enough for anything that only needs
    per-round scores / outcome (summarize(), annotate_outcomes.py); if you
    need the full original sample, join back to the enriched dataset by
    `sample_id`."""

    sample = Sample(
        sample_id=d["sample_id"],
        prompt="",
        category=d.get("category", ""),
        category_macro=d.get("category_macro", ""),
        target_claim=d.get("target_claim", ""),
        value_axis="",
        risk_type="",
        expected_safe_behavior="",
    )
    return SampleResult(
        sample=sample,
        rounds=[_round_from_dict(r) for r in d.get("rounds", [])],
        terminated_early=d.get("terminated_early", False),
        terminated_at_round=d.get("terminated_at_round"),
        recovery_prompt=d.get("recovery_prompt"),
        recovery_response=d.get("recovery_response"),
        recovery_scs=d.get("recovery_scs"),
        rwi_scores=_rwi_scores_from_dict(d.get("rwi_scores")),
        rwi_reviewed_round_idx=d.get("rwi_reviewed_round_idx"),
    )


def load_sample_results_jsonl(path: str) -> list[SampleResult]:
    return [sample_result_from_dict(d) for d in load_results_jsonl(path)]
