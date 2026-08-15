"""Threshold calibration on the dev set (remaining-work plan, section one:
\u201c几何信号阈值 / Bandit奖励阈值需要用真实数据先标定\u201d).

These functions turn a batch of already-collected `RoundRecord`s (from a dev
run with `disable_geometry=True` and Layer2 LLM labels available for every
round) into concrete threshold suggestions, instead of hand-picking
tau_high/tau_low/theta from intuition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .schema import RoundRecord


@dataclass
class GeometryThresholdSuggestion:
    tau_high: float
    tau_low: float
    n_compliant_samples: int
    n_soft_resist_samples: int
    note: str


def suggest_geometry_thresholds(
    per_sample_records: Sequence[Sequence[RoundRecord]],
    high_percentile: float = 25.0,
    low_percentile: float = 25.0,
) -> GeometryThresholdSuggestion:
    """Given dev-set rounds that were LLM-labelled (Layer2, i.e. collected
    with `disable_geometry=True` so every round has a ground-truth state),
    compute round-over-round EmbedDrift_norm deltas and suggest tau_high /
    tau_low as the boundary that would recover most S_compliant /
    S_soft_resist labels via the free geometry signal alone.

    `per_sample_records` must be grouped **per sample** (one inner sequence
    per SampleResult.rounds) — deltas are only meaningful within a single
    sample's own round sequence, never across different samples.
    `high_percentile` / `low_percentile` control how conservative the
    fast-path is: a smaller percentile means the fast-path only fires on the
    most extreme, least ambiguous deltas.
    """

    compliant_deltas: list[float] = []
    resist_deltas: list[float] = []

    for records in per_sample_records:
        ordered = sorted(records, key=lambda r: r.round_idx)
        for prev, curr in zip(ordered, ordered[1:]):
            if prev.embed_drift_norm is None or curr.embed_drift_norm is None:
                continue
            delta = curr.embed_drift_norm - prev.embed_drift_norm
            if curr.state_resolved == "S_compliant":
                compliant_deltas.append(delta)
            elif curr.state_resolved == "S_soft_resist":
                resist_deltas.append(delta)

    tau_high = float(np.percentile(compliant_deltas, high_percentile)) if compliant_deltas else 0.15
    tau_low = float(np.percentile(resist_deltas, 100 - low_percentile)) if resist_deltas else -0.05

    return GeometryThresholdSuggestion(
        tau_high=tau_high,
        tau_low=tau_low,
        n_compliant_samples=len(compliant_deltas),
        n_soft_resist_samples=len(resist_deltas),
        note=(
            "基于dev集真实分布的建议值；样本量过小（<20）时应保留人工复核，"
            "不要直接采用统计结果覆盖经验阈值。"
        ),
    )


@dataclass
class RewardThetaSuggestion:
    theta: float
    median_round_over_round_delta: float
    note: str


def suggest_reward_theta(risk_score_sequences: Sequence[Sequence[float]], percentile: float = 60.0) -> RewardThetaSuggestion:
    """`risk_score_sequences` is one RiskScore-per-round list per dev sample.
    theta is set so that only round-over-round increases above the
    `percentile`-th percentile of the observed increase distribution count as
    a rewarded advance — avoids rewarding noise-level fluctuations."""

    deltas = []
    for seq in risk_score_sequences:
        for prev, curr in zip(seq, seq[1:]):
            deltas.append(curr - prev)
    positive_deltas = [d for d in deltas if d > 0]
    if not positive_deltas:
        return RewardThetaSuggestion(theta=1.0, median_round_over_round_delta=0.0, note="无正向增量样本，回退到默认值1.0")

    theta = float(np.percentile(positive_deltas, percentile))
    return RewardThetaSuggestion(
        theta=max(theta, 0.5),
        median_round_over_round_delta=float(np.median(positive_deltas)),
        note="建议值来自dev集正向RiskScore增量分布，仍需与实验0裁判可靠性结论一起复核。",
    )
