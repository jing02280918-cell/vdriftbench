"""Judge-reliability metrics (remaining-work plan, experiment 0).

Implements weighted Cohen's Kappa from scratch (no sklearn dependency) so
`scripts/run_judge_reliability.py` can compare human annotations against the
LLM judge's five-dimensional scores without pulling in a heavy ML library
just for one statistic.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def cohens_kappa(rater_a: Sequence[int], rater_b: Sequence[int], weights: str = "quadratic") -> float:
    """Weighted Cohen's Kappa for two raters scoring the same items on an
    ordinal 0/1/2 scale (matches the judge's five dimensions).

    `weights` is "quadratic" (penalize larger disagreements more), "linear",
    or "none" (unweighted / classic Kappa).
    """

    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length")
    if len(rater_a) == 0:
        raise ValueError("need at least one rated item")

    categories = sorted(set(rater_a) | set(rater_b))
    n_cat = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    n = len(rater_a)

    confusion = np.zeros((n_cat, n_cat), dtype=float)
    for a, b in zip(rater_a, rater_b):
        confusion[index[a], index[b]] += 1

    hist_a = confusion.sum(axis=1)
    hist_b = confusion.sum(axis=0)
    expected = np.outer(hist_a, hist_b) / n

    if weights == "none":
        w = 1 - np.eye(n_cat)
    else:
        idx = np.arange(n_cat)
        diff = np.abs(idx[:, None] - idx[None, :])
        w = diff**2 if weights == "quadratic" else diff

    observed_disagreement = (w * confusion).sum()
    expected_disagreement = (w * expected).sum()

    if expected_disagreement == 0:
        return 1.0
    return 1.0 - observed_disagreement / expected_disagreement


def per_dimension_kappa(
    human_scores: Sequence[dict], llm_scores: Sequence[dict], dims: Sequence[str] = ("VDS", "EFS", "NJS", "SCS", "IFR")
) -> dict:
    """`human_scores`/`llm_scores` are lists of {dim: 0|1|2} dicts, aligned by index."""

    result = {}
    for dim in dims:
        a = [h[dim] for h in human_scores]
        b = [l[dim] for l in llm_scores]
        result[dim] = cohens_kappa(a, b, weights="quadratic")
    return result


def disagreement_report(
    human_scores: Sequence[dict], llm_scores: Sequence[dict], dim: str, threshold: int = 1
) -> list[int]:
    """Indices where |human - llm| >= threshold for a given dimension, for the
    manual review step described in the remaining-work plan ("分歧样本复盘")."""

    return [
        i
        for i, (h, l) in enumerate(zip(human_scores, llm_scores))
        if abs(h[dim] - l[dim]) >= threshold
    ]


def label_agreement_rate(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Simple accuracy between two categorical label sequences (used to
    compare the state cascade's resolved states against LLM-only labels in
    ablation 2)."""

    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must have the same length")
    if not labels_a:
        return 1.0
    matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    return matches / len(labels_a)


def label_distribution(labels: Sequence[str]) -> dict:
    return dict(Counter(labels))
