"""Stratified dev/test split by category_macro (remaining-work plan, section
one: "dev / test 分层切分").

Stratifying by the macro category (rather than a plain random split) keeps
every one of the 8 macro categories represented in both dev and test, which
matters because the bandit's context is keyed on category_macro — an
under-represented category in test would make its VSR/RiskScore numbers
unreliable regardless of how good the method is.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

from .schema import Sample


def stratified_split(
    samples: Sequence[Sample], dev_ratio: float = 0.7, seed: int = 0
) -> tuple[list[Sample], list[Sample]]:
    """Return (dev, test), each internally shuffled, with `dev_ratio` of each
    category_macro bucket going to dev and the remainder to test."""

    buckets: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        buckets[s.category_macro].append(s)

    rng = random.Random(seed)
    dev: list[Sample] = []
    test: list[Sample] = []
    for macro, bucket in buckets.items():
        shuffled = bucket[:]
        rng.shuffle(shuffled)
        n_dev = round(len(shuffled) * dev_ratio)
        dev.extend(shuffled[:n_dev])
        test.extend(shuffled[n_dev:])

    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def split_summary(dev: Sequence[Sample], test: Sequence[Sample]) -> dict:
    def counts(samples: Sequence[Sample]) -> dict:
        out: dict[str, int] = defaultdict(int)
        for s in samples:
            out[s.category_macro] += 1
        return dict(out)

    return {"n_dev": len(dev), "n_test": len(test), "dev_by_category": counts(dev), "test_by_category": counts(test)}
