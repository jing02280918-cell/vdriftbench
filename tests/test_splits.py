from vdriftbench.schema import Sample
from vdriftbench.splits import split_summary, stratified_split


def _make_samples(macro: str, n: int, prefix: str) -> list[Sample]:
    return [
        Sample(
            sample_id=f"{prefix}{i}",
            prompt=f"prompt {i}",
            category=macro,
            category_macro=macro,
            target_claim="claim",
            value_axis="axis",
            risk_type="risk",
            expected_safe_behavior="safe",
        )
        for i in range(n)
    ]


def test_stratified_split_keeps_every_macro_category_in_both_splits():
    samples = _make_samples("历史类", 10, "H") + _make_samples("经济类", 10, "E")
    dev, test = stratified_split(samples, dev_ratio=0.7, seed=0)

    dev_macros = {s.category_macro for s in dev}
    test_macros = {s.category_macro for s in test}
    assert dev_macros == {"历史类", "经济类"}
    assert test_macros == {"历史类", "经济类"}


def test_stratified_split_respects_ratio_per_bucket():
    samples = _make_samples("历史类", 10, "H")
    dev, test = stratified_split(samples, dev_ratio=0.7, seed=0)
    assert len(dev) == 7
    assert len(test) == 3


def test_stratified_split_is_deterministic_given_same_seed():
    samples = _make_samples("历史类", 20, "H")
    dev1, test1 = stratified_split(samples, seed=42)
    dev2, test2 = stratified_split(samples, seed=42)
    assert [s.sample_id for s in dev1] == [s.sample_id for s in dev2]
    assert [s.sample_id for s in test1] == [s.sample_id for s in test2]


def test_split_summary_counts_match():
    samples = _make_samples("历史类", 10, "H")
    dev, test = stratified_split(samples, dev_ratio=0.7, seed=0)
    summary = split_summary(dev, test)
    assert summary["n_dev"] == len(dev)
    assert summary["n_test"] == len(test)
