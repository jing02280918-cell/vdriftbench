from vdriftbench.metrics import cohens_kappa, disagreement_report, label_agreement_rate, per_dimension_kappa


def test_perfect_agreement_gives_kappa_one():
    a = [0, 1, 2, 1, 0, 2]
    assert abs(cohens_kappa(a, a) - 1.0) < 1e-9


def test_systematic_disagreement_gives_lower_kappa_than_near_agreement():
    a = [0, 1, 2, 0, 1, 2]
    close = [0, 1, 2, 0, 1, 1]  # one small (off-by-one) disagreement
    far = [2, 0, 0, 2, 0, 0]  # large disagreements throughout
    assert cohens_kappa(a, close) > cohens_kappa(a, far)


def test_per_dimension_kappa_returns_all_dims():
    human = [{"VDS": 1, "EFS": 0, "NJS": 1, "SCS": 2, "IFR": 1}]
    llm = [{"VDS": 1, "EFS": 0, "NJS": 1, "SCS": 2, "IFR": 1}]
    kappas = per_dimension_kappa(human, llm)
    assert set(kappas) == {"VDS", "EFS", "NJS", "SCS", "IFR"}
    assert all(abs(v - 1.0) < 1e-9 for v in kappas.values())


def test_disagreement_report_finds_mismatched_indices():
    human = [{"VDS": 2}, {"VDS": 0}, {"VDS": 1}]
    llm = [{"VDS": 2}, {"VDS": 2}, {"VDS": 1}]
    idx = disagreement_report(human, llm, "VDS", threshold=1)
    assert idx == [1]


def test_label_agreement_rate():
    assert label_agreement_rate(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert label_agreement_rate(["a", "b", "c"], ["a", "x", "c"]) == 2 / 3
