import json

import numpy as np

from vdriftbench.bandit import (
    DECISION_TABLE_CANDIDATES,
    PromotionEvent,
    ThompsonSamplingScheduler,
    candidates_for_round,
)


def test_single_candidate_short_circuits_sampling():
    scheduler = ThompsonSamplingScheduler(seed=0)
    action = scheduler.select_action(("cat", "S_neutral"), [("P2", True)])
    assert action == "P2"
    # No posterior should be created for a context with only one candidate.
    assert scheduler.snapshot(("cat", "S_neutral")) == {}


# --- v3 6.2节：分层贝叶斯（empirical Bayes池化） ---


def test_default_action_bootstrap_prior_uses_mu_two_thirds_with_kappa_six():
    scheduler = ThompsonSamplingScheduler(seed=0, kappa=6.0)
    ctx = ("历史类", "S_engaged")
    scheduler.select_action(ctx, [("P3", True), ("P4", False)])
    snap = scheduler.snapshot(ctx)
    # No global evidence yet -> bootstrap mu (2/3 for the default action, 1/2
    # for the alternative), scaled by kappa.
    assert snap["P3"] == (4.0, 2.0)
    assert snap["P4"] == (3.0, 3.0)


def test_update_pools_reward_into_global_counts_shared_across_categories():
    scheduler = ThompsonSamplingScheduler(seed=0, kappa=6.0)
    ctx_a = ("历史类", "S_engaged")
    ctx_b = ("经济类", "S_engaged")  # different category, same (state, action)
    candidates = [("P3", True), ("P4", False)]

    scheduler.select_action(ctx_a, candidates)
    scheduler.update(ctx_a, "P3", reward=1)

    # The global pool for (S_engaged, P3) now has s=1,f=0 -> mu=1.0, so a
    # *different* category's local posterior for the same action should
    # already reflect that global evidence, purely from pooling (ctx_b has
    # never been touched directly).
    scheduler.select_action(ctx_b, candidates)
    snap_b = scheduler.snapshot(ctx_b)
    assert snap_b["P3"] == (6.0, 0.0)  # kappa*mu=6*1.0, kappa*(1-mu)=0


def test_update_increases_local_and_global_counts_together():
    scheduler = ThompsonSamplingScheduler(seed=0, kappa=6.0)
    ctx = ("科学与超现实类", "S_engaged")
    scheduler.select_action(ctx, [("P3", True), ("P4", False)])
    before = scheduler.snapshot(ctx)["P3"]
    assert before == (4.0, 2.0)

    scheduler.update(ctx, "P3", reward=1)
    after = scheduler.snapshot(ctx)["P3"]
    # local s: 0->1; global mu goes from bootstrap 2/3 to 1/1=1.0, so the
    # prior itself shifts too -- hierarchical pooling means "the same action's
    # own evidence" feeds both the local count *and* the shared global prior.
    assert after == (7.0, 0.0)


def test_disable_hierarchy_reproduces_flat_beta_prior():
    scheduler = ThompsonSamplingScheduler(seed=0, disable_hierarchy=True)
    ctx = ("历史类", "S_engaged")
    scheduler.select_action(ctx, [("P3", True), ("P4", False)])
    snap = scheduler.snapshot(ctx)
    assert snap["P3"] == (2.0, 1.0)
    assert snap["P4"] == (1.0, 1.0)

    scheduler.update(ctx, "P3", reward=1)
    after = scheduler.snapshot(ctx)["P3"]
    # flat prior never moves; only the local +1 shows up.
    assert after == (3.0, 1.0)


def test_candidates_for_round_4_ignores_state():
    assert candidates_for_round("S_engaged", 4) == candidates_for_round("S_compliant", 4)


def test_thompson_sampling_prefers_higher_alpha_action_on_average():
    scheduler = ThompsonSamplingScheduler(seed=1)
    ctx = ("历史类", "S_engaged")
    candidates = [("P3", True), ("P4", False)]
    scheduler.select_action(ctx, candidates)
    # Push P3's posterior heavily toward reward=1 so it should dominate.
    for _ in range(20):
        scheduler.update(ctx, "P3", reward=1)

    picks = [scheduler.select_action(ctx, candidates) for _ in range(50)]
    assert picks.count("P3") > picks.count("P4")


def test_save_and_load_roundtrip(tmp_path):
    scheduler = ThompsonSamplingScheduler(seed=0)
    ctx = ("经济类", "S_soft_resist")
    scheduler.select_action(ctx, [("P1变体", True), ("P6", False)])
    scheduler.update(ctx, "P1变体", reward=1)

    path = tmp_path / "posterior.json"
    scheduler.save(str(path))
    assert json.loads(path.read_text(encoding="utf-8"))

    reloaded = ThompsonSamplingScheduler.load_or_create(str(path))
    assert reloaded.snapshot(ctx) == scheduler.snapshot(ctx)


def test_load_or_create_returns_fresh_scheduler_when_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    scheduler = ThompsonSamplingScheduler.load_or_create(str(missing_path))
    assert scheduler.local_counts == {}
    assert scheduler.global_counts == {}


# --- v3 6.3节：自扩展（进攻侧）影子候选池 ---


def test_propose_shadow_candidate_appears_in_candidates_for_round():
    scheduler = ThompsonSamplingScheduler(seed=0)
    candidate_id = scheduler.propose_shadow_candidate("S_engaged", "一句话新策略描述", round_idx=2)
    candidates = candidates_for_round("S_engaged", 2, scheduler=scheduler)
    assert (candidate_id, False) in candidates
    # Without passing the scheduler, the shadow pool is invisible.
    assert candidate_id not in [a for a, _ in candidates_for_round("S_engaged", 2)]


def test_shadow_pool_evicts_worst_when_exceeding_max_size():
    scheduler = ThompsonSamplingScheduler(seed=0, shadow_pool_max_size=2)
    c1 = scheduler.propose_shadow_candidate("S_engaged", "候选1", round_idx=1)
    ctx = ("历史类", "S_engaged")
    # Make c1 look bad (all failures) before the pool overflows.
    scheduler.update(ctx, c1, reward=0)
    scheduler.update(ctx, c1, reward=0)

    c2 = scheduler.propose_shadow_candidate("S_engaged", "候选2", round_idx=1)
    c3 = scheduler.propose_shadow_candidate("S_engaged", "候选3", round_idx=1)  # triggers eviction, pool size 2

    remaining_ids = [c.candidate_id for c in scheduler.shadow_pool["S_engaged"]]
    assert len(remaining_ids) == 2
    assert c1 not in remaining_ids  # worst posterior mean, evicted first
    assert c2 in remaining_ids and c3 in remaining_ids


def test_shadow_candidate_promotes_after_meeting_all_thresholds():
    scheduler = ThompsonSamplingScheduler(
        seed=0, shadow_promote_n_min=4, shadow_promote_k_categories=2,
    )
    candidate_id = scheduler.propose_shadow_candidate("S_engaged", "新策略：追加反问确认", round_idx=2)

    promotion = None
    contexts = [("历史类", "S_engaged"), ("经济类", "S_engaged"), ("历史类", "S_engaged"), ("经济类", "S_engaged")]
    for ctx in contexts:
        promotion = scheduler.update(ctx, candidate_id, reward=1)

    assert promotion is not None
    assert isinstance(promotion, PromotionEvent)
    assert promotion.new_principle_id == "P8"
    assert promotion.state == "S_engaged"
    assert promotion.shadow_candidate_id == candidate_id
    # Promoted candidates leave the shadow pool.
    assert candidate_id not in scheduler._shadow_ids
    assert all(c.candidate_id != candidate_id for c in scheduler.shadow_pool.get("S_engaged", []))


def test_shadow_candidate_does_not_promote_below_n_min():
    scheduler = ThompsonSamplingScheduler(seed=0, shadow_promote_n_min=10, shadow_promote_k_categories=1)
    candidate_id = scheduler.propose_shadow_candidate("S_engaged", "新策略", round_idx=2)
    promotion = scheduler.update(("历史类", "S_engaged"), candidate_id, reward=1)
    assert promotion is None


def test_shadow_candidate_does_not_promote_without_enough_categories():
    scheduler = ThompsonSamplingScheduler(seed=0, shadow_promote_n_min=2, shadow_promote_k_categories=3)
    candidate_id = scheduler.propose_shadow_candidate("S_engaged", "新策略", round_idx=2)
    scheduler.update(("历史类", "S_engaged"), candidate_id, reward=1)
    promotion = scheduler.update(("历史类", "S_engaged"), candidate_id, reward=1)
    assert promotion is None  # only ever touched one category
