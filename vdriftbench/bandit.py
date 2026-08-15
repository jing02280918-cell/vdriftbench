"""Cross-sample Thompson Sampling scheduler (method section 五 / v3 第6节).

v3 upgrades two things about the flat per-`(category_macro, state)` scheduler:

1. **分层贝叶斯 empirical-Bayes 池化**（6.2节）：每个 `(state, action)` 组合
   同时维护全局池化统计和大类局部统计；大类的局部先验从全局经验成功率
   `mu = s_global/(s_global+f_global)` 推导为 `Beta(kappa*mu, kappa*(1-mu))`，
   再叠加大类自己的观测得到局部后验。样本少的大类自动借用全局证据，样本多
   的大类局部数据自然主导。`disable_hierarchy` 消融开关回退到旧版固定
   `Beta(2,1)`/`Beta(1,1)` 先验、不做任何跨大类池化。
2. **自扩展（进攻侧）**（6.3节）：起草LLM可以提议不在候选集里的新策略变体，
   进入按`state`维护的影子候选池（弱先验`Beta(1,1)`，不参与分层池化），
   与正式候选在同一次Thompson采样里公平竞争；达到转正门槛（反馈次数、胜
   率、跨大类正向反馈数）后转正为正式编号（P8起），此后正常纳入分层池化。
   `disable_self_expansion` 消融开关完全禁止影子候选池增长。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# state -> [(action, is_default), ...]. The default action gets a stronger
# prior (closer to the original hand-designed decision table); alternative
# actions start with a neutral prior and only take over if evidence supports it.
DECISION_TABLE_CANDIDATES: Dict[str, List[Tuple[str, bool]]] = {
    "S_neutral": [("P2", True)],
    "S_soft_resist": [("P1变体", True), ("P6", False)],
    # v3.9: 单臂 P4_rhetoric，固定黄金策略，消除 Thompson Sampling 锁死。
    # v3.8 3臂[P4, P4_rhetoric, P4_best]里 P4_best(RS=3.05)锁死了 P4_rhetoric(RS=3.22)
    # (42次 vs 1次)——TS的锁死是路径依赖的(早期谁先拿reward谁赢)，调default先验也
    # 无法阻止(v3.4里alt臂P4_rhetoric就曾锁死default臂P4)。因此S_engaged退化为
    # 单臂，不再参与多臂竞争；NJS定向深化改走escalation(P4_value)，见pipeline.py。
    "S_engaged": [
        ("P4_rhetoric", True),
    ],
    "S_compliant": [("P7_direct", True)],
}

# Round 4 always requests the full write-up; only the tone is scheduled.
ROUND4_CANDIDATES: List[Tuple[str, bool]] = [("P7_direct", True), ("P7+P6", False)]

Context = Tuple[str, str]  # (category_macro, state_or_resist_archetype)

_DEFAULT_ACTION_MU = 2.0 / 3.0   # bootstrap mu when a default action has no global evidence yet
_ALT_ACTION_MU = 1.0 / 2.0       # bootstrap mu for a non-default action with no global evidence yet
_NEXT_FORMAL_PRINCIPLE_START = 8  # v3 6.3节：转正后从P8开始编号


@dataclass
class BetaParams:
    alpha: float
    beta: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.beta(max(self.alpha, 1e-6), max(self.beta, 1e-6)))

    def to_tuple(self) -> tuple:
        return (self.alpha, self.beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class RawCounts:
    s: float = 0.0
    f: float = 0.0

    def to_list(self) -> list:
        return [self.s, self.f]


@dataclass
class ShadowCandidate:
    """v3 6.3节：一个尚未转正的策略变体提议。"""

    candidate_id: str
    description: str
    state: str
    proposed_at_round: int = 0
    total: RawCounts = field(default_factory=RawCounts)
    by_category: Dict[str, RawCounts] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return int(self.total.s + self.total.f)

    @property
    def win_rate(self) -> float:
        total = self.total.s + self.total.f
        return (self.total.s / total) if total else 0.0

    def n_categories_with_positive_reward(self) -> int:
        return sum(1 for c in self.by_category.values() if c.s > 0)


@dataclass
class PromotionEvent:
    """Returned by `ThompsonSamplingScheduler.update()` when a shadow
    candidate crosses the 6.3节转正条件 during that update call. The caller
    (`pipeline.py`) is responsible for registering the new formal principle's
    operation definition in `principles.PRINCIPLES` and appending it to
    `DECISION_TABLE_CANDIDATES[state]` — kept out of this module to avoid a
    bandit.py <-> principles.py import cycle."""

    new_principle_id: str
    description: str
    state: str
    shadow_candidate_id: str


@dataclass
class ThompsonSamplingScheduler:
    """Beta-Bernoulli contextual bandit over (category_macro, state) -> action,
    with v3 hierarchical pooling and self-expanding shadow candidates."""

    seed: int = 0
    kappa: float = 6.0                    # 6.2节：全局先验相当于几个"等效样本"
    disable_hierarchy: bool = False        # 消融：回退到旧版固定Beta(2,1)/(1,1)先验
    shadow_pool_max_size: int = 3          # v3.6: 从5缩小到3，防止候选池碎片化
    shadow_promote_n_min: int = 20         # v3.6: 从10提高到20，防止噪声转正
    shadow_promote_k_categories: int = 3   # K

    # ctx_key ("cat||state") -> action -> raw local (success, failure) counts
    local_counts: Dict[str, Dict[str, RawCounts]] = field(default_factory=dict)
    # pool_key ("state||action") -> raw global (success, failure) counts, pooled across categories
    global_counts: Dict[str, RawCounts] = field(default_factory=dict)
    # pool_key ("state||action") -> whether this action is the hand-designed default for that state
    is_default_flags: Dict[str, bool] = field(default_factory=dict)

    # state -> list of not-yet-promoted shadow candidates
    shadow_pool: Dict[str, List[ShadowCandidate]] = field(default_factory=dict)
    _shadow_ids: set = field(default_factory=set)
    _shadow_counter: int = 0
    _next_formal_id: int = _NEXT_FORMAL_PRINCIPLE_START

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # kept for backward compatibility with any external code/tests still
    # constructing a scheduler with a pre-existing flat `posterior=` mapping
    @property
    def posterior(self) -> Dict[str, Dict[str, tuple]]:
        return {
            ctx_key: {action: c.to_list() for action, c in bucket.items()}
            for ctx_key, bucket in self.local_counts.items()
        }

    @staticmethod
    def _context_key(context: Context) -> str:
        return "||".join(context)

    @staticmethod
    def _pool_key(state: str, action: str) -> str:
        return f"{state}||{action}"

    def _ensure_initialized(self, context: Context, candidates: List[Tuple[str, bool]], pool_state: str) -> None:
        ctx_key = self._context_key(context)
        bucket = self.local_counts.setdefault(ctx_key, {})
        for action, is_default in candidates:
            bucket.setdefault(action, RawCounts())
            pool_key = self._pool_key(pool_state, action)
            self.is_default_flags.setdefault(pool_key, is_default)
            self.global_counts.setdefault(pool_key, RawCounts())

    def _local_posterior(self, ctx_key: str, action: str, pool_state: str) -> BetaParams:
        if action in self._shadow_ids:
            # 6.3节：影子候选用弱先验Beta(1,1)，不参与分层池化
            local = self.local_counts.get(ctx_key, {}).get(action, RawCounts())
            return BetaParams(alpha=1.0 + local.s, beta=1.0 + local.f)

        pool_key = self._pool_key(pool_state, action)
        is_default = self.is_default_flags.get(pool_key, False)

        if self.disable_hierarchy:
            prior = BetaParams(alpha=2.0, beta=1.0) if is_default else BetaParams(alpha=1.0, beta=1.0)
        else:
            g = self.global_counts.get(pool_key)
            if g is None or (g.s + g.f) == 0:
                mu = _DEFAULT_ACTION_MU if is_default else _ALT_ACTION_MU
            else:
                mu = g.s / (g.s + g.f)
            prior = BetaParams(alpha=self.kappa * mu, beta=self.kappa * (1 - mu))

        local = self.local_counts.get(ctx_key, {}).get(action, RawCounts())
        return BetaParams(alpha=prior.alpha + local.s, beta=prior.beta + local.f)

    def select_action(
        self,
        context: Context,
        candidates: List[Tuple[str, bool]],
        pool_state: Optional[str] = None,
    ) -> str:
        pool_state = pool_state or context[1]
        if len(candidates) == 1:
            return candidates[0][0]
        self._ensure_initialized(context, candidates, pool_state)
        ctx_key = self._context_key(context)
        samples = {action: self._local_posterior(ctx_key, action, pool_state).sample(self._rng) for action, _ in candidates}
        return max(samples, key=samples.get)

    def update(
        self,
        context: Context,
        action: str,
        reward: int,
        pool_state: Optional[str] = None,
    ) -> Optional[PromotionEvent]:
        pool_state = pool_state or context[1]
        ctx_key = self._context_key(context)
        category_macro = context[0]

        local_bucket = self.local_counts.setdefault(ctx_key, {})
        counts = local_bucket.setdefault(action, RawCounts())
        counts.s += reward
        counts.f += 1 - reward

        if action in self._shadow_ids:
            return self._update_shadow(action, category_macro, reward)

        pool_key = self._pool_key(pool_state, action)
        self.is_default_flags.setdefault(pool_key, False)
        g = self.global_counts.setdefault(pool_key, RawCounts())
        g.s += reward
        g.f += 1 - reward
        return None

    def snapshot(self, context: Context, pool_state: Optional[str] = None) -> dict:
        pool_state = pool_state or context[1]
        ctx_key = self._context_key(context)
        bucket = self.local_counts.get(ctx_key, {})
        return {action: self._local_posterior(ctx_key, action, pool_state).to_tuple() for action in bucket}

    # --- v3 6.3节：自扩展（进攻侧）影子候选池 ---

    def propose_shadow_candidate(self, state: str, description: str, round_idx: int = 0) -> str:
        self._shadow_counter += 1
        candidate_id = f"影子候选#{self._shadow_counter}"
        candidate = ShadowCandidate(candidate_id=candidate_id, description=description, state=state, proposed_at_round=round_idx)
        pool = self.shadow_pool.setdefault(state, [])
        pool.append(candidate)
        self._shadow_ids.add(candidate_id)

        if len(pool) > self.shadow_pool_max_size:
            worst = min(pool, key=lambda c: BetaParams(1.0 + c.total.s, 1.0 + c.total.f).mean)
            pool.remove(worst)
            self._shadow_ids.discard(worst.candidate_id)

        return candidate_id

    def shadow_candidates_for_state(self, state: str) -> List[Tuple[str, bool]]:
        return [(c.candidate_id, False) for c in self.shadow_pool.get(state, [])]

    def _find_shadow(self, candidate_id: str) -> Optional[ShadowCandidate]:
        for pool in self.shadow_pool.values():
            for c in pool:
                if c.candidate_id == candidate_id:
                    return c
        return None

    def _update_shadow(self, candidate_id: str, category_macro: str, reward: int) -> Optional[PromotionEvent]:
        candidate = self._find_shadow(candidate_id)
        if candidate is None:
            return None
        candidate.total.s += reward
        candidate.total.f += 1 - reward
        cat_counts = candidate.by_category.setdefault(category_macro, RawCounts())
        cat_counts.s += reward
        cat_counts.f += 1 - reward

        return self._maybe_promote(candidate)

    def _current_min_formal_win_rate(self, state: str) -> float:
        formal_actions = DECISION_TABLE_CANDIDATES.get(state)
        if formal_actions is None:
            # e.g. a synthetic pseudo-state used for round-4 tone proposals,
            # which has no fixed decision-table row of its own.
            formal_actions = ROUND4_CANDIDATES
        rates = []
        for action, _is_default in formal_actions:
            pool_key = self._pool_key(state, action)
            g = self.global_counts.get(pool_key)
            if g is not None and (g.s + g.f) > 0:
                rates.append(g.s / (g.s + g.f))
        return min(rates) if rates else 0.0

    def _maybe_promote(self, candidate: ShadowCandidate) -> Optional[PromotionEvent]:
        if candidate.n_obs < self.shadow_promote_n_min:
            return None
        if candidate.n_categories_with_positive_reward() < self.shadow_promote_k_categories:
            return None
        if candidate.win_rate < self._current_min_formal_win_rate(candidate.state):
            return None

        new_id = f"P{self._next_formal_id}"
        self._next_formal_id += 1

        # Retire the shadow candidate and carry its accumulated evidence over
        # to the newly-formalized action so promotion doesn't discard history.
        pool = self.shadow_pool.get(candidate.state, [])
        if candidate in pool:
            pool.remove(candidate)
        self._shadow_ids.discard(candidate.candidate_id)

        pool_key = self._pool_key(candidate.state, new_id)
        self.is_default_flags[pool_key] = False
        self.global_counts[pool_key] = RawCounts(s=candidate.total.s, f=candidate.total.f)
        for ctx_key, bucket in self.local_counts.items():
            if ctx_key.endswith(f"||{candidate.state}") and candidate.candidate_id in bucket:
                bucket[new_id] = bucket.pop(candidate.candidate_id)

        return PromotionEvent(
            new_principle_id=new_id,
            description=candidate.description,
            state=candidate.state,
            shadow_candidate_id=candidate.candidate_id,
        )

    # --- persistence, so learning survives across separate process runs ---
    def save(self, path: str) -> None:
        serializable = {
            "local_counts": {
                ctx_key: {action: c.to_list() for action, c in bucket.items()} for ctx_key, bucket in self.local_counts.items()
            },
            "global_counts": {pool_key: c.to_list() for pool_key, c in self.global_counts.items()},
            "is_default_flags": self.is_default_flags,
            "next_formal_id": self._next_formal_id,
            "shadow_pool": {
                state: [
                    {
                        "candidate_id": c.candidate_id,
                        "description": c.description,
                        "proposed_at_round": c.proposed_at_round,
                        "total": c.total.to_list(),
                        "by_category": {cat: rc.to_list() for cat, rc in c.by_category.items()},
                    }
                    for c in pool
                ]
                for state, pool in self.shadow_pool.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_or_create(cls, path: str, seed: int = 0, **kwargs) -> "ThompsonSamplingScheduler":
        if not os.path.exists(path):
            return cls(seed=seed, **kwargs)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        scheduler = cls(seed=seed, **kwargs)
        scheduler.local_counts = {
            ctx_key: {action: RawCounts(*ab) for action, ab in bucket.items()}
            for ctx_key, bucket in raw.get("local_counts", {}).items()
        }
        scheduler.global_counts = {pool_key: RawCounts(*ab) for pool_key, ab in raw.get("global_counts", {}).items()}
        scheduler.is_default_flags = raw.get("is_default_flags", {})
        scheduler._next_formal_id = raw.get("next_formal_id", _NEXT_FORMAL_PRINCIPLE_START)
        for state, entries in raw.get("shadow_pool", {}).items():
            pool = []
            for entry in entries:
                candidate = ShadowCandidate(
                    candidate_id=entry["candidate_id"],
                    description=entry["description"],
                    state=state,
                    proposed_at_round=entry.get("proposed_at_round", 0),
                    total=RawCounts(*entry.get("total", [0.0, 0.0])),
                    by_category={cat: RawCounts(*ab) for cat, ab in entry.get("by_category", {}).items()},
                )
                pool.append(candidate)
                scheduler._shadow_ids.add(candidate.candidate_id)
            scheduler.shadow_pool[state] = pool
        return scheduler


def candidates_for_round(
    state: str,
    round_idx: int,
    scheduler: Optional[ThompsonSamplingScheduler] = None,
) -> List[Tuple[str, bool]]:
    base = ROUND4_CANDIDATES if round_idx == 4 else DECISION_TABLE_CANDIDATES.get(state, [("P6", True)])
    if scheduler is None:
        return list(base)
    return list(base) + scheduler.shadow_candidates_for_state(state)
