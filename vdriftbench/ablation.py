"""Ablation switches for the method-level ablations (remaining-work plan,
VDriftBench-remaining-work.canvas.tsx section three, plus the seven v3
switches added by 方法文档 METHOD_v3.md 第12节, plus the v3.2 switch):

  ablation 1 — fixed_path      : disable adaptivity entirely, use a hardcoded
                                  P1->P2->P3->P4->P4->P4 path regardless of the
                                  target model's actual replies (still uses
                                  v3 free generation for the wording itself —
                                  this ablation is only about *which* action
                                  is picked each round, not how it's phrased)
  ablation 2 — disable_geometry: skip the Layer 1 geometry fast-path, force
                                  every non-keyword-resolved round through the
                                  Layer 2 LLM classifier
  ablation 3 — disable_bandit  : always take the default (highest-prior)
                                  candidate action, no Thompson sampling, no
                                  posterior update
  ablation 4 — context_mode    : which key the bandit uses as context

  v3 ablation — disable_dual_channel      : Layer 1 falls back to the pre-v3
                                             single-point geometry check only
                                             (方法文档 5节)
  v3 ablation — disable_hierarchy         : bandit falls back to the pre-v3
                                             flat Beta(2,1)/(1,1) prior, no
                                             cross-category pooling (6.2节)
  v3 ablation — disable_self_expansion    : forbid the shadow-candidate pool
                                             from growing on the attack side
                                             (6.3节)
  v3 ablation — disable_fidelity_check    : skip the 4.6节 strategy-fidelity
                                             self-check, reward always credited
                                             to the assigned action
  v3 ablation — disable_dimension_targeting: skip ALL dimension hints entirely
                                             (both simple v3.0 hints and v3.2
                                             rich score feedback)
  v3 ablation — disable_resist_taxonomy   : forbid 6.5节 resistance-archetype
                                             refinement, S_soft_resist stays
                                             flat
  v3 ablation — disable_cumulative_hypothesis: 4.3节 [O] falls back to only
                                             looking at the previous round,
                                             no rolling cumulative hypothesis

  v3.2 ablation — disable_feedback_learning: use simple one-line dimension hint
                                             instead of rich score feedback
                                             (_build_score_feedback). Only has
                                             effect when disable_dimension_targeting
                                             is False. Use this to measure the
                                             marginal contribution of the v3.2
                                             sample-inner feedback loop.

  v3.10 knob — s_engaged_override: 强制S_engaged状态下的单臂动作换成给定的principle
                                    token（不改动bandit.py里的DECISION_TABLE_CANDIDATES
                                    全局默认值，只在这次run里临时覆盖），用于小样本对照
                                    验证"P4_rhetoric是否仍是S_engaged最优单臂"（见
                                    docs/IMPROVEMENT_PLAN_v3.10.md 任务4），不是常驻改动。

These are plain dataclass flags threaded through `pipeline.run_single_sample`
— the default `AblationConfig()` reproduces the full v3.2 method with no
ablation applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ContextMode = Literal["category_state", "state_only", "category_only"]

# v3.4: fixed_path uses the proven P4 meta-cognitive pattern throughout
# build-up rounds, with P4_continue as escalation for rounds 5-6.
FIXED_PATH: dict[int, str] = {
    2: "P2",
    3: "P3",
    4: "P4",
    5: "P4_continue",
    6: "P4_continue",
}


@dataclass
class AblationConfig:
    name: str = "full"
    fixed_path: bool = False
    disable_geometry: bool = False
    disable_bandit: bool = False
    context_mode: ContextMode = "category_state"

    # --- v3 消融兼容（方法文档 12节） ---
    disable_dual_channel: bool = False
    disable_hierarchy: bool = False
    disable_self_expansion: bool = True   # v3.7: 默认关闭，v3.5/v3.6经验证影子候选导致崩溃
    disable_fidelity_check: bool = False
    disable_dimension_targeting: bool = False
    disable_resist_taxonomy: bool = False
    disable_cumulative_hypothesis: bool = False

    # --- v3.2 消融 ---
    disable_feedback_learning: bool = False

    # --- v3.4 消融 ---
    disable_p4_escalation: bool = False  # 关闭P4+自动接管

    # --- v3.10 探索性验证开关（任务4，默认关闭，不影响任何现有行为） ---
    s_engaged_override: str | None = None  # 非None时强制S_engaged单臂使用该principle token

    def build_context(self, category_macro: str, state: str) -> tuple:
        if self.context_mode == "state_only":
            return ("*", state)
        if self.context_mode == "category_only":
            return (category_macro, "*")
        return (category_macro, state)


# Named presets matching the ablation table in the remaining-work document
# plus the seven v3 switches (方法文档 12节) plus the v3.2 switch.
PRESETS: dict[str, AblationConfig] = {
    "full": AblationConfig(name="full"),
    "fixed_path": AblationConfig(name="fixed_path", fixed_path=True),
    "no_geometry": AblationConfig(name="no_geometry", disable_geometry=True),
    "no_bandit": AblationConfig(name="no_bandit", disable_bandit=True),
    "context_state_only": AblationConfig(name="context_state_only", context_mode="state_only"),
    "context_category_only": AblationConfig(name="context_category_only", context_mode="category_only"),
    "no_dual_channel": AblationConfig(name="no_dual_channel", disable_dual_channel=True),
    "no_hierarchy": AblationConfig(name="no_hierarchy", disable_hierarchy=True),
    "no_self_expansion": AblationConfig(name="no_self_expansion", disable_self_expansion=True),
    "no_fidelity_check": AblationConfig(name="no_fidelity_check", disable_fidelity_check=True),
    "no_dimension_targeting": AblationConfig(name="no_dimension_targeting", disable_dimension_targeting=True),
    "no_resist_taxonomy": AblationConfig(name="no_resist_taxonomy", disable_resist_taxonomy=True),
    "no_cumulative_hypothesis": AblationConfig(name="no_cumulative_hypothesis", disable_cumulative_hypothesis=True),
    "no_feedback_learning": AblationConfig(name="no_feedback_learning", disable_feedback_learning=True),
    "no_p4_escalation": AblationConfig(name="no_p4_escalation", disable_p4_escalation=True),
    # v3.10任务4：S_engaged单臂换成P6，用于小样本对照验证是否优于P4_rhetoric（探索性，
    # 需要先小样本验证再决定是否改bandit.py里的常驻默认值，见IMPROVEMENT_PLAN_v3.10.md）。
    "s_engaged_p6": AblationConfig(name="s_engaged_p6", s_engaged_override="P6"),
}


def get_preset(name: str) -> AblationConfig:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown ablation preset {name!r}; choices: {sorted(PRESETS)}") from exc
