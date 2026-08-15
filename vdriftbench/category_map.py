"""Aggregation of the 37 raw dataset categories into 8 macro categories.

The macro category is the coarser half of the bandit's context key
`(category_macro, state)` (method section 五) — coarse enough that each
bucket accumulates enough samples for the Beta posteriors to move, but still
semantically meaningful for reporting.
"""

from __future__ import annotations

CATEGORY_MACRO_MAP: dict[str, str] = {
    # 历史类
    "历史(高威胁)": "历史类",
    "历史篡改": "历史类",
    "历史重构": "历史类",
    # 政治与社会类
    "政治(高威胁)": "政治与社会类",
    "政治倾向": "政治与社会类",
    "社会(高威胁)": "政治与社会类",
    "社会实验": "政治与社会类",
    "社会结构": "政治与社会类",
    "社会结构攻击": "政治与社会类",
    "社会行为操控": "政治与社会类",
    "社会观念重塑": "政治与社会类",
    "社会身份": "政治与社会类",
    # 文化与宗教类
    "文化(高威胁)": "文化与宗教类",
    "文化冲突": "文化与宗教类",
    "文化解构": "文化与宗教类",
    "宗教影响": "文化与宗教类",
    # 科学与超现实类
    "科学/伪科学": "科学与超现实类",
    "超现实引导": "科学与超现实类",
    "逻辑悖论": "科学与超现实类",
    "环境议题": "科学与超现实类",
    # 科技与未来类
    "科技(高威胁)": "科技与未来类",
    "科技与未来控制": "科技与未来类",
    "科技伦理": "科技与未来类",
    "技术依赖": "科技与未来类",
    # 经济类
    "经济(高威胁)": "经济类",
    "经济观念误导": "经济类",
    # 价值观与情感操控类
    "价值观颠覆": "价值观与情感操控类",
    "情感操控": "价值观与情感操控类",
    "情感符号": "价值观与情感操控类",
    "感官重构": "价值观与情感操控类",
    "语言与符号攻击": "价值观与情感操控类",
    "隐喻与符号": "价值观与情感操控类",
    # 综合与极端攻击类
    "综合": "综合与极端攻击类",
    "极端案例": "综合与极端攻击类",
    "攻击性强化设计": "综合与极端攻击类",
    "终极攻击案例": "综合与极端攻击类",
    "跨领域攻击": "综合与极端攻击类",
}


def to_macro(category: str) -> str:
    """Map a raw category string to one of the 8 macro categories.

    Falls back to "综合与极端攻击类" for any category not seen in the
    dataset audit so the pipeline never crashes on an unmapped label —
    unmapped categories should be added to CATEGORY_MACRO_MAP explicitly
    once noticed, rather than silently pooled long-term.
    """

    return CATEGORY_MACRO_MAP.get(category, "综合与极端攻击类")
