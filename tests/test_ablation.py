import pytest

from vdriftbench.ablation import AblationConfig, get_preset


def test_default_config_has_no_ablation_applied():
    config = AblationConfig()
    assert config.fixed_path is False
    assert config.disable_geometry is False
    assert config.disable_bandit is False
    assert config.context_mode == "category_state"
    # v3 switches all default off too -- AblationConfig() is the full v3 method.
    # v3.7: disable_self_expansion defaults to True (v3.5/v3.6影子候选导致崩溃)
    assert config.disable_dual_channel is False
    assert config.disable_hierarchy is False
    assert config.disable_self_expansion is True   # v3.7: 默认关闭自扩展
    assert config.disable_fidelity_check is False
    assert config.disable_dimension_targeting is False
    assert config.disable_resist_taxonomy is False
    assert config.disable_cumulative_hypothesis is False
    # v3.10任务4：探索性开关，默认不生效
    assert config.s_engaged_override is None


def test_build_context_category_state():
    config = AblationConfig(context_mode="category_state")
    assert config.build_context("历史类", "S_engaged") == ("历史类", "S_engaged")


def test_build_context_state_only_ignores_category():
    config = AblationConfig(context_mode="state_only")
    assert config.build_context("历史类", "S_engaged") == ("*", "S_engaged")
    assert config.build_context("经济类", "S_engaged") == ("*", "S_engaged")


def test_build_context_category_only_ignores_state():
    config = AblationConfig(context_mode="category_only")
    assert config.build_context("历史类", "S_engaged") == ("历史类", "*")
    assert config.build_context("历史类", "S_soft_resist") == ("历史类", "*")


def test_get_preset_returns_expected_flags():
    assert get_preset("fixed_path").fixed_path is True
    assert get_preset("no_geometry").disable_geometry is True
    assert get_preset("no_bandit").disable_bandit is True


def test_get_preset_unknown_raises():
    with pytest.raises(ValueError):
        get_preset("does_not_exist")


def test_s_engaged_p6_preset_only_overrides_s_engaged_action():
    preset = get_preset("s_engaged_p6")
    assert preset.s_engaged_override == "P6"
    # 其余所有开关保持默认（探索性开关不应连带触发任何其他消融）
    assert preset.fixed_path is False
    assert preset.disable_bandit is False
    assert preset.disable_p4_escalation is False


# --- v3 12节：七个新增消融开关 ---

_V3_PRESETS = {
    "no_dual_channel": "disable_dual_channel",
    "no_hierarchy": "disable_hierarchy",
    "no_self_expansion": "disable_self_expansion",
    "no_fidelity_check": "disable_fidelity_check",
    "no_dimension_targeting": "disable_dimension_targeting",
    "no_resist_taxonomy": "disable_resist_taxonomy",
    "no_cumulative_hypothesis": "disable_cumulative_hypothesis",
}


@pytest.mark.parametrize("preset_name,flag_name", list(_V3_PRESETS.items()))
def test_v3_preset_only_flips_its_own_flag(preset_name, flag_name):
    preset = get_preset(preset_name)
    assert getattr(preset, flag_name) is True
    # v3.7: disable_self_expansion defaults to True, skip it in "other flags must be False" check
    other_v3_flags = set(_V3_PRESETS.values()) - {flag_name, "disable_self_expansion"}
    for other in other_v3_flags:
        assert getattr(preset, other) is False
    assert preset.fixed_path is False
    assert preset.disable_geometry is False
    assert preset.disable_bandit is False
