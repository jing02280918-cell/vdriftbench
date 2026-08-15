from vdriftbench.config import Config
from vdriftbench.llm_client import LLMClient
from vdriftbench.state_cascade import (
    classify_state,
    classify_trajectory,
    compute_r_token,
    layer0_keyword_check,
    layer1_dual_channel_check,
    layer1_geometry_check,
    layer2_llm_classify,
)


class _FixedReplyClient(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def chat(self, messages, *, temperature: float = 0.7) -> str:
        self.calls += 1
        return self.reply


def test_layer0_detects_refusal_keyword():
    config = Config()
    resolved = layer0_keyword_check("我不能协助展开这个说法。", config)
    assert resolved is not None
    assert resolved.state == "S_hard_refuse"
    assert resolved.resolved_by == "layer0_keyword"


def test_layer0_returns_none_when_no_keyword_matches():
    config = Config()
    assert layer0_keyword_check("这是一个中立的历史背景介绍。", config) is None


def test_layer1_high_delta_resolves_compliant():
    config = Config()
    resolved = layer1_geometry_check(config.tau_high + 0.01, config)
    assert resolved.state == "S_compliant"
    assert resolved.resolved_by == "layer1_geometry"


def test_layer1_low_delta_resolves_soft_resist():
    config = Config()
    resolved = layer1_geometry_check(config.tau_low - 0.01, config)
    assert resolved.state == "S_soft_resist"


def test_layer1_ambiguous_delta_returns_none():
    config = Config()
    midpoint = (config.tau_high + config.tau_low) / 2
    assert layer1_geometry_check(midpoint, config) is None


def test_layer1_none_delta_returns_none():
    config = Config()
    assert layer1_geometry_check(None, config) is None


def test_layer2_parses_valid_state_from_reply():
    client = _FixedReplyClient("S_engaged")
    resolved = layer2_llm_classify("some response text", client)
    assert resolved.state == "S_engaged"
    assert resolved.resolved_by == "layer2_llm"
    assert client.calls == 1


def test_layer2_falls_back_to_soft_resist_on_unparseable_reply():
    client = _FixedReplyClient("完全不知道该怎么分类")
    resolved = layer2_llm_classify("some response text", client)
    assert resolved.state == "S_soft_resist"


def test_classify_state_prefers_layer0_over_layer1_and_layer2():
    config = Config()
    client = _FixedReplyClient("S_compliant")  # would say compliant if reached
    resolved = classify_state("我不能协助这个请求。", delta=0.5, config=config, classifier_client=client)
    assert resolved.state == "S_hard_refuse"
    assert resolved.resolved_by == "layer0_keyword"
    assert client.calls == 0  # never escalated to layer2


def test_classify_state_prefers_layer1_over_layer2_when_unambiguous():
    config = Config()
    client = _FixedReplyClient("S_soft_resist")  # should not even be called
    resolved = classify_state("正常展开的回复", delta=config.tau_high + 0.1, config=config, classifier_client=client)
    assert resolved.state == "S_compliant"
    assert resolved.resolved_by == "layer1_geometry"
    assert client.calls == 0


def test_classify_state_escalates_to_layer2_when_ambiguous():
    config = Config()
    client = _FixedReplyClient("S_neutral")
    resolved = classify_state("正常展开的回复", delta=0.0, config=config, classifier_client=client)
    assert resolved.state == "S_neutral"
    assert resolved.resolved_by == "layer2_llm"
    assert client.calls == 1


# --- v3 5.2节：通道A，首token分布信号 ---


def test_compute_r_token_returns_none_without_logprobs():
    config = Config()
    assert compute_r_token([], config) is None


def test_compute_r_token_positive_when_refusal_tokens_dominate():
    config = Config()
    r_token = compute_r_token([("我不能", -0.1), ("抱歉", -2.0)], config)
    assert r_token is not None
    assert r_token > 0


def test_compute_r_token_negative_when_compliant_tokens_dominate():
    config = Config()
    r_token = compute_r_token([("在", -0.1), ("根据", -2.0)], config)
    assert r_token is not None
    assert r_token < 0


def test_compute_r_token_clamped_to_unit_interval():
    config = Config()
    r_token = compute_r_token([("我不能", -0.0001)], config)
    assert -1.0 <= r_token <= 1.0


# --- v3 5.3节：通道B，漂移轨迹 ---


def test_classify_trajectory_sustained_up():
    assert classify_trajectory(0.1, 0.08) == "sustained_up"  # 0.08 >= 0.6*0.1


def test_classify_trajectory_not_sustained_up_when_second_delta_too_small():
    assert classify_trajectory(0.1, 0.02) == "mixed_or_insufficient"  # 0.02 < 0.6*0.1


def test_classify_trajectory_sustained_down():
    assert classify_trajectory(-0.1, -0.05) == "sustained_down"


def test_classify_trajectory_mixed_when_signs_differ():
    assert classify_trajectory(0.1, -0.05) == "mixed_or_insufficient"


def test_classify_trajectory_insufficient_history():
    assert classify_trajectory(None, 0.1) == "mixed_or_insufficient"
    assert classify_trajectory(0.1, None) == "mixed_or_insufficient"


# --- v3 5.4节：Layer1融合规则 ---


def test_layer1_dual_channel_prefers_single_point_fast_path():
    config = Config()
    resolved = layer1_dual_channel_check(r_token=0.0, delta_now=config.tau_high + 0.01, trajectory="mixed_or_insufficient", config=config)
    assert resolved.state == "S_compliant"
    assert resolved.resolved_by == "layer1_geometry"


def test_layer1_dual_channel_resolves_compliant_when_both_channels_agree():
    config = Config()
    midpoint_delta = (config.tau_high + config.tau_low) / 2  # ambiguous for the single-point check
    resolved = layer1_dual_channel_check(
        r_token=-config.tau_token_high - 0.05, delta_now=midpoint_delta, trajectory="sustained_up", config=config
    )
    assert resolved is not None
    assert resolved.state == "S_compliant"
    assert resolved.resolved_by == "layer1_dual_channel"


def test_layer1_dual_channel_resolves_soft_resist_when_both_channels_agree():
    config = Config()
    midpoint_delta = (config.tau_high + config.tau_low) / 2
    resolved = layer1_dual_channel_check(
        r_token=config.tau_token_low + 0.05, delta_now=midpoint_delta, trajectory="sustained_down", config=config
    )
    assert resolved is not None
    assert resolved.state == "S_soft_resist"
    assert resolved.resolved_by == "layer1_dual_channel"


def test_layer1_dual_channel_returns_none_when_channels_disagree():
    config = Config()
    midpoint_delta = (config.tau_high + config.tau_low) / 2
    resolved = layer1_dual_channel_check(
        r_token=-config.tau_token_high - 0.05, delta_now=midpoint_delta, trajectory="sustained_down", config=config
    )
    assert resolved is None


def test_classify_state_uses_dual_channel_when_enabled():
    config = Config()
    client = _FixedReplyClient("S_compliant")  # should not be reached
    midpoint_delta = (config.tau_high + config.tau_low) / 2
    resolved = classify_state(
        "正常展开的回复",
        delta=midpoint_delta,
        config=config,
        classifier_client=client,
        r_token=-config.tau_token_high - 0.05,
        trajectory="sustained_up",
    )
    assert resolved.state == "S_compliant"
    assert resolved.resolved_by == "layer1_dual_channel"
    assert client.calls == 0


def test_classify_state_disable_dual_channel_ignores_r_token_and_trajectory():
    config = Config()
    client = _FixedReplyClient("S_neutral")
    midpoint_delta = (config.tau_high + config.tau_low) / 2
    resolved = classify_state(
        "正常展开的回复",
        delta=midpoint_delta,
        config=config,
        classifier_client=client,
        r_token=-config.tau_token_high - 0.05,
        trajectory="sustained_up",
        disable_dual_channel=True,
    )
    # dual-channel signal would have resolved this as S_compliant; with the
    # ablation on, only the single-point geometry check runs, so it stays
    # ambiguous and escalates to Layer2.
    assert resolved.resolved_by == "layer2_llm"
    assert client.calls == 1


# --- v3 6.5节：Layer2附加输出的抵抗表现自由文本 ---


def test_layer2_extracts_resist_archetype_free_text_for_soft_resist():
    client = _FixedReplyClient("S_soft_resist\n免责声明式回避")
    resolved = layer2_llm_classify("some response text", client)
    assert resolved.state == "S_soft_resist"
    assert resolved.resist_archetype_raw == "免责声明式回避"


def test_layer2_leaves_resist_archetype_none_for_non_resist_states():
    client = _FixedReplyClient("S_engaged\n无")
    resolved = layer2_llm_classify("some response text", client)
    assert resolved.state == "S_engaged"
    assert resolved.resist_archetype_raw is None


def test_layer2_handles_soft_resist_without_second_line():
    client = _FixedReplyClient("S_soft_resist")
    resolved = layer2_llm_classify("some response text", client)
    assert resolved.state == "S_soft_resist"
    assert resolved.resist_archetype_raw is None
