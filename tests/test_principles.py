import pytest

from vdriftbench.llm_client import MockLLMClient
from vdriftbench.principles import (
    PRINCIPLES, DIMENSION_HINTS, _history_block, _MAX_DRAFT_HISTORY_CHARS,
    draft_utterance, render_template,
)
from vdriftbench.schema import RoundRecord, Sample

SAMPLE = Sample(
    sample_id="T0001",
    prompt="示例prompt",
    category="历史(高威胁)",
    category_macro="历史类",
    target_claim="示例target_claim",
    value_axis="历史真实性",
    risk_type="历史篡改",
    expected_safe_behavior="指出该说法缺乏依据",
)

ALL_TOKENS = ["P1", "P1变体", "P2", "P3", "P3+P4", "P4", "P4_continue",
              "P4_rank", "P4_best", "P4_value", "P4_evidence", "P6_strong",
              "P5", "P6", "P7_direct", "P7_gentle", "P7+P6"]


# --- v3: PRINCIPLES holds operation definitions, not templates ---


@pytest.mark.parametrize("token", ALL_TOKENS)
def test_principles_dict_has_nonempty_operation_definition(token):
    assert token in PRINCIPLES
    assert isinstance(PRINCIPLES[token], str)
    assert len(PRINCIPLES[token].strip()) > 0


def test_dimension_hints_cover_the_four_scored_dimensions():
    assert set(DIMENSION_HINTS) == {"VDS", "EFS", "NJS", "IFR"}
    for hint in DIMENSION_HINTS.values():
        assert isinstance(hint, str) and hint.strip()


# --- v3: draft_utterance drives one O-T-R-F(-S) call to the draft LLM ---


@pytest.mark.parametrize("token", ALL_TOKENS)
def test_draft_utterance_returns_nonempty_response_for_every_known_principle(token):
    draft_client = MockLLMClient(persona="draft")
    result = draft_utterance(token, SAMPLE, history=[], round_idx=1, draft_client=draft_client)
    assert isinstance(result.response, str)
    assert len(result.response.strip()) > 0
    assert result.fidelity_label == "match" or result.fidelity_label.startswith("drift_to:")


def test_draft_utterance_raises_on_unknown_principle():
    draft_client = MockLLMClient(persona="draft")
    with pytest.raises(ValueError):
        draft_utterance("P99", SAMPLE, history=[], round_idx=1, draft_client=draft_client)


def test_draft_utterance_populates_observation_and_thought():
    draft_client = MockLLMClient(persona="draft")
    result = draft_utterance("P2", SAMPLE, history=[], round_idx=2, draft_client=draft_client)
    assert result.observation
    assert result.thought


def test_draft_utterance_passes_dimension_hint_into_prompt():
    class _CapturingClient(MockLLMClient):
        def chat(self, messages, *, temperature=0.7):
            self.last_user_content = messages[-1]["content"]
            return super().chat(messages, temperature=temperature)

    client = _CapturingClient(persona="draft")
    draft_utterance(
        "P3", SAMPLE, history=[], round_idx=2, draft_client=client, dimension_hint="尝试引入一个证据锚点"
    )
    assert "尝试引入一个证据锚点" in client.last_user_content


def test_draft_utterance_passes_prev_observation_into_prompt():
    class _CapturingClient(MockLLMClient):
        def chat(self, messages, *, temperature=0.7):
            self.last_user_content = messages[-1]["content"]
            return super().chat(messages, temperature=temperature)

    client = _CapturingClient(persona="draft")
    draft_utterance(
        "P3", SAMPLE, history=[], round_idx=3, draft_client=client, prev_observation="之前观察到的假设文本"
    )
    assert "之前观察到的假设文本" in client.last_user_content


# --- legacy pure-function renderer, kept for backward compatibility only ---


@pytest.mark.parametrize("token", ALL_TOKENS)
def test_render_template_returns_nonempty_string_for_every_known_principle(token):
    text = render_template(token, SAMPLE, history=[], round_idx=1)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


def test_render_template_includes_target_claim_for_inducing_principles():
    text = render_template("P2", SAMPLE, history=[], round_idx=2)
    assert SAMPLE.target_claim in text


# --- v3.9: draft 历史块截断，防止 8192-token 上下文溢出 ---


def test_history_block_truncates_long_history_to_tail():
    long_response = "很长的目标模型回复内容。" * 200  # ~2200 chars
    history = [
        RoundRecord(
            round_idx=i, state_resolved=None, resolved_by=None, principle_used="P4_rhetoric",
            bandit_context=None, bandit_posterior_snapshot=None,
            prompt=f"第{i}轮提问", response=long_response, scores=None, embed_drift_norm=None,
        )
        for i in range(1, 6)
    ]
    text = _history_block(history)
    assert len(text) <= _MAX_DRAFT_HISTORY_CHARS + len("[...更早的轮次已省略，仅保留最近约3000字...]\n")
    assert "更早的轮次已省略" in text
    # 末尾（最近轮次）必须保留，开头（早期轮次）被截掉
    assert "第5轮提问" in text
    assert "第1轮提问" not in text


def test_history_block_does_not_truncate_short_history():
    history = [
        RoundRecord(
            round_idx=1, state_resolved=None, resolved_by=None, principle_used="P1",
            bandit_context=None, bandit_posterior_snapshot=None,
            prompt="提问", response="简短回复", scores=None, embed_drift_norm=None,
        )
    ]
    text = _history_block(history)
    assert "更早的轮次已省略" not in text
    assert "第1轮" in text


def test_render_template_raises_on_unknown_principle():
    with pytest.raises(ValueError):
        render_template("P99", SAMPLE, history=[], round_idx=1)
