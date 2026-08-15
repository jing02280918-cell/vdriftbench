import pytest

from vdriftbench.llm_client import OpenAICompatibleClient, build_llm_client


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "DEEPSEEK_API_KEY",
        "VDB_API_KEY",
        "VDB_BASE_URL",
        "VDB_TARGET_API_KEY",
        "VDB_TARGET_BASE_URL",
        "VDB_TARGET_DISABLE_THINKING",
        "VDB_DISABLE_THINKING",
    ):
        monkeypatch.delenv(key, raising=False)


def test_build_llm_client_mock_ignores_credentials():
    client = build_llm_client("deepseek-v4-flash", mock=True, persona="target")
    text = client.chat([{"role": "user", "content": "hi"}])
    assert isinstance(text, str) and text


def test_deepseek_model_falls_back_to_deepseek_api_key_and_official_endpoint(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-deepseek-env")

    client = build_llm_client("deepseek-v4-flash", mock=False, persona="target")

    assert isinstance(client, OpenAICompatibleClient)
    assert str(client._client.base_url).rstrip("/") == "https://api.deepseek.com"
    assert client._client.api_key == "sk-from-deepseek-env"


def test_persona_specific_env_takes_priority_over_deepseek_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-generic")
    monkeypatch.setenv("VDB_TARGET_API_KEY", "sk-persona-specific")
    monkeypatch.setenv("VDB_TARGET_BASE_URL", "https://my-proxy.example.com/v1")

    client = build_llm_client("deepseek-v4-flash", mock=False, persona="target")

    assert client._client.api_key == "sk-persona-specific"
    assert str(client._client.base_url).rstrip("/") == "https://my-proxy.example.com/v1"


def test_non_deepseek_model_does_not_pick_up_deepseek_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-not-be-used")

    client = build_llm_client("gpt-4o-mini", mock=False, persona="target")

    assert client._client.api_key != "sk-should-not-be-used"


def test_thinking_disabled_by_default_for_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")

    client = build_llm_client("deepseek-v4-flash", mock=False, persona="target")

    assert client.extra_body == {"thinking": {"type": "disabled"}}


def test_thinking_can_be_re_enabled_via_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setenv("VDB_TARGET_DISABLE_THINKING", "0")

    client = build_llm_client("deepseek-v4-flash", mock=False, persona="target")

    assert client.extra_body is None


def test_thinking_toggle_not_applied_to_non_deepseek_models(monkeypatch):
    client = build_llm_client("gpt-4o-mini", mock=False, persona="target")

    assert client.extra_body is None


def test_chat_with_logprobs_falls_back_to_plain_chat_when_unsupported(monkeypatch):
    client = OpenAICompatibleClient(model="whatever", api_key="sk-x", base_url="https://example.invalid")
    client.max_retries = 1

    def _boom(**kwargs):
        raise RuntimeError("this endpoint does not support logprobs")

    monkeypatch.setattr(client._client.chat.completions, "create", _boom)
    monkeypatch.setattr(client, "chat", lambda messages, temperature=0.7: "fallback reply")

    text, logprobs = client.chat_with_logprobs([{"role": "user", "content": "hi"}])

    assert text == "fallback reply"
    assert logprobs == []
