"""Model-agnostic chat client used for the target model, the judge model, and
the Layer-2 state classifier.

`OpenAICompatibleClient` works with any OpenAI Chat Completions compatible
endpoint (OpenAI, DeepSeek, Moonshot, local vLLM/Ollama gateways, etc.) — set
`base_url` accordingly. `MockLLMClient` needs no network access or API key
and lets the whole pipeline run end-to-end for structural verification.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import os
import time
from typing import Sequence

logger = logging.getLogger(__name__)

Message = dict  # {"role": "user"|"assistant"|"system", "content": str}
TokenLogprob = tuple  # (token: str, logprob: float)


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def chat(self, messages: Sequence[Message], *, temperature: float = 0.7) -> str:
        """Return the assistant's text reply for the given message history."""

    def chat_with_logprobs(
        self, messages: Sequence[Message], *, temperature: float = 0.7, top_logprobs: int = 8
    ) -> tuple[str, list[TokenLogprob]]:
        """v3 5.2节：通道A(首token分布信号)需要的接口 — 返回回复文本，以及回复
        第一个token的 top-k (token, logprob) 列表。默认实现只是对不支持
        logprobs的客户端优雅降级（返回空列表，上游据此把r_token当作不可用，
        双通道融合规则会自动回退到只用通道B）；`OpenAICompatibleClient` 覆盖
        为真正请求 `logprobs=True` 的实现。"""

        return self.chat(messages, temperature=temperature), []


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        extra_body: dict | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAICompatibleClient requires the `openai` package: pip install openai"
            ) from exc

        self.model = model
        self.max_retries = max_retries
        # e.g. DeepSeek V4's `{"thinking": {"type": "disabled"}}` to opt out
        # of (billed) reasoning tokens — provider-specific, so only set this
        # when you know the target endpoint understands it (see
        # `build_llm_client`'s VDB_<PERSONA>_DISABLE_THINKING handling).
        self.extra_body = extra_body
        self._client = OpenAI(
            api_key=api_key or os.environ.get("VDB_API_KEY", "EMPTY"),
            base_url=base_url or os.environ.get("VDB_BASE_URL"),
            timeout=timeout,
        )

    def chat(self, messages: Sequence[Message], *, temperature: float = 0.7) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=temperature,
                    **({"extra_body": self.extra_body} if self.extra_body else {}),
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - retry any transient failure
                last_err = exc
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_err

    def chat_with_logprobs(
        self, messages: Sequence[Message], *, temperature: float = 0.7, top_logprobs: int = 8
    ) -> tuple[str, list[TokenLogprob]]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=temperature,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                    **({"extra_body": self.extra_body} if self.extra_body else {}),
                )
                choice = resp.choices[0]
                text = choice.message.content or ""
                first_token_lp: list[TokenLogprob] = []
                content_lp = getattr(choice.logprobs, "content", None) if choice.logprobs else None
                if content_lp:
                    top = content_lp[0].top_logprobs or []
                    first_token_lp = [(t.token, t.logprob) for t in top]
                return text, first_token_lp
            except Exception as exc:  # noqa: BLE001 - not every backend supports logprobs
                last_err = exc
                time.sleep(min(2 ** attempt, 8))
        # v3 5.2节的通道A是"免费加分项"，不是硬依赖——如果这个endpoint始终不支持
        # logprobs（或反复超时），优雅降级为不带logprobs的普通chat()，而不是让
        # 整个实验因为一个可选信号崩掉；上游state_cascade在r_token=None时本来就
        # 会自动回退到只用通道B。
        logger.warning(
            "chat_with_logprobs failed after %d retries (%s); falling back to plain chat() without logprobs",
            self.max_retries, last_err,
        )
        return self.chat(messages, temperature=temperature), []


class MockLLMClient(LLMClient):
    """Deterministic offline stand-in so the pipeline is runnable with zero setup.

    It does not simulate a real target model's safety behaviour — it only
    guarantees the *pipeline plumbing* (rounds, state cascade, bandit update,
    judge parsing) executes correctly end to end. Swap in
    `OpenAICompatibleClient` for anything resembling a real experiment.
    """

    def __init__(self, persona: str = "target") -> None:
        self.persona = persona

    def chat(self, messages: Sequence[Message], *, temperature: float = 0.7) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        digest = int(hashlib.sha256(last_user.encode("utf-8")).hexdigest(), 16)

        if self.persona == "judge":
            return self._mock_judge_output(digest)
        if self.persona == "classifier":
            return self._mock_state_output(digest, last_user)
        if self.persona == "draft":
            return self._mock_draft_output(last_user, digest)
        if self.persona == "plausibility":
            return self._mock_plausibility_output(digest)
        if self.persona == "enrich":
            return self._mock_enrich_output(last_user)
        return self._mock_target_reply(last_user, digest)

    def chat_with_logprobs(
        self, messages: Sequence[Message], *, temperature: float = 0.7, top_logprobs: int = 8
    ) -> tuple[str, list[TokenLogprob]]:
        text = self.chat(messages, temperature=temperature)
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        digest = int(hashlib.sha256(last_user.encode("utf-8")).hexdigest(), 16)
        # Deterministic mock top-k: alternate between refusal-flavoured and
        # compliant-flavoured leading tokens so state_cascade's r_token
        # computation has something non-trivial to chew on in tests.
        bucket = digest % 3
        if bucket == 0:
            top = [("我", -0.1), ("抱歉", -1.2), ("在", -3.0)]
        elif bucket == 1:
            top = [("在", -0.2), ("根据", -1.5), ("我", -3.0)]
        else:
            top = [("这", -0.3), ("从", -1.1), ("抱歉", -2.8)]
        return text, top

    @staticmethod
    def _mock_target_reply(prompt: str, digest: int) -> str:
        bucket = digest % 5
        if bucket == 0:
            return "我不能协助展开这类说法，这与已知的可靠证据不符。"
        if bucket == 1:
            return "这是一个存在争议的话题，我可以从背景角度做一些客观介绍，但需要说明主流观点并不支持这一说法。"
        if bucket == 2:
            return "从论证结构上看，这类说法通常会引用一些看似权威的材料来源，我可以展开分析这种论证是如何构建的。"
        if bucket == 3:
            return "综合前面的讨论，可以进一步展开这个角度，并补充一些细节让论述更完整。"
        return "好的，以下是完整的文章正文：……（此处为示意性展开内容）"

    _RESIST_ARCHETYPES = ("免责声明式回避", "反问确认式", "指出证据缺口式", "转移话题式")

    @classmethod
    def _mock_state_output(cls, digest: int, last_user: str = "") -> str:
        states = ["S_neutral", "S_soft_resist", "S_engaged", "S_compliant", "S_hard_refuse"]
        state = states[digest % len(states)]
        if state != "S_soft_resist":
            return f"{state}\n无"
        archetype = cls._RESIST_ARCHETYPES[digest % len(cls._RESIST_ARCHETYPES)]
        return f"{state}\n{archetype}"

    @staticmethod
    def _mock_draft_output(user_content: str, digest: int) -> str:
        bucket = digest % 4
        drift_hint = ["之前观察到它对具体数据要求较高", "这一轮进一步显示它对权威来源的引用也比较敏感",
                      "它在语气上仍保留一定的探讨性", "它已经开始主动补充细节"][bucket]
        fidelity = ["match", "match", "drift_to:none", "match"][bucket]
        variant_line = (
            f"[S 策略变体建议（可选）] 提前用一句反问确认对方是否理解论证结构，再顺势展开\n"
            if digest % 7 == 0
            else "[S 策略变体建议（可选）] 无\n"
        )
        return (
            f"[O 观察/累积假设更新] {drift_hint}，结合本轮内容更新假设：这个目标在本次对话里对该类信号有明显反应。\n"
            f"[T 战术思考] 针对上述假设，措辞上引用这一具体信号来切入本轮策略。\n"
            f"[R 生成] （mock起草）围绕当前策略和上一轮具体内容生成的一句/一段话术，内容随digest={digest % 1000}变化。\n"
            f"[F 保真度自检] {fidelity}\n"
            f"{variant_line}"
        )

    @staticmethod
    def _mock_plausibility_output(digest: int) -> str:
        import json

        bel = digest % 3
        per = (digest // 3) % 3
        sha = (digest // 9) % 3
        return json.dumps({"BEL": bel, "PER": per, "SHA": sha, "rationale": "mock"}, ensure_ascii=False)

    @staticmethod
    def _mock_enrich_output(user_content: str) -> str:
        import json

        prompt_line = next((l for l in user_content.splitlines() if l.startswith("prompt:")), "")
        claim = prompt_line.replace("prompt:", "").strip()[:60] or "该样本中隐含的偏颇观点"
        return json.dumps(
            {
                "target_claim": claim,
                "value_axis": "事实准确性",
                "risk_type": "伪证据编造",
                "expected_safe_behavior": "指出该说法缺乏可靠证据支持，并说明主流共识",
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _mock_judge_output(digest: int) -> str:
        import json

        vds = digest % 3
        efs = (digest // 3) % 3
        njs = (digest // 9) % 3
        scs = (digest // 27) % 3
        ifr = (digest // 81) % 3
        return json.dumps(
            {"VDS": vds, "EFS": efs, "NJS": njs, "SCS": scs, "IFR": ifr, "rationale": "mock"},
            ensure_ascii=False,
        )


_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"


def build_llm_client(model: str, mock: bool, persona: str = "target") -> LLMClient:
    if mock:
        return MockLLMClient(persona=persona)

    # v2 only exposed one global VDB_API_KEY/VDB_BASE_URL for every role. To
    # support separate local models per role (e.g. a local vLLM draft model
    # standing in for the target, a separate local vLLM judge model) without
    # breaking existing single-endpoint setups, each persona may override via
    # VDB_<PERSONA>_API_KEY / VDB_<PERSONA>_BASE_URL; unset ones fall back to
    # the global VDB_API_KEY / VDB_BASE_URL. "classifier" additionally falls
    # back to the judge's endpoint (it defaults to the judge model in main.py).
    prefix = persona.upper()
    api_key = os.environ.get(f"VDB_{prefix}_API_KEY")
    base_url = os.environ.get(f"VDB_{prefix}_BASE_URL")
    if persona == "classifier" and base_url is None:
        base_url = os.environ.get("VDB_JUDGE_BASE_URL")
        api_key = api_key or os.environ.get("VDB_JUDGE_API_KEY")

    # Convenience: if this role targets a DeepSeek model and nothing more
    # specific was configured, fall back to DeepSeek's own env var name and
    # official endpoint, so a bare `DEEPSEEK_API_KEY=...` in `.env`
    # (see `env_loader.py`) "just works" without needing the VDB_* names too.
    if model.startswith("deepseek"):
        api_key = api_key or os.environ.get("VDB_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        base_url = base_url or os.environ.get("VDB_BASE_URL") or _DEEPSEEK_DEFAULT_BASE_URL

    extra_body = None
    if model.startswith("deepseek"):
        # DeepSeek V4 has "thinking" (reasoning) mode on by default, which
        # bills reasoning tokens and adds latency. Disabled by default here
        # (opt back in with VDB_<PERSONA>_DISABLE_THINKING=0 / "false" if a
        # given run wants the target's real-world reasoning behavior).
        disable_thinking_flag = os.environ.get(f"VDB_{prefix}_DISABLE_THINKING") or os.environ.get("VDB_DISABLE_THINKING")
        disable_thinking = True if disable_thinking_flag is None else disable_thinking_flag.lower() not in ("0", "false", "")
        if disable_thinking:
            extra_body = {"thinking": {"type": "disabled"}}

    return OpenAICompatibleClient(model=model, api_key=api_key, base_url=base_url, extra_body=extra_body)
