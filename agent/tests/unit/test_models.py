import asyncio
from collections.abc import AsyncGenerator

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from app import models
from app.models import FallbackLlm, PinnedLlm, QuotaExhausted, classify, generation_config, retry_after_seconds


class FlakyLlm(BaseLlm):
    """Raises the given errors in order, then answers; records the model it was asked for."""

    errors: list = []
    seen_models: list = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        self.seen_models.append(llm_request.model)
        if self.errors:
            raise self.errors.pop(0)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text="ok")]))


# error texts as returned by the providers during the 2026-09-29 probe
MISTRAL_429 = RuntimeError('429 {"message":"Rate limit exceeded","type":"rate_limited","code":"1300"}')
GEMINI_503 = RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
GEMINI_DAILY = RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
GEMINI_MINUTE = RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 7.5s.")
GROQ_TOO_LARGE = RuntimeError("413 Request too large for model on tokens per minute (TPM): Limit 8000, Requested 9500")


def test_classify_provider_errors():
    assert classify(MISTRAL_429) == "rate"
    assert classify(GEMINI_503) == "unavailable"
    assert classify(GEMINI_DAILY) == "daily"
    assert classify(GEMINI_MINUTE) == "rate"
    assert classify(GROQ_TOO_LARGE) == "too_large"
    assert classify(RuntimeError("RateLimitError: Request too large for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
                                 "Limit 8000, Requested 11722, please reduce your message size")) == "too_large"
    assert classify(ValueError("bad schema")) == "other"


def test_retry_after_parsing():
    assert retry_after_seconds(GEMINI_MINUTE) == 7.5
    assert retry_after_seconds(RuntimeError("{'retryDelay': '35s'}")) == 35.0
    assert retry_after_seconds(MISTRAL_429) is None


def run(llm: BaseLlm) -> list[LlmResponse]:
    async def go():
        return [r async for r in llm.generate_content_async(LlmRequest(model="ignored"), stream=False)]
    return asyncio.run(go())


def test_pinned_waits_and_retries_same_model(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(models.asyncio, "sleep", fake_sleep)
    inner = FlakyLlm(model="mistral/mistral-small-latest", errors=[MISTRAL_429, GEMINI_503], seen_models=[])
    out = run(PinnedLlm(model="mistral/mistral-small-latest", inner=inner))
    assert len(sleeps) == 2
    assert inner.seen_models == ["mistral/mistral-small-latest"] * 3  # never switched
    assert out[0].custom_metadata["answered_by"] == "mistral/mistral-small-latest"
    assert out[0].custom_metadata["attempts"] == 3


def test_pinned_stops_on_daily_quota(monkeypatch):
    monkeypatch.setattr(models.asyncio, "sleep", lambda s: asyncio.sleep(0))
    inner = FlakyLlm(model="gemini-3.1-flash-lite", errors=[GEMINI_DAILY], seen_models=[])
    with pytest.raises(QuotaExhausted, match="QUOTA_EXHAUSTED"):
        run(PinnedLlm(model="gemini-3.1-flash-lite", inner=inner))


def test_pinned_does_not_retry_other_errors():
    inner = FlakyLlm(model="groq/openai/gpt-oss-120b", errors=[GROQ_TOO_LARGE], seen_models=[])
    with pytest.raises(RuntimeError, match="413"):
        run(PinnedLlm(model="groq/openai/gpt-oss-120b", inner=inner))


def test_fallback_moves_on_and_logs_answering_model():
    gemini = FlakyLlm(model="gemini-3.8-flash", errors=[GEMINI_503], seen_models=[])
    mistral = FlakyLlm(model="mistral/mistral-small-latest", errors=[MISTRAL_429], seen_models=[])
    ollama = FlakyLlm(model="ollama_chat/qwen3:8b", errors=[], seen_models=[])
    fb = FallbackLlm(model="fallback:gemini-3.8-flash", chain=[gemini, mistral, ollama])
    out = run(fb)
    meta = out[0].custom_metadata
    assert meta["answered_by"] == "ollama_chat/qwen3:8b"
    assert [f.split(":")[0] for f in meta["fallback_from"]] == ["gemini-3.8-flash", "mistral/mistral-small-latest"]
    assert ollama.seen_models == ["ollama_chat/qwen3:8b"]  # request re-targeted to the answering model
    # failed providers cool down: the next call goes straight to the one that worked
    out2 = run(fb)
    assert out2[0].custom_metadata["answered_by"] == "ollama_chat/qwen3:8b"
    assert gemini.seen_models == ["gemini-3.8-flash"]


def test_generation_config_per_model_family():
    g = generation_config("gemini-3.1-flash-lite", 1000)
    assert g.temperature == 1.0 and g.thinking_config.thinking_level == types.ThinkingLevel.LOW
    m = generation_config("mistral/mistral-small-latest", 1000)
    assert m.temperature == 0.0 and m.thinking_config is None


def test_wrappers_force_tool_path_for_structured_output():
    from google.adk.models.lite_llm import LiteLlm
    groq = LiteLlm(model="groq/openai/gpt-oss-120b")
    assert PinnedLlm(model=groq.model, inner=groq).capabilities.output_schema_and_tools is False
    assert FallbackLlm(model="fallback", chain=[groq]).capabilities.output_schema_and_tools is False


def test_thought_parts_dropped_for_litellm_providers_only():
    from app.models import _prepare
    history = [types.Content(role="model", parts=[types.Part(text="plan...", thought=True), types.Part(text="answer")])]
    req = LlmRequest(model="x", contents=history)
    groq = _prepare(req, "groq/openai/gpt-oss-120b")
    assert [p.text for p in groq.contents[0].parts] == ["answer"]
    gemini = _prepare(req, "gemini-3.1-flash-lite")
    assert len(gemini.contents[0].parts) == 2
    assert len(req.contents[0].parts) == 2  # the session history itself is untouched


def _text_response(text):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))


def test_text_tool_calls_from_small_models_are_repaired():
    from app.models import _repair_text_tool_call
    req = LlmRequest(model="ollama_chat/qwen3:8b")
    req.tools_dict = {"set_model_response": object(), "soql_query": object()}
    # the exact shape qwen3:8b produced on 2026-09-29
    r = _repair_text_tool_call(_text_response('set_model_response\n{"kind": "answer", "answer": "TX", "known_details": []}'), req)
    call = r.content.parts[0].function_call
    assert call.name == "set_model_response" and call.args["answer"] == "TX"
    assert r.custom_metadata["repaired_tool_call"] == "set_model_response"
    r = _repair_text_tool_call(_text_response('```json\n{"name": "soql_query", "arguments": {"query": "SELECT Id FROM Case"}}\n```'), req)
    assert r.content.parts[0].function_call.name == "soql_query"
    # a final answer written as plain words becomes the set_model_response call (no crash)
    r = _repair_text_tool_call(_text_response("The answer is TX."), req)
    assert r.content.parts[0].function_call.args == {"kind": "answer", "answer": "The answer is TX.", "known_details": []}
    assert r.custom_metadata["repaired_tool_call"] == "set_model_response(plain text)"
    r = _repair_text_tool_call(_text_response('{"answer": "TX", "known_details": []}'), req)
    assert r.content.parts[0].function_call.args["kind"] == "answer"
    # without set_model_response in the request, plain text is left alone
    plain = LlmRequest(model="m")
    plain.tools_dict = {"soql_query": object()}
    assert _repair_text_tool_call(_text_response("The answer is TX."), plain).content.parts[0].text == "The answer is TX."
    # a tool that is not in the request is never called
    r = _repair_text_tool_call(_text_response('delete_records\n{"id": 1}'), req)
    assert r.content.parts[0].function_call.name == "set_model_response"


def test_foreign_tool_calls_become_text_before_falling_back_to_gemini():
    from app.models import _textify_foreign_calls
    groq_call = types.Part.from_function_call(name="soql_query", args={"query": "SELECT COUNT() FROM Case"})
    groq_call.function_call.id = "c1"
    result = types.Part.from_function_response(name="soql_query", response={"totalSize": 42})
    result.function_response.id = "c1"
    gemini_call = types.Part(function_call=types.FunctionCall(name="get_record", args={}, id="c2"), thought_signature=b"sig")
    req = LlmRequest(model="gemini-3.1-flash-lite", contents=[
        types.Content(role="user", parts=[types.Part.from_text(text="How many cases?")]),
        types.Content(role="model", parts=[groq_call]),
        types.Content(role="user", parts=[result]),
        types.Content(role="model", parts=[gemini_call]),
    ])
    out = _textify_foreign_calls(req)
    assert out.contents[1].parts[0].text.startswith("[earlier tool call] soql_query(")
    assert out.contents[2].parts[0].text.startswith("[result of soql_query]") and "42" in out.contents[2].parts[0].text
    assert out.contents[3].parts[0].function_call.name == "get_record"  # Gemini's own signed call is kept
