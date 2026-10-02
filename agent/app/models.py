"""Model factory for every role, on free API keys or local Ollama.

Model strings:
  gemini-3.1-flash-lite            native Gemini through the AI Studio key (google-genai)
  mistral/mistral-small-latest     LiteLLM (any provider/model string LiteLLM accepts)
  groq/openai/gpt-oss-120b
  openrouter/qwen/qwen3.8-27b:free
  ollama_chat/qwen3:8b             local Ollama (ADK docs: use ollama_chat, not ollama)

Two wrappers decide what happens on provider errors:
  PinnedLlm    measured runs. Always the same model; on a rate limit or overload it
               waits (honouring the provider's retry delay) and retries. A daily quota
               raises QuotaExhausted so the run stops and can be resumed later.
  FallbackLlm  local demo only (CRMROUTE_FALLBACK=1). On any provider error it moves
               to the next model: Gemini -> Mistral -> Groq -> OpenRouter -> Ollama.
Both tag every response with the model that answered (custom_metadata.answered_by).
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from collections.abc import AsyncGenerator

from google.adk.models import BaseLlm, Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

from .config import (
    FALLBACK_CHAIN,
    MAX_RETRY_WAIT_S,
    OLLAMA_API_BASE,
    OLLAMA_NUM_CTX,
    OLLAMA_THINK,
    THINKING_LEVEL,
)

logger = logging.getLogger("crmroute.models")


class QuotaExhausted(RuntimeError):
    """A daily quota is used up; waiting minutes will not help. Resume the run later."""


def provider_of(model: str) -> str:
    if "/" not in model:
        return "gemini"
    return model.split("/", 1)[0]


def is_gemini3(model: str) -> bool:
    return re.search(r"gemini-3", model) is not None


def generation_config(model: str, max_tokens: int) -> types.GenerateContentConfig:
    """Same rule as the benchmark patch: Gemini 3 at temperature 1.0 (Google's advice)
    with a fixed thinking level; every other model at temperature 0.0."""
    if is_gemini3(model):
        cfg = types.GenerateContentConfig(temperature=1.0, max_output_tokens=max_tokens)
        if provider_of(model) == "gemini":
            cfg.thinking_config = types.ThinkingConfig(thinking_level=types.ThinkingLevel(THINKING_LEVEL.upper()))
        return cfg
    return types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_tokens)


def _raw_model(model: str) -> BaseLlm:
    if provider_of(model) == "gemini":
        # retries are handled by the wrappers below so every wait is logged
        return Gemini(model=model, retry_options=types.HttpRetryOptions(attempts=1))
    extra: dict = {"num_retries": 0}
    if model.startswith("ollama"):
        extra.update(api_base=OLLAMA_API_BASE, num_ctx=OLLAMA_NUM_CTX)
        if not OLLAMA_THINK:
            extra["reasoning_effort"] = "none"  # LiteLLM maps this to Ollama think=false
    if is_gemini3(model):  # gemini/... through LiteLLM
        extra["reasoning_effort"] = THINKING_LEVEL
    return LiteLlm(model=model, **extra)


# ---- error handling ------------------------------------------------------------------

def classify(err: Exception) -> str:
    """too_large | daily | rate | unavailable | other"""
    text = f"{type(err).__name__} {err}"
    low = text.lower()
    status = getattr(err, "code", None) or getattr(err, "status_code", None)
    if "request too large" in low or "reduce your message size" in low or "context length" in low             or "maximum context" in low or "context_length_exceeded" in low:
        return "too_large"  # never succeeds on this model, whatever the wait
    if "perday" in low.replace(" ", "") or "per day" in low or "daily" in low or "requests_per_day" in low:
        return "daily"
    if status == 429 or "429" in text or "resource_exhausted" in low or "ratelimit" in low.replace(" ", "") \
            or "rate limit" in low or "rate_limited" in low:
        return "rate"
    if status in (500, 502, 503, 504) or any(s in low for s in (
            "503", "unavailable", "overloaded", "high demand", "serviceunavailable", "apiconnectionerror",
            "timeout", "internalservererror", "connection refused",
            # transient network failures (DNS, dropped connections) on the way to the same model
            "connecterror", "name resolution", "getaddrinfo", "connection reset", "remoteprotocolerror",
            "server disconnected", "cannot connect to host",
            # the same failures while refreshing a Vertex access token (google-auth over requests)
            "transporterror", "nameresolutionerror", "failed to resolve", "name or service not known")):
        return "unavailable"
    return "other"


def retry_after_seconds(err: Exception) -> float | None:
    text = str(err)
    for pattern in (r"retry in ([\d.]+)\s*s", r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", r"retry-after['\"]?\s*[:=]\s*['\"]?([\d.]+)"):
        m = re.search(pattern, text, re.I)
        if m:
            return float(m.group(1))
    return None


def _prepare(llm_request: LlmRequest, model: str) -> LlmRequest:
    """Retarget a copy of the request at `model`. For LiteLLM providers, drop thought
    parts from the history: ADK replays them as `reasoning_content`, which Groq rejects.
    Native Gemini keeps them (thought signatures are needed for tool calls)."""
    request = llm_request.model_copy(update={"model": model})
    if provider_of(model) != "gemini":
        contents = []
        for content in request.contents:
            parts = [p for p in (content.parts or []) if not p.thought]
            if parts:
                contents.append(content.model_copy(update={"parts": parts}))
        request.contents = contents
    return request


def _textify_foreign_calls(request: LlmRequest) -> LlmRequest:
    """Demo fallback only: Gemini 3 rejects history function calls that carry no thought
    signature ("Function call is missing a thought_signature"), which happens after
    another provider made a tool call in this turn. Rewrite such calls and their results
    as plain text so Gemini can continue the conversation."""
    foreign_ids: set = set()
    for content in request.contents:
        for p in content.parts or []:
            if p.function_call and not p.thought_signature:
                foreign_ids.add(p.function_call.id or p.function_call.name)
    if not foreign_ids:
        return request
    contents = []
    for content in request.contents:
        parts = []
        for p in content.parts or []:
            call, resp = p.function_call, p.function_response
            if call and (call.id or call.name) in foreign_ids:
                parts.append(types.Part.from_text(text=f"[earlier tool call] {call.name}({json.dumps(call.args or {})[:600]})"))
            elif resp and (resp.id or resp.name) in foreign_ids:
                parts.append(types.Part.from_text(text=f"[result of {resp.name}] {json.dumps(resp.response or {}, default=str)[:4000]}"))
            else:
                parts.append(p)
        contents.append(content.model_copy(update={"parts": parts}))
    request.contents = contents
    return request


def _json_object(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _repair_text_tool_call(resp: LlmResponse, llm_request: LlmRequest) -> LlmResponse:
    """Small local models sometimes write a tool call as text ("set_model_response {...}"
    or {"name": ..., "arguments": {...}}) instead of calling it. Turn such text into
    the function call it spells, for tools that exist in this request only."""
    content = resp.content
    if not content or not content.parts or any(p.function_call for p in content.parts):
        return resp
    text = "".join(p.text or "" for p in content.parts if not p.thought).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    tools = set(getattr(llm_request, "tools_dict", {}) or {})
    if not text or not tools:
        return resp
    name, args = None, None
    head = re.match(r"^\W*([A-Za-z_][\w]*)\W", text)
    if head and head.group(1) in tools:
        name, args = head.group(1), _json_object(text[head.end() - 1:])
    else:
        blob = _json_object(text)
        if blob and blob.get("name") in tools and isinstance(blob.get("arguments") or blob.get("args"), dict):
            name, args = blob["name"], blob.get("arguments") or blob.get("args")
    repaired = name
    if (not name or args is None) and "set_model_response" in tools:
        # The final answer written as text instead of the tool call ADK expects: a bare
        # SolverOutput object, or plain words. Plain words become the answer (the checker
        # still validates its form) instead of an unhandled ValidationError.
        blob = _json_object(text)
        if blob and "answer" in blob:
            name, args, repaired = "set_model_response", {"kind": "answer", **blob}, "set_model_response(json text)"
        elif not text.lstrip().startswith(("{", "[")):
            name, args, repaired = "set_model_response", {"kind": "answer", "answer": text, "known_details": []}, "set_model_response(plain text)"
    if not name or args is None:
        return resp
    resp.content = types.Content(role="model", parts=[types.Part.from_function_call(name=name, args=args)])
    resp.custom_metadata = {**(resp.custom_metadata or {}), "repaired_tool_call": repaired}
    return resp


def _uniform_capabilities(model: BaseLlm):
    """Structured output always goes through ADK's set_model_response tool. Otherwise
    LiteLLM models get a JSON response_format next to the tools, which Groq rejects
    ("json mode cannot be combined with tool/function calling"); AI Studio Gemini
    already takes the tool path."""
    return model.capabilities.model_copy(update={"output_schema_and_tools": False})


def _tag(resp: LlmResponse, model: str, **meta) -> LlmResponse:
    resp.custom_metadata = {**(resp.custom_metadata or {}), "answered_by": model, **meta}
    return resp


class PinnedLlm(BaseLlm):
    """One fixed model; wait and retry on rate limits and overloads, never switch."""

    inner: BaseLlm
    max_wait_s: float = MAX_RETRY_WAIT_S

    @property
    def capabilities(self):
        return _uniform_capabilities(self.inner)

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        request = _prepare(llm_request, self.inner.model)
        waited, attempt = 0.0, 0
        while True:
            try:
                responses = [r async for r in self.inner.generate_content_async(request, stream)]
            except Exception as err:  # noqa: BLE001 - classified below
                kind = classify(err)
                if kind == "daily":
                    raise QuotaExhausted(f"QUOTA_EXHAUSTED {self.inner.model}: {str(err)[:300]}") from err
                if kind not in ("rate", "unavailable") or waited >= self.max_wait_s:
                    raise
                delay = retry_after_seconds(err) or min(60.0, 4.0 * 2 ** attempt)
                delay = delay + random.uniform(0, 1.5)
                logger.warning("%s %s; waiting %.1fs (attempt %d, waited %.0fs)", self.inner.model, kind, delay, attempt + 1, waited)
                await asyncio.sleep(delay)
                waited += delay
                attempt += 1
                continue
            for r in responses:
                r = _repair_text_tool_call(r, llm_request)
                yield _tag(r, self.inner.model, attempts=attempt + 1, waited_s=round(waited, 1))
            return


class FallbackLlm(BaseLlm):
    """Demo only: try each model in order, skipping ones that recently failed."""

    chain: list[BaseLlm]
    cooldown_until: dict[str, float] = Field(default_factory=dict)

    @property
    def capabilities(self):
        return _uniform_capabilities(self.chain[0])

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        failures: list[str] = []
        now = time.time()
        for candidate in self.chain:
            if self.cooldown_until.get(candidate.model, 0) > now:
                failures.append(f"{candidate.model}: cooling down")
                continue
            request = _prepare(llm_request, candidate.model)
            if provider_of(candidate.model) == "gemini":
                request = _textify_foreign_calls(request)
            try:
                responses = [r async for r in candidate.generate_content_async(request, stream)]
            except Exception as err:  # noqa: BLE001 - any provider error moves to the next one
                kind = classify(err)
                self.cooldown_until[candidate.model] = now + {"daily": 3600, "rate": 60, "unavailable": 30}.get(kind, 10)
                failures.append(f"{candidate.model}: {kind}: {str(err)[:160]}")
                logger.warning("fallback: %s failed (%s), trying next", candidate.model, kind)
                continue
            for r in responses:
                r = _repair_text_tool_call(r, llm_request)
                yield _tag(r, candidate.model, fallback_from=failures)
            return
        raise RuntimeError("all providers failed: " + " | ".join(failures))


def build_model(model: str, fallback: bool = False) -> BaseLlm:
    if not fallback:
        return PinnedLlm(model=model, inner=_raw_model(model))
    order = [model] + [m for m in FALLBACK_CHAIN if m != model]
    return FallbackLlm(model=f"fallback:{model}", chain=[_raw_model(m) for m in order])
