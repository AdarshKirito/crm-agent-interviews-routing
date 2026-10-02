from types import SimpleNamespace

import httpx
import pytest

from app import nodes, usage
from app import state_keys as K
from app.guard import prompt_guard


@pytest.mark.asyncio
async def test_prompt_guard_reports_each_chunk_without_request_content(monkeypatch):
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": "0.01"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }))
    monkeypatch.setattr(prompt_guard, "GROQ_API_KEY", "test-only")
    monkeypatch.setattr(prompt_guard.httpx, "AsyncClient", lambda **kwargs: client_class(transport=transport, **kwargs))
    recorded = []
    assert await prompt_guard.injection_score("x" * (prompt_guard.CHUNK_CHARS + 1), on_usage=recorded.append) == 0.01
    assert recorded == [{"prompt_tokens": 10, "completion_tokens": 1}] * 2


@pytest.mark.asyncio
async def test_failed_prompt_guard_attempt_reports_unknown_usage(monkeypatch):
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    monkeypatch.setattr(prompt_guard, "GROQ_API_KEY", "test-only")
    monkeypatch.setattr(prompt_guard.httpx, "AsyncClient", lambda **kwargs: client_class(transport=transport, **kwargs))
    recorded = []
    assert await prompt_guard.injection_score("hello", on_usage=recorded.append) is None
    assert recorded == [None]


@pytest.mark.asyncio
async def test_screen_includes_guard_usage_in_session_totals(monkeypatch):
    async def guard(text, **kwargs):
        kwargs["on_usage"]({"prompt_tokens": 10, "completion_tokens": 1})
        return 0.0

    monkeypatch.setattr(nodes, "injection_score", guard)
    monkeypatch.setattr(nodes, "get_router", lambda: SimpleNamespace(predict=lambda text: SimpleNamespace(
        task_type="knowledge_qa", confidence=1.0, top=[])))
    logged = []
    monkeypatch.setattr(usage, "log_call", logged.append)
    state = {K.AUDIENCE: "employee", K.CONVERSATION: [{"role": "user", "text": "private request contents"}]}
    events = [event async for event in nodes.screen(SimpleNamespace(state=state, session=SimpleNamespace(id="test-session")))]
    assert events
    assert state[K.USAGE]["model_calls"] == 1
    # Prompt Guard 2 86M has a Groq list price, so the guard call is part of cost per task
    assert state[K.USAGE]["unpriced_model_calls"] == 0
    assert state[K.USAGE]["cost_complete"] is True
    assert state[K.USAGE]["total_cost_usd"] == pytest.approx(11 * 0.04 / 1e6)
    assert logged[0]["component"] == "prompt_guard"
    assert logged[0]["model"] == "groq/meta-llama/llama-prompt-guard-2-86m"
    assert logged[0]["session_id"] == "test-session"
    assert "private request contents" not in str(logged)


@pytest.mark.asyncio
async def test_unpriced_guard_model_marks_cost_incomplete(monkeypatch):
    async def guard(text, **kwargs):
        kwargs["on_usage"]({"prompt_tokens": 10, "completion_tokens": 1})
        return 0.0

    monkeypatch.setattr(nodes, "injection_score", guard)
    monkeypatch.setattr(nodes, "PROMPT_GUARD_MODEL", "example/unpriced-guard")
    monkeypatch.setattr(nodes, "get_router", lambda: SimpleNamespace(predict=lambda text: SimpleNamespace(
        task_type="knowledge_qa", confidence=1.0, top=[])))
    monkeypatch.setattr(usage, "log_call", lambda record: None)
    state = {K.AUDIENCE: "employee", K.CONVERSATION: [{"role": "user", "text": "hello"}]}
    [event async for event in nodes.screen(SimpleNamespace(state=state, session=SimpleNamespace(id="s")))]
    assert state[K.USAGE]["unpriced_model_calls"] == 1
    assert state[K.USAGE]["cost_complete"] is False
