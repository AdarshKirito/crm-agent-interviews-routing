"""Unit tests for the benchmark patch's rate-limit handling (no network).

  vendor/CRMArena/.venv/Scripts/python -m pytest tests/test_bench_retry.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "CRMArena"))
from crm_sandbox.agents import utils  # noqa: E402


def fake_litellm(monkeypatch, outcomes):
    """outcomes: exceptions to raise in order, then a successful response."""
    import litellm

    calls = []

    def completion(**kwargs):
        calls.append(kwargs["model"])
        if outcomes:
            raise outcomes.pop(0)
        return SimpleNamespace(model=kwargs["model"], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(utils._time, "sleep", lambda s: None)
    utils.CALLS.clear()
    return calls


def test_waits_on_rate_limit_and_keeps_the_same_model(monkeypatch):
    calls = fake_litellm(monkeypatch, [RuntimeError("429 Rate limit exceeded"), RuntimeError("503 UNAVAILABLE high demand")])
    res = utils.completion_with_retry("judge", model="mistral/mistral-medium-latest", messages=[])
    assert res.model == "mistral/mistral-medium-latest"
    assert calls == ["mistral/mistral-medium-latest"] * 3
    assert utils.CALLS[-1]["attempts"] == 3 and utils.CALLS[-1]["role"] == "judge"
    assert utils.CALLS[-1]["answered_by"] == "mistral/mistral-medium-latest"


def test_daily_quota_stops_the_run(monkeypatch):
    fake_litellm(monkeypatch, [RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    with pytest.raises(utils.QuotaExhausted):
        utils.completion_with_retry("agent", model="gemini/gemini-3.1-flash-lite", messages=[])


def test_oversized_request_is_not_retried(monkeypatch):
    calls = fake_litellm(monkeypatch, [RuntimeError("RateLimitError: Request too large ... please reduce your message size")])
    with pytest.raises(utils.RequestTooLarge):
        utils.completion_with_retry("agent", model="groq/openai/gpt-oss-120b", messages=[])
    assert len(calls) == 1


def test_retry_wait_respects_configured_budget(monkeypatch):
    fake_litellm(monkeypatch, [RuntimeError("503 UNAVAILABLE"), RuntimeError("503 UNAVAILABLE")])
    monkeypatch.setenv("CRMARENA_MAX_RETRY_WAIT_S", "1")
    waits = []
    monkeypatch.setattr(utils._time, "sleep", waits.append)
    with pytest.raises(RuntimeError, match="UNAVAILABLE"):
        utils.completion_with_retry("agent", model="fixed/model", messages=[])
    assert sum(waits) == 1


def test_missing_remote_price_is_unknown_but_local_api_charge_is_zero(monkeypatch):
    fake_litellm(monkeypatch, [])
    utils.completion_with_retry("agent", model="unknown/provider-model", messages=[])
    assert utils.CALLS[-1]["cost_usd"] is None
    assert utils.CALLS[-1]["cost_complete"] is False
    utils.completion_with_retry("judge", model="ollama_chat/local-model", messages=[])
    assert utils.CALLS[-1]["cost_usd"] == 0
    assert utils.CALLS[-1]["cost_complete"] is True
