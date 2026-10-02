from google.genai import types

from app import state_keys as K
from app import usage


def test_unknown_model_price_is_not_zero():
    assert usage.cost_usd("other/unpriced-model", 1000, 0, 100, 0) is None
    assert usage.cost_usd("ollama_chat/qwen3:8b", 1000, 0, 100, 0) == 0.0


def test_every_response_counted_even_without_usage(monkeypatch):
    logged = []
    monkeypatch.setattr(usage, "log_call", logged.append)
    state = {K.ORG: "b2b"}
    usage.add_usage(state, "gemini-3.8-flash", "solver_big", None,
                    {"session_id": "session-1", "attempts": 2, "waited_s": 4})
    totals = state[K.USAGE]
    assert totals["model_calls"] == 1
    assert totals["missing_usage_calls"] == 1
    assert totals["cost_complete"] is False
    assert totals["total_cost_usd"] == 0.0
    assert logged[0]["cost_usd"] is None
    assert logged[0]["session_id"] == "session-1"
    assert logged[0]["attempts"] == 2


def test_mixed_prices_keep_known_subtotal_and_flag_unknown(monkeypatch):
    monkeypatch.setattr(usage, "log_call", lambda record: None)
    state = {}
    tokens = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=1000, cached_content_token_count=200,
        candidates_token_count=100, thoughts_token_count=50,
    )
    usage.add_usage(state, "gemini-3.8-flash", "solver_big", tokens)
    assert state[K.USAGE]["cost_complete"] is True
    known_cost = usage.cost_usd("gemini-3.8-flash", 1000, 200, 100, 50)
    usage.add_usage(state, "other/unpriced-model", "solver_big", tokens)
    totals = state[K.USAGE]
    assert totals["total_cost_usd"] == known_cost
    assert totals["model_calls"] == 2
    assert totals["unpriced_model_calls"] == 1
    assert totals["cost_complete"] is False
    assert state[K.CALLS][-1]["cost_usd"] is None
    assert totals["component:solver_big"]["prompt"] == 2000
