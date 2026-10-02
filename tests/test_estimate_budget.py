"""Budget planning uses saved measurements, never credentials or model requests."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import estimate_budget as budget


def checkpoint(tmp_path, rows):
    path = tmp_path / "results_fixture.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_agent_checkpoint_input_counts_thoughts_and_excludes_policy(tmp_path):
    checkpoint(tmp_path, [{"agent_info": {"calls": [
        {"model": "gemini-3.1-flash-lite", "component": "solver_big", "prompt": 100, "cached": 20, "output": 5, "thoughts": 10},
        {"model": "groq/example", "component": "policy_check", "prompt": 50, "output": 10},
    ]}}, {"error": True}, {"interactive": True}])
    stats = budget.measured_agent([str(tmp_path)])
    assert stats["n"] == 1 and stats["calls"] == 1
    assert stats["prompt"] == 100 and stats["output"] == 15 and stats["cached"] == 20


def test_old_smoke_jsonl_is_still_supported(tmp_path):
    path = tmp_path / "smoke.jsonl"
    path.write_text(json.dumps({"model": "example", "model_calls": 2, "prompt_tokens": 100,
                                "output_tokens": 10, "seconds": 2}) + "\n")
    assert budget.measured_agent([str(path)])["tokens"] == 110


def test_raw_call_log_has_no_task_denominator_and_is_rejected(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps({"component": "solver_big", "prompt": 10, "output": 1}) + "\n")
    with pytest.raises(ValueError, match="not raw per-call"):
        budget.measured_agent([str(path)])


def test_missing_token_usage_is_not_assumed_zero(tmp_path):
    checkpoint(tmp_path, [{"agent_info": {"calls": [{"model": "unknown", "usage": None}]}}])
    with pytest.raises(ValueError, match="no token usage"):
        budget.measured_agent([str(tmp_path)])


def test_react_excludes_judge_calls(tmp_path):
    checkpoint(tmp_path, [{"llm_calls": [
        {"role": "agent", "requested": "gemini/model", "prompt_tokens": 100, "completion_tokens": 30},
        {"role": "judge", "prompt_tokens": 1000, "completion_tokens": 300},
    ]}])
    stats = budget.measured_react(str(tmp_path))
    assert stats["calls"] == 1 and stats["tokens"] == 130


def test_provider_prefix_and_paid_quota_are_distinct():
    assert budget.limits_for("gemini/gemini-3.1-flash-lite")["rpd"] == 500
    assert budget.limits_for("vertex_ai/gemini-3.1-flash-lite")["paid"] is True
    assert budget.limits_for("new-provider/new-model")["note"] == "unknown"


def test_estimated_prices_distinguish_unknown_from_local_and_include_cache():
    assert budget.estimate_cost("unknown/model", 100, 0, 10, "2026-09-30") is None
    assert budget.estimate_cost("ollama_chat/qwen3:8b", 100, 0, 10, "2026-09-30") == 0
    cost = budget.estimate_cost("vertex_ai/gemini-3.8-flash", 1_000_000, 500_000, 1_000_000, "2026-09-30")
    assert cost == pytest.approx(0.375 + 0.0375 + 3.75)
    assert budget.estimate_cost("gemini-3.8-flash", 1_000_000, 500_000, 1_000_000, "2027-01-01") == 2 * cost


def test_budget_limit_fails_before_any_run(tmp_path, monkeypatch, capsys):
    source = checkpoint(tmp_path, [{"agent_info": {"calls": [
        {"model": "gemini-3.1-flash-lite", "component": "solver_big", "prompt": 1000, "output": 100},
    ]}}])
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"b2b": {"single_turn": [1], "multi_turn": []}, "b2c": {"single_turn": [], "multi_turn": []}}))
    monkeypatch.setattr(sys, "argv", ["estimate_budget.py", "--split", str(split), "--systems", "full",
        "--big", "gemini-3.8-flash", "--small", "gemini-3.1-flash-lite", "--judge", "ollama_chat/qwen3:8b",
        "--user", "ollama_chat/qwen3:8b", "--backend", "vertex", "--agent-results", str(source), "--max-cost-usd", "0"])
    with pytest.raises(SystemExit, match="budget not verified"):
        budget.main()
    assert "not an invoice or enforced spending cap" in capsys.readouterr().out
