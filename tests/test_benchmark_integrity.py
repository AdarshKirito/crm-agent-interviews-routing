"""Offline regressions for benchmark failures, resume pins and adapter state."""
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "vendor" / "CRMArena"
sys.path.insert(0, str(BENCH))

from crm_sandbox.agents.remote_agent import RemoteAgent
from crm_sandbox.agents import utils
from crm_sandbox.env.connect_sandbox import SalesforceConnector
from crm_sandbox.env.users import LLMUserSimulationEnv


@pytest.fixture
def runner(monkeypatch, tmp_path):
    assets = ModuleType("crm_sandbox.data.assets")
    task = {"idx": 1, "task": "lead_qualification", "answer": ["Budget"], "query": "Qualify this lead"}
    for name in ("TASKS_ORIGINAL", "TASKS_B2B", "TASKS_B2B_INTERACTIVE", "TASKS_B2C", "TASKS_B2C_INTERACTIVE"):
        setattr(assets, name, [task])
    for name in ("SCHEMA_ORIGINAL", "B2B_SCHEMA", "B2C_SCHEMA", "EXTERNAL_FACING_TASKS"):
        setattr(assets, name, [])
    monkeypatch.setitem(sys.modules, assets.__name__, assets)
    spec = importlib.util.spec_from_file_location("benchmark_runner_test", BENCH / "run_tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.args = SimpleNamespace(log_dir=str(tmp_path), model="test-model", agent_strategy="remote",
        task_category="all", org_type="b2b", interactive=False, task_ids_file=None, reuse_results=True,
        dry_run=False, judge_model="test-judge", judge_provider="test", user_model="test-user", user_provider="test",
        max_user_turns=10, max_turns=20, remote_url="http://localhost", remote_app="app", agent_eval_mode="aided",
        max_consecutive_errors=3, llm_provider="test", privacy_aware_prompt=False)
    module.ChatEnv = lambda **kw: SimpleNamespace(evaluator=SimpleNamespace(total_cost=0))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module


def agent_factory(outcome):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.info = {"usage": {"total_cost_usd": 0.25}, "turns": [{"events": 2}]}

        def act(self, env, index):
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def get_messages(self):
            return [{"role": "user", "content": "Keep this on error"}]
    return FakeAgent


def rows(runner):
    return json.loads(next(Path(runner.args.log_dir).glob("results_*.json")).read_text())


def test_api_failure_returns_nonzero_and_preserves_partial_usage(runner):
    runner.RemoteAgent = agent_factory(requests.HTTPError("500 Server Error"))
    assert runner.run() == 1
    result = rows(runner)[0]
    assert result["error"] is True
    assert result["agent_info"]["usage"]["total_cost_usd"] == 0.25


def test_daily_quota_is_checkpointed_and_resumable(runner):
    runner.RemoteAgent = agent_factory(utils.QuotaExhausted("QUOTA_EXHAUSTED daily"))
    assert runner.run() == 1
    assert rows(runner)[0]["failure"] == "quota_exhausted"
    runner.RemoteAgent = agent_factory(1)
    assert runner.run() == 0
    assert len(rows(runner)) == 1 and not rows(runner)[0].get("error")
    assert rows(runner)[0]["previous_attempts"][0]["failure"] == "quota_exhausted"


def test_scored_failure_is_complete_and_resume_does_not_open_environment(runner):
    runner.RemoteAgent = agent_factory(0)
    assert runner.run() == 0
    runner.ChatEnv = lambda **kw: pytest.fail("Completed run must not log in or call models")
    assert runner.run() == 0


def test_resume_rejects_changed_judge(runner):
    runner.RemoteAgent = agent_factory(1)
    runner.run()
    runner.args.judge_model = "different-judge"
    with pytest.raises(ValueError, match="configuration"):
        runner.run()


def test_unknown_task_id_is_not_silently_dropped(runner, tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"b2b": {"single_turn": [12345]}}))
    runner.args.task_ids_file = str(split)
    with pytest.raises(ValueError, match="task IDs"):
        runner.run()


def test_dry_run_leaves_checkpoint_untouched(runner):
    runner.RemoteAgent = agent_factory(1)
    runner.run()
    checkpoint = next(Path(runner.args.log_dir).glob("results_*.json"))
    before = checkpoint.read_bytes()
    runner.args.dry_run = True
    runner.args.reuse_results = False
    runner.run()
    assert checkpoint.read_bytes() == before


def test_legacy_checkpoint_is_preserved_on_resume(runner):
    checkpoint = Path(runner.args.log_dir) / "results_test-model_remote_all_b2b.json"
    checkpoint.write_text(json.dumps([{"task_id": 1, "reward": 1}]))
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="no configuration"):
        runner.run()
    assert checkpoint.read_bytes() == before


def test_oversized_request_is_a_scored_failure_not_an_api_retry(runner):
    runner.RemoteAgent = agent_factory(utils.RequestTooLarge("too large"))
    assert runner.run() == 0
    assert rows(runner)[0]["error"] is False
    assert rows(runner)[0]["reward"] == 0


def test_remote_failure_retains_failed_turn_and_state(monkeypatch):
    agent = RemoteAgent("http://localhost", "app", "b2b", "internal", interactive=True)
    env = SimpleNamespace(reset=lambda **kw: ("First question", {}), current_user_turn=0)
    env.step = lambda action: ("Follow-up question", 0, False, {})
    monkeypatch.setattr(agent, "_create_session", lambda *a: None)
    monkeypatch.setattr(agent, "_get_state", lambda *a: {"crm_usage": {"total_cost_usd": .5}, "crm_calls": [{"model": "pinned"}]})
    turns = iter([("Answer", []), requests.HTTPError("500 Server Error")])
    def run_turn(*args):
        item = next(turns)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(agent, "_run_turn", run_turn)
    with pytest.raises(requests.HTTPError):
        agent.act(env, 1)
    assert agent.get_messages()[-1] == {"role": "user", "content": "Follow-up question"}
    assert agent.info["total_cost"] == .5
    assert agent.info["calls"] == [{"model": "pinned"}]


@pytest.mark.parametrize("events", [[], [{"errorCode": "INTERNAL", "errorMessage": "solver failed"}], {"bad": "envelope"}])
def test_remote_does_not_grade_invalid_or_errored_responses(monkeypatch, events):
    agent = RemoteAgent("http://localhost", "app", "b2b", "internal")
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: events)
    monkeypatch.setattr(agent.http, "post", lambda *a, **kw: response)
    with pytest.raises(RuntimeError):
        agent._run_turn("session", "question")


def test_remote_multipart_answer_and_empty_state_reset():
    events = [{"content": {"parts": [{"text": "private thought", "thought": True}, {"text": "Budget"}, {"text": " and Need"}]}},
              {"actions": {"stateDelta": {"crm_final_response": ""}}}]
    assert RemoteAgent._final_text(events) == "Budget and Need"


def test_simulator_constructor_does_not_spend_an_unrecorded_call(monkeypatch):
    calls = []
    monkeypatch.setattr(LLMUserSimulationEnv, "generate_next_message", lambda self, messages: calls.append(messages) or "Hello")
    user = LLMUserSimulationEnv("test", "test")
    assert calls == []
    assert user.reset(instruction="Task") == "Hello"
    assert len(calls) == 1


def test_salesforce_dns_error_is_not_masked_as_index_error():
    connector = object.__new__(SalesforceConnector)
    def fail(query):
        raise requests.ConnectionError("DNS resolution failed")
    connector.sf = SimpleNamespace(query_all=fail)
    with pytest.raises(requests.ConnectionError, match="DNS"):
        connector.run_query("SELECT Id FROM Case")


def test_salesforce_query_error_remains_available_to_solver():
    from simple_salesforce.exceptions import SalesforceMalformedRequest
    connector = object.__new__(SalesforceConnector)
    def fail(query):
        raise SalesforceMalformedRequest("https://example.test", 400, "Case", [{"errorCode": "MALFORMED_QUERY", "message": "Bad field"}])
    connector.sf = SimpleNamespace(query_all=fail)
    assert connector.run_query("SELECT wrong FROM Case") == ("MALFORMED_QUERY: Bad field", 0)
