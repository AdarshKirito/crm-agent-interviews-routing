"""Exercise the actual ADK graph with scripted models and in-memory CRM tools.

No API keys, model inference, embedding downloads, or running services are needed.
"""

from types import SimpleNamespace

import pytest
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

from app import agent as agent_module
from app import nodes, usage
from app import state_keys as K
from app.models import PinnedLlm

from .scripted_llm import ScriptedLlm, call

KNOWN_ID = "005Wt000003NDqDIAW"
INVENTED_ID = "005Wt000009ZZZZIAW"


def output(kind, answer, details=None):
    return lambda request: call("set_model_response", kind=kind, answer=answer, known_details=details or [])


@pytest.fixture
def isolated_graph(monkeypatch):
    task_type = {"value": "handle_time"}

    async def no_injection(text, **kwargs):
        return None

    def soql_query(query: str) -> dict:
        """Read CRM records for the supplied SOQL query."""
        return {"records": [{"OwnerId": KNOWN_ID}]}

    monkeypatch.setattr(nodes, "injection_score", no_injection)
    monkeypatch.setattr(nodes, "get_router", lambda: SimpleNamespace(predict=lambda text: SimpleNamespace(
        task_type=task_type["value"], confidence=1.0, top=[(task_type["value"], 1.0)])))
    monkeypatch.setattr(nodes, "MODE", "no_route")
    monkeypatch.setattr(nodes, "load_routing_table", lambda: {"default": "big", "tiers": {}})
    monkeypatch.setattr(agent_module, "TOOLSETS", [FunctionTool(soql_query)])
    monkeypatch.setattr(usage, "log_call", lambda record: None)
    return task_type


async def run_turns(messages, steps, state=None):
    scripted = ScriptedLlm(model="gemini-3.8-flash", steps=steps, requests=[])
    solver = PinnedLlm(model=scripted.model, inner=scripted)
    unused = ScriptedLlm(steps=[], requests=[])
    app = App(name="app", root_agent=agent_module.build_root_agent(solver, unused, unused))
    runner = InMemoryRunner(app=app)
    try:
        session = await runner.session_service.create_session(
            app_name="app", user_id="test", state={K.ORG: "b2b", K.AUDIENCE: "employee", **(state or {})})
        finals = []
        for message in messages:
            async for _ in runner.run_async(user_id="test", session_id=session.id,
                                           new_message=types.Content(role="user", parts=[types.Part(text=message)])):
                pass
            session = await runner.session_service.get_session(app_name="app", user_id="test", session_id=session.id)
            finals.append(session.state[K.FINAL])
        return finals, session.state, scripted
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_actual_tool_callback_does_not_validate_models_own_invented_id(isolated_graph):
    finals, state, scripted = await run_turns(["Which agent has the lowest handle time?"], [
        lambda request: call("soql_query", query="SELECT OwnerId FROM Case LIMIT 1"),
        output("answer", INVENTED_ID),
        output("answer", KNOWN_ID[:15]),
    ])
    assert finals == [KNOWN_ID]
    assert state[K.RETRIES] == 1
    assert INVENTED_ID not in state[K.EVIDENCE]
    assert "do not appear" in str(scripted.requests[-1].config.system_instruction)
    assert state[K.USAGE]["model_calls"] == 3


@pytest.mark.asyncio
async def test_multi_turn_follow_up_after_answer_keeps_state_and_resets_budget(isolated_graph):
    isolated_graph["value"] = "lead_qualification"
    finals, state, scripted = await run_turns([
        "Is lead 00QWt00000897NTMAY qualified?",
        "Could you confirm whether their budget meets the requirements?",
    ], [
        lambda request: call("soql_query", query="SELECT OwnerId FROM Case LIMIT 1"),
        output("answer", "Budget", ["lead: 00QWt00000897NTMAY"]),
        output("answer", "Budget"),
    ], {K.INTERACTIVE: True})
    assert finals == ["Budget", "Budget"]
    assert state[K.TURN] == 2
    assert state[K.TOOL_CALLS_TURN] == 0
    assert len(state[K.CONVERSATION]) == 4
    assert "lead: 00QWt00000897NTMAY" in str(scripted.requests[-1].config.system_instruction)


@pytest.mark.asyncio
async def test_multi_turn_clarification_then_answer(isolated_graph):
    isolated_graph["value"] = "best_region_identification"
    finals, state, _ = await run_turns(["Which state closes cases quickest?", "The past two quarters."], [
        output("clarify", "Which time period should I use?"), output("answer", "Michigan"),
    ], {K.INTERACTIVE: True})
    assert finals == ["Which time period should I use?", "MI"]
    assert state[K.CLARIFICATIONS] == ["Which time period should I use?"]
