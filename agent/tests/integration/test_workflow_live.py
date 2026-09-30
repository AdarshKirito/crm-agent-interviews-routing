"""End-to-end workflow tests against the live MCP servers and CRMArena-Pro orgs.

Only the language model is scripted (see scripted_llm.py); tool calls hit the real
Salesforce MCP server and the hybrid search server. Run with both servers up:

  CRMROUTE_LIVE_MCP=1 .venv/Scripts/python -m pytest tests/integration/test_workflow_live.py -v
"""

import json
import os
import re
import uuid

import pytest
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from app import nodes
from app import state_keys as K
from app.agent import build_root_agent

from .scripted_llm import ScriptedLlm, call, last_tool_text

pytestmark = pytest.mark.skipif(os.getenv("CRMROUTE_LIVE_MCP") != "1", reason="needs live MCP servers (CRMROUTE_LIVE_MCP=1)")

# These tests exercise the graph, not the fitted routing table (agent/app/data/routing.yaml
# changes whenever dev is refitted): every task goes to the big solver unless a test says otherwise.
ALL_BIG = {"default": "big", "tiers": {}}
nodes.load_routing_table = lambda: ALL_BIG

B2B_CUSTOMER = "003Wt00000JqmLtIAJ"
ID18 = re.compile(r"\b005[A-Za-z0-9]{15}\b")


def answer(kind: str, text: str, details=None):
    return lambda req: json.dumps({"kind": kind, "answer": text, "known_details": details or []})


async def run_turns(state: dict, messages: list[str], scripts: dict[str, list]):
    models = {name: ScriptedLlm(steps=list(scripts.get(name, [])), requests=[])
              for name in ("solver_big", "solver_small", "policy_check")}
    app = App(name="app", root_agent=build_root_agent(models["solver_big"], models["solver_small"], models["policy_check"]))
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="app", user_id="t", session_id=uuid.uuid4().hex, state=state)
    finals = []
    for message in messages:
        async for _ in runner.run_async(user_id="t", session_id=session.id,
                                        new_message=types.Content(role="user", parts=[types.Part.from_text(text=message)])):
            pass
        s = await runner.session_service.get_session(app_name="app", user_id="t", session_id=session.id)
        finals.append(s.state.get(K.FINAL))
    return finals, s.state, models


@pytest.mark.asyncio
async def test_employee_task_uses_real_salesforce_data_and_checker_normalises_id():
    def answer_with_first_owner_15(req):
        owner = ID18.search(last_tool_text(req)).group(0)
        return json.dumps({"kind": "answer", "answer": f"The agent is {owner[:15]}", "known_details": []})

    finals, state, models = await run_turns(
        {K.ORG: "b2b", K.AUDIENCE: "employee", K.INTERACTIVE: False,
         K.TASK_CONTEXT: "## Handle Time Policy - Definition: the time from case open to close. - Today's date: 2023-01-16"},
        ["In the past four months, which agent had the lowest average handle time? Return only the Id of the agent."],
        {"solver_big": [
            lambda req: call("soql_query", query="SELECT OwnerId, COUNT(Id) n FROM Case WHERE Status = 'Closed' GROUP BY OwnerId ORDER BY COUNT(Id) DESC LIMIT 3"),
            answer_with_first_owner_15,
        ]},
    )
    final = finals[0]
    assert ID18.fullmatch(final), final  # 15-char Id expanded to the 18-char form seen in the tool result
    assert final in state[K.EVIDENCE]
    assert state[K.ROUTE]["task_type"] == "handle_time" and state[K.ROUTE]["tier"] == "big"
    assert state[K.GUARD]["source"] == "employee_session"
    usage = state[K.USAGE]
    assert usage["model_calls"] == 2 and usage["total_cost_usd"] > 0
    # the instruction the model saw carried the task context and the schema
    instruction = str(models["solver_big"].requests[0].config.system_instruction)
    assert "Handle Time Policy" in instruction and "VoiceCallTranscript__c" in instruction


@pytest.mark.asyncio
async def test_checker_retries_once_on_unverified_id():
    def real_owner(req):
        owner = ID18.search(last_tool_text(req) or "").group(0)
        return json.dumps({"kind": "answer", "answer": owner, "known_details": []})

    finals, state, models = await run_turns(
        {K.ORG: "b2c", K.AUDIENCE: "employee", K.INTERACTIVE: False, K.TASK_CONTEXT: "- Today's date: 2023-01-16"},
        ["Which agent closed the most cases? Return only the Id of the agent."],
        {"solver_big": [
            lambda req: call("soql_query", query="SELECT OwnerId, COUNT(Id) n FROM Case GROUP BY OwnerId ORDER BY COUNT(Id) DESC LIMIT 1"),
            answer("answer", "005Wt000009ZZZZIAW"),  # made-up Id -> checker feedback -> retry
            real_owner,
        ]},
    )
    assert state[K.RETRIES] == 1
    assert "do not appear" in str(models["solver_big"].requests[-1].config.system_instruction)
    assert ID18.fullmatch(finals[0]) and finals[0] != "005Wt000009ZZZZIAW"


@pytest.mark.asyncio
async def test_customer_asking_about_another_customer_id_is_refused_without_model_calls():
    finals, state, models = await run_turns(
        {K.ORG: "b2c", K.AUDIENCE: "customer", K.INTERACTIVE: False,
         K.TASK_CONTEXT: "The customer you are interacting with is logged in as Id: 003Ws00000DYTrJIAX."},
        ["Could you provide me with the latest products bought by 003Ws00000DYTkrIAH?"],
        {},
    )
    assert "confidential" in finals[0].lower() or "privacy" in finals[0].lower()
    assert state[K.GUARD]["source"] == "id_check"
    assert not any(m.requests for m in models.values())


@pytest.mark.asyncio
async def test_customer_tool_guard_blocks_internal_data_then_refuses():
    finals, state, _ = await run_turns(
        {K.ORG: "b2b", K.AUDIENCE: "customer", K.INTERACTIVE: False,
         K.TASK_CONTEXT: f"The customer you are interacting with is logged in as Id: {B2B_CUSTOMER}."},
        ["Which agent secured the highest total sales amount in spring 2021?"],
        {
            "policy_check": [lambda req: json.dumps({"decision": "allow", "category": "none", "rationale": "test: let the tool guard act"})],
            "solver_big": [
                lambda req: call("soql_query", query="SELECT OwnerId, SUM(TotalAmount) FROM Order GROUP BY OwnerId"),
                answer("refuse", "I can't share that; it is confidential internal data."),
            ],
        },
    )
    blocked = state[K.BLOCKED_TOOLS]
    assert blocked and blocked[0]["tool"] == "soql_query"
    assert "confidential" in finals[0].lower()


@pytest.mark.asyncio
async def test_customer_knowledge_question_uses_hybrid_search_and_hides_confidential_articles():
    def answer_from_titles(req):
        payload = json.loads(last_tool_text(req))
        assert payload["articles"], payload
        return json.dumps({"kind": "answer", "answer": payload["articles"][0]["Title"], "known_details": []})

    finals, state, models = await run_turns(
        {K.ORG: "b2b", K.AUDIENCE: "customer", K.INTERACTIVE: False,
         K.TASK_CONTEXT: "- Use the information retrieved from the knowledge articles to answer the question in a concise manner"},
        ["Who can benefit from the 10% discount tier?"],
        {
            "policy_check": [lambda req: json.dumps({"decision": "allow", "category": "none", "rationale": "test"})],
            "solver_big": [lambda req: call("search_knowledge", query="10% discount tier volume discount", top_k=5), answer_from_titles],
        },
    )
    tool_text = last_tool_text(models["solver_big"].requests[-1])
    assert "Volume-Based Discounts" not in tool_text  # withheld from a customer session
    assert '"withheld_confidential"' in tool_text
    assert finals[0]
    # the policy classifier saw the request and the sensitive-phrase signal
    policy_prompt = str(models["policy_check"].requests[0].config.system_instruction)
    assert "discount tier" in policy_prompt
    # ...and its verdict never entered the solver's conversation (small models echoed it)
    solver_contents = str(models["solver_big"].requests[0].contents)
    assert '"decision"' not in solver_contents and "rationale" not in solver_contents


@pytest.mark.asyncio
async def test_multi_turn_clarify_then_answer_keeps_conversation_in_state():
    def answer_first_state(req):
        return json.dumps({"kind": "answer", "answer": "MI", "known_details": ["time period: past 6 quarters"]})

    finals, state, models = await run_turns(
        {K.ORG: "b2b", K.AUDIENCE: "employee", K.INTERACTIVE: True, K.TASK_CONTEXT: "- Today's date: 2022-10-26"},
        ["Which states have the quickest case closure time?", "The past 6 quarters."],
        {"solver_big": [answer("clarify", "Which time period should I use?"), answer_first_state]},
    )
    assert finals == ["Which time period should I use?", "MI"]
    assert state[K.CLARIFICATIONS] == ["Which time period should I use?"]
    second = str(models["solver_big"].requests[-1].config.system_instruction)
    assert "Which states have the quickest case closure time?" in second  # earlier turn carried in state
    assert [t["role"] for t in state[K.CONVERSATION]] == ["user", "agent", "user", "agent"]


@pytest.mark.asyncio
async def test_routing_table_sends_a_task_type_to_the_small_solver():
    nodes.load_routing_table = lambda: {"default": "big", "tiers": {"best_region_identification": "small"}}
    try:
        finals, state, models = await run_turns(
            {K.ORG: "b2b", K.AUDIENCE: "employee", K.INTERACTIVE: False, K.TASK_CONTEXT: "- Today's date: 2022-10-26"},
            ["Which states have the quickest case closure time in the past 6 quarters? Return only the two-letter code."],
            {"solver_small": [answer("answer", "MI")]},
        )
    finally:
        nodes.load_routing_table = lambda: ALL_BIG
    assert state[K.ROUTE]["task_type"] == "best_region_identification" and state[K.ROUTE]["tier"] == "small"
    assert finals == ["MI"] and not models["solver_big"].requests
