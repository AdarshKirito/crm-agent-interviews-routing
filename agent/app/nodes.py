"""Function nodes of the crmroute workflow."""

import asyncio
from typing import Any, Literal

from google.adk import Event
from google.genai import types
from pydantic import BaseModel, Field

from . import state_keys as K
from .checker import check_answer, has_refusal_wording
from .config import MAX_CHECKER_RETRIES, MAX_CLARIFYING_QUESTIONS, MODE, PROMPT_GUARD_THRESHOLD, ROUTER_MIN_CONFIDENCE
from .guard import pii
from .guard.pii import analyze_request, scrub
from .guard.prompt_guard import injection_score
from .guard.sensitive import PERSON_ID_PREFIXES, ids_in, in_ids, load_map
from .guard.tool_guard import current_self_ids
from .router import get_router, load_routing_table
from .task_specs import CONFIDENTIALITY_TYPES, spec_for

REFUSALS = {
    "private_customer_information": "it concerns another customer's personal or account information, which is private",
    "internal_operation_data": "it asks for internal company data (such as sales, performance or case analytics), which is confidential",
    "confidential_company_knowledge": "it asks for confidential company information (such as internal pricing, bundling or competitor analysis)",
    "prompt_injection": "it tries to override my instructions",
}


class GuardVerdict(BaseModel):
    decision: Literal["allow", "refuse"]
    category: Literal["private_customer_information", "internal_operation_data", "confidential_company_knowledge", "none"]
    rationale: str = Field(description="one sentence")


class SolverOutput(BaseModel):
    kind: Literal["answer", "clarify", "refuse"]
    answer: str = Field(description="The final answer only, or the one clarifying question, or the refusal")
    known_details: list[str] = Field(default_factory=list, description="Task details the user has given so far")


def refusal_text(category: str | None) -> str:
    why = REFUSALS.get(category or "", "it asks for information that is private or confidential")
    return (f"I'm sorry, but I can't help with that request because {why}. "
            "For privacy and confidentiality reasons I'm not able to share this information.")


def _text(node_input: Any) -> str:
    if isinstance(node_input, types.Content):
        return "\n".join(p.text for p in node_input.parts or [] if p.text).strip()
    return str(node_input or "").strip()


def _user_turns(state) -> list[str]:
    return [t["text"] for t in state.get(K.CONVERSATION) or [] if t["role"] == "user"]


def intake(ctx, node_input: Any):
    conversation = list(ctx.state.get(K.CONVERSATION) or []) + [{"role": "user", "text": _text(node_input)}]
    return Event(state={
        K.CONVERSATION: conversation,
        K.TURN: int(ctx.state.get(K.TURN) or 0) + 1,
        K.RETRIES: 0,
        K.TOOL_CALLS_TURN: 0,
        K.FEEDBACK: "",
        K.DRAFT: {},
        K.CHECK: {},
    })


async def screen(ctx):
    """Cheap request checks. Routes: refuse | allow | classify."""
    state = ctx.state
    turns = _user_turns(state)
    all_text = "\n".join(turns)
    context = state.get(K.TASK_CONTEXT) or ""
    prediction = await asyncio.to_thread(get_router().predict, f"{all_text}\n{context}".strip())
    signals: dict[str, Any] = {
        "nearest_type": prediction.task_type,
        "nearest_confidence": round(prediction.confidence, 3),
        "nearest_top": [[t, round(c, 3)] for t, c in prediction.top],
        "prompt_guard": await injection_score(turns[-1] if turns else ""),
    }
    if signals["prompt_guard"] is not None and signals["prompt_guard"] >= PROMPT_GUARD_THRESHOLD:
        yield Event(state={K.SCREEN: signals, K.GUARD: {"decision": "refuse", "category": "prompt_injection", "source": "prompt_guard"}},
                    route="refuse")
        return
    if state.get(K.AUDIENCE) != "customer":
        yield Event(state={K.SCREEN: signals, K.GUARD: {"decision": "allow", "source": "employee_session"}}, route="allow")
        return
    entities = await asyncio.to_thread(analyze_request, all_text)
    self_ids = current_self_ids(state)
    other_ids = [i for i in ids_in(all_text) if i.startswith(PERSON_ID_PREFIXES) and not in_ids(i, self_ids)]
    signals["pii"] = [f"{e['type']}:{e['text']}" for e in entities if e["type"] != "SALESFORCE_ID"]
    signals["pii_engine"] = pii.ENGINE
    signals["terms"] = load_map().request_signals(all_text)
    signals["other_customer_ids"] = other_ids
    if other_ids:
        yield Event(state={K.SCREEN: signals, K.GUARD: {"decision": "refuse", "category": "private_customer_information",
                                                         "source": "id_check"}}, route="refuse")
        return
    yield Event(state={K.SCREEN: signals}, route="classify")


def make_policy_check(llm, model_name: str, max_tokens: int = 2048):
    """Build the policy-classifier node. It calls the pinned model directly (not as an
    agent node), so its verdict goes to state only and never enters the conversation
    the solver sees -- as an agent node, small solvers echoed its rationale as their answer."""
    from google.adk.models.llm_request import LlmRequest

    from .models import generation_config
    from .prompts import policy_instruction
    from .usage import add_usage

    async def policy_check(ctx):
        config = generation_config(model_name, max_tokens)
        config.system_instruction = policy_instruction(ctx)
        config.response_schema = GuardVerdict
        config.response_mime_type = "application/json"
        request = LlmRequest(model=model_name, config=config, contents=[types.Content(role="user", parts=[
            types.Part.from_text(text="Return your verdict on the customer messages above as JSON.")])])
        last = None
        async for response in llm.generate_content_async(request, stream=False):
            last = response
        meta = dict((last.custom_metadata if last else None) or {})
        add_usage(ctx.state, meta.get("answered_by") or model_name, "policy_check", last.usage_metadata if last else None, meta)
        text = "".join(p.text or "" for p in ((last.content.parts if last and last.content else None) or []) if not p.thought)
        start, end = text.rfind("{"), text.rfind("}")
        # the verdict is the last JSON object in the reply (some models print reasoning first)
        while 0 <= start < end:
            try:
                return GuardVerdict.model_validate_json(text[start:end + 1]).model_dump()
            except ValueError:
                start = text.rfind("{", 0, start) if start > 0 else -1
        return {}

    return policy_check


def decide(ctx, node_input: Any):
    """Combine the policy classifier's verdict with the screening signals."""
    verdict = node_input if isinstance(node_input, dict) else {}
    if not verdict.get("decision"):
        # No usable verdict: fall back to the nearest-example signal.
        screen_signals = ctx.state.get(K.SCREEN) or {}
        refuse = screen_signals.get("nearest_type") in CONFIDENTIALITY_TYPES and screen_signals.get("nearest_confidence", 0) >= 0.6
        verdict = {"decision": "refuse" if refuse else "allow",
                   "category": screen_signals.get("nearest_type") if refuse else "none", "rationale": "classifier unavailable"}
    decision = "refuse" if str(verdict.get("decision")).lower().startswith("refuse") else "allow"
    yield Event(state={K.GUARD: {**verdict, "decision": decision, "source": "policy_classifier"}}, route=decision)


def refuse(ctx):
    guard = ctx.state.get(K.GUARD) or {}
    text = refusal_text(guard.get("category"))
    conversation = list(ctx.state.get(K.CONVERSATION) or []) + [{"role": "agent", "text": text}]
    yield Event(message=text, state={K.FINAL: text, K.CONVERSATION: conversation})


def route_task(ctx):
    signals = ctx.state.get(K.SCREEN) or {}
    task_type = signals.get("nearest_type")
    confidence = float(signals.get("nearest_confidence") or 0.0)
    table = load_routing_table()
    tier = table.get("tiers", {}).get(task_type, table.get("default", "big"))
    reason = "routing table"
    if MODE == "no_route":
        tier, reason = "big", "routing disabled"
    elif MODE == "all_small":
        tier, reason = "small", "small model for every task (dev measurement run)"
    elif confidence < ROUTER_MIN_CONFIDENCE:
        tier, reason = "big", "low router confidence"
    route = {"task_type": task_type, "confidence": confidence, "tier": tier, "reason": reason}
    yield Event(state={K.ROUTE: route}, route=tier)


def check(ctx, node_input: Any):
    state = ctx.state
    draft = node_input if isinstance(node_input, dict) else (state.get(K.DRAFT) or {})
    route = state.get(K.ROUTE) or {}
    result = check_answer(
        str(draft.get("kind") or "answer").lower(),
        str(draft.get("answer") or ""),
        spec_for(route.get("task_type")),
        "\n".join([state.get(K.EVIDENCE) or "", state.get(K.TASK_CONTEXT) or "", *_user_turns(state)]),
        customer=state.get(K.AUDIENCE) == "customer",
        interactive=bool(state.get(K.INTERACTIVE)),
        clarifications=len(state.get(K.CLARIFICATIONS) or []),
        max_clarifications=MAX_CLARIFYING_QUESTIONS,
    )
    retries = int(state.get(K.RETRIES) or 0)
    summary = {"ok": result.ok, "problems": result.problems, "retries": retries, "answer": result.answer}
    if not result.ok and retries < MAX_CHECKER_RETRIES:
        yield Event(state={K.CHECK: summary, K.RETRIES: retries + 1, K.FEEDBACK: " ".join(result.problems)},
                    route=f"retry_{route.get('tier', 'big')}")
        return
    kind = str(draft.get("kind") or "answer").lower()
    if not result.ok and kind == "clarify":
        kind = "answer"  # out of questions (or single-turn): send what we have
    yield Event(state={K.CHECK: summary, K.DRAFT: {**draft, "kind": kind, "answer": result.answer}}, route="ok")


def finalize(ctx):
    state = ctx.state
    draft = state.get(K.DRAFT) or {}
    kind = str(draft.get("kind") or "answer")
    text = str(draft.get("answer") or "").strip() or "None"
    updates: dict[str, Any] = {}
    if kind == "refuse" and not has_refusal_wording(text):
        text = refusal_text(None)
    if kind == "clarify":
        updates[K.CLARIFICATIONS] = list(state.get(K.CLARIFICATIONS) or []) + [text]
    redacted: list[str] = []
    if state.get(K.AUDIENCE) == "customer":
        text, redacted = scrub(text)
    details = list(dict.fromkeys([*(state.get(K.DETAILS) or []), *(draft.get("known_details") or [])]))
    conversation = list(state.get(K.CONVERSATION) or []) + [{"role": "agent", "text": text}]
    updates.update({K.FINAL: text, K.DETAILS: details[-12:], K.CONVERSATION: conversation})
    if redacted:
        updates["crm_redacted"] = redacted
    yield Event(message=text, state=updates)
