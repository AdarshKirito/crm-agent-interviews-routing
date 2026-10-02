"""Instruction providers. They are callables, so ADK does not template them and SOQL
braces are safe."""

from functools import lru_cache

from google.adk.agents.readonly_context import ReadonlyContext

from . import state_keys as K
from .config import DATA_DIR, MAX_CLARIFYING_QUESTIONS
from .guard.sensitive import self_ids_from_context
from .task_specs import BANT, STAGES, spec_for

ANSWER_FORMS = {
    "id": "only the Salesforce Id (e.g. 005Wt000003NDqDIAW), or None if no record qualifies",
    "ids": "only the Salesforce Ids, comma-separated, or None if no record qualifies",
    "state": "only the two-letter US state code (e.g. CA)",
    "month": "only the month name (e.g. November), or None if the top month is tied or there are no records",
    "stage": f"only one stage name from: {', '.join(STAGES)}, or None if the current stage is already right",
    "bant": f"only the failing factors from: {', '.join(BANT)} (comma-separated), or None if the lead qualifies",
    "text": "a short phrase or comma-separated list (at most about 15 words) in the source's own wording, "
            "no full sentences, preamble or explanation",
    "refusal": "a refusal that says the information is confidential",
}


@lru_cache(maxsize=4)
def load_schema(org: str) -> str:
    path = DATA_DIR / f"schema_{org}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _conversation(state) -> str:
    turns = state.get(K.CONVERSATION) or []
    if len(turns) <= 1:
        return ""
    lines = [f"{'User' if t['role'] == 'user' else 'You'}: {t['text']}" for t in turns[:-1]]
    return "## Conversation so far\n" + "\n".join(lines) + "\n\n"


def solver_instruction(ctx: ReadonlyContext) -> str:
    state = ctx.state
    org = state.get(K.ORG) or "b2b"
    customer = state.get(K.AUDIENCE) == "customer"
    interactive = bool(state.get(K.INTERACTIVE))
    route = state.get(K.ROUTE) or {}
    spec = spec_for(route.get("task_type"))
    parts = []

    if customer:
        own = set(state.get(K.SELF_IDS) or []) | self_ids_from_context(state.get(K.TASK_CONTEXT) or "")
        me = ", ".join(sorted(i for i in own if i.startswith("003"))) or "unknown"
        parts.append(
            "You are the customer-support agent of this company, talking to a logged-in customer "
            f"(their contact Id: {me}). Help with their own records and with public product and service "
            "information. Other customers' data, internal operational data and confidential company "
            "knowledge must not be shared: refuse such requests and say the information is confidential.\n"
        )
    else:
        parts.append("You are a Salesforce CRM analyst helping an employee of the company. Internal data is available to them.\n")

    parts.append(
        "Work only from data you retrieve with your tools; never guess Ids or values. An empty result from a "
        "correctly filtered query is an answer: if no records match the period or condition asked about, answer "
        "None instead of widening the search or picking a value.\n"
        "Tools: soql_query (read-only SOQL), sosl_search (text search), describe_object (fields of an object), "
        "get_record (one record by Id), search_knowledge (the company's knowledge articles).\n"
        "SOQL tips: use relationship fields (e.g. Owner.Name, Account.ShippingState), aggregate with COUNT()/AVG() "
        "and GROUP BY, compare dates with literals like 2024-01-31T00:00:00Z, and compute durations yourself from "
        "the returned timestamps. Filter narrowly and fetch only the fields you need.\n"
        "Time periods: a relative period ('last 4 months', 'past two quarters') ends on today's date from the task "
        "context, so bound BOTH ends of every date filter. The org also holds records dated after today's date; "
        "they are outside the period. If a relative period is asked about and no today's date is given, ask the "
        "user for today's date or the exact dates.\n"
    )

    task_context = (state.get(K.TASK_CONTEXT) or "").strip()
    if task_context:
        parts.append(f"## Task context\n{task_context}\n")
    extra = (state.get(K.EXTRA_CONTEXT) or "").strip()
    if extra:
        parts.append(f"## Domain notes\n{extra}\n")

    if route.get("task_type") and spec.answer != "refusal":
        hints = "\n".join(f"- {h}" for h in spec.hints)
        parts.append(
            f"## Likely task (router guess, confidence {route.get('confidence', 0):.2f})\n"
            f"The request most likely asks for {spec.goal}. If the request clearly asks for something else, follow the request.\n"
            f"Answer form: {ANSWER_FORMS[spec.answer]}.\n{hints}\n"
        )

    parts.append(_conversation(state))
    details = state.get(K.DETAILS) or []
    if details:
        parts.append("## Details the user has given so far\n" + "\n".join(f"- {d}" for d in details) + "\n")
    notes = state.get(K.NOTES) or []
    if notes and len(state.get(K.CONVERSATION) or []) > 1:
        parts.append("## Tool results from earlier turns\n" + "\n".join(f"- {n}" for n in notes) + "\n")

    asked = len(state.get(K.CLARIFICATIONS) or [])
    if interactive and asked < MAX_CLARIFYING_QUESTIONS:
        need = ", ".join(spec.details) if spec.details else "whatever the request leaves open"
        parts.append(
            "## Talking with the user\n"
            f"The user may not say everything at once. This kind of task needs: {need}. If a needed detail is "
            "missing, set kind=clarify and ask ONE short question that names exactly what is missing (you have "
            f"asked {asked} so far). What the request is about (the lead, case, opportunity, quote or product, or "
            "the region for a routing request) can only come from the user: ask right away instead of searching for "
            "candidates. A relative period ('the last five quarters') needs today's date; if the conversation has "
            "not given it, ask for it. If the user says they do not have a detail, do not ask for it again: work "
            "with what they gave. Do not ask about facts you can look up once you know the record.\n"
        )
    else:
        parts.append("## Answer now\nDo not ask questions. If something is ambiguous, choose the most reasonable reading and answer.\n")

    feedback = (state.get(K.FEEDBACK) or "").strip()
    if feedback:
        parts.append(f"## Your previous answer failed a check\n{feedback}\nFix this, verifying with tools where needed.\n")

    parts.append(
        "## Final output\n"
        "When you are done, call the `set_model_response` tool (do not write your answer as plain text) with:\n"
        "- kind: \"answer\", \"clarify\" or \"refuse\"\n"
        "- answer: for kind answer, ONLY the final answer in the answer form above, no sentences or explanation; "
        "for clarify, your one question; for refuse, one sentence saying you cannot share this because it is "
        "private or confidential\n"
        "- known_details: the task details the user has given so far, as a list of short strings\n"
    )
    schema = load_schema(org)
    if schema:
        parts.append(f"## Salesforce schema\n{schema}")
    return "\n".join(p for p in parts if p)


@lru_cache(maxsize=1)
def _policy_text() -> str:
    return (DATA_DIR.parent / "guard" / "policy.md").read_text(encoding="utf-8")


def policy_instruction(ctx: ReadonlyContext) -> str:
    state = ctx.state
    screen = state.get(K.SCREEN) or {}
    turns = [t["text"] for t in state.get(K.CONVERSATION) or [] if t["role"] == "user"]
    context = (state.get(K.TASK_CONTEXT) or "").strip()
    stated = screen.get("self_id_stated") or next((i for i in state.get(K.SELF_IDS) or [] if i.startswith("003")), None)
    if not context and stated:
        context = f"Contact Id interacting: {stated} (stated by the customer in this conversation)"
    return (
        f"{_policy_text()}\n\n## Session\n"
        f"Logged-in customer context: {context or '(none given)'}\n\n"
        "## Customer messages (oldest first)\n" + "\n".join(f"- {t}" for t in turns) + "\n\n"
        "## Signals (hints, not decisions)\n"
        f"- personal data detected: {screen.get('pii') or 'none'}\n"
        f"- sensitive phrases: {screen.get('terms') or 'none'}\n"
        f"- nearest task type: {screen.get('nearest_type')} ({screen.get('nearest_confidence', 0):.2f})\n"
    )
