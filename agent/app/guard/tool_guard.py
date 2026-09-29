"""Tool-call guard (before_tool_callback) and result bookkeeping (after_tool_callback).

Customer-facing sessions: calls that would touch internal objects, other customers'
records or confidential articles are refused before they reach Salesforce, and
confidential articles are removed from knowledge-search results. Every session:
tool results are kept as evidence for the checker and as notes for later turns.
"""

import json
from typing import Any

from google.adk.tools import BaseTool, ToolContext

from .. import state_keys as K
from .sensitive import ids_in, in_ids, load_map, self_ids_from_context

MAX_EVIDENCE_CHARS = 300_000
MAX_NOTE_CHARS = 1_200
MAX_NOTES = 8


def _text_of(response: Any) -> str:
    if isinstance(response, dict):
        parts = [c.get("text", "") for c in response.get("content") or [] if isinstance(c, dict)]
        if parts:
            return "\n".join(parts)
        if "error" in response:
            return str(response["error"])
    return json.dumps(response, default=str) if not isinstance(response, str) else response


def _is_error(response: Any) -> bool:
    return isinstance(response, dict) and bool(response.get("isError") or response.get("is_error") or "error" in response)


def current_self_ids(state: Any) -> set[str]:
    ids = set(state.get(K.SELF_IDS) or [])
    return ids | self_ids_from_context(state.get(K.TASK_CONTEXT) or "")


def before_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> dict | None:
    state = tool_context.state
    if state.get(K.AUDIENCE) != "customer":
        return None
    smap = load_map()
    self_ids = current_self_ids(state)
    name = tool.name
    if name == "soql_query":
        verdict = smap.check_soql(str(args.get("query", "")), self_ids)
    elif name == "sosl_search":
        verdict = smap.check_sosl(str(args.get("query", "")))
    elif name == "get_record":
        verdict = smap.check_get_record(
            str(args.get("object_name", "")), str(args.get("record_id", "")), args.get("fields"), self_ids
        )
    else:  # describe_object, search_knowledge
        return None
    if verdict.allowed:
        return None
    blocked = list(state.get(K.BLOCKED_TOOLS) or [])
    blocked.append({"tool": name, "args": args, "reason": verdict.reason, "category": verdict.category})
    state[K.BLOCKED_TOOLS] = blocked
    return {
        "isError": True,
        "content": [{"type": "text", "text": f"Blocked by the confidentiality policy: {verdict.reason} "
                     "If the request needs this data, refuse and say the information is confidential."}],
    }


def _filter_confidential_articles(response: dict) -> dict:
    text = _text_of(response)
    try:
        payload = json.loads(text)
    except ValueError:
        return response
    articles = payload.get("articles") or []
    smap = load_map()
    kept = [a for a in articles if not smap.is_confidential_article(a.get("Title"))]
    if len(kept) == len(articles):
        return response
    payload["articles"] = kept
    payload["returned"] = len(kept)
    payload["withheld_confidential"] = len(articles) - len(kept)
    payload["note"] = ("Some matching articles are internal and were withheld. If the question can only be "
                       "answered from internal material, refuse and say it is confidential.")
    return {**response, "content": [{"type": "text", "text": json.dumps(payload)}]}


def after_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any) -> dict | None:
    state = tool_context.state
    replaced = None
    if (
        tool.name == "search_knowledge"
        and state.get(K.AUDIENCE) == "customer"
        and isinstance(tool_response, dict)
        and not _is_error(tool_response)
    ):
        replaced = _filter_confidential_articles(tool_response)
        tool_response = replaced
    text = _text_of(tool_response)

    evidence = (state.get(K.EVIDENCE) or "") + "\n" + text
    state[K.EVIDENCE] = evidence[-MAX_EVIDENCE_CHARS:]

    notes = list(state.get(K.NOTES) or [])
    call = json.dumps(args, default=str)[:300]
    notes.append(f"{tool.name}({call}) -> {'ERROR ' if _is_error(tool_response) else ''}{text[:MAX_NOTE_CHARS]}")
    state[K.NOTES] = notes[-MAX_NOTES:]

    # A successful call tied to the customer's own ids extends the set of their ids
    # (their account, orders, cases) for later scoped calls.
    if state.get(K.AUDIENCE) == "customer" and not _is_error(tool_response):
        self_ids = current_self_ids(state)
        if any(in_ids(i, self_ids) for i in ids_in(json.dumps(args, default=str))):
            state[K.SELF_IDS] = sorted(self_ids | ids_in(text))
    return replaced
