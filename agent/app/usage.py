"""Token accounting and a per-call log, from each response's usage_metadata.

`cost_usd` estimates list-price usage, not the account's invoice or trial-credit
balance. Unknown prices and missing token metadata are explicitly incomplete;
`total_cost_usd` remains the known subtotal for compatibility. Local Ollama has
zero API charges (hardware costs are not estimated). Output includes thinking
tokens; cached input uses the cache rate.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from . import state_keys as K
from .config import CALL_LOG

PRICES_PER_M = {
    "gemini-3.8-flash": {"input": 0.75, "cached": 0.075, "output": 3.75},
    "gemini-3.1-flash-lite": {"input": 0.25, "cached": 0.025, "output": 1.50},
    # Groq model page, checked 2026-10-02: $0.04 per 1M input and output tokens (no cache tier).
    "llama-prompt-guard-2-86m": {"input": 0.04, "cached": 0.04, "output": 0.04},
}
if os.getenv("CRMROUTE_PRICE_DATE", "") >= "2027-01-01":
    PRICES_PER_M["gemini-3.8-flash"] = {"input": 1.50, "cached": 0.15, "output": 7.50}


def cost_usd(model: str, prompt: int, cached: int, output: int, thoughts: int) -> float | None:
    if model.startswith(("ollama/", "ollama_chat/")):
        return 0.0
    price = PRICES_PER_M.get(model.split("/")[-1])
    if price is None:
        return None
    billable_input = max(prompt - cached, 0)
    return (billable_input * price["input"] + cached * price["cached"] + (output + thoughts) * price["output"]) / 1e6


def log_call(record: dict) -> None:
    if not CALL_LOG:
        return
    try:
        path = Path(CALL_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def add_usage(state: Any, model: str, component: str, usage: types.GenerateContentResponseUsageMetadata | None,
              meta: dict | None = None) -> None:
    """Accumulate one model call into state[crm_usage] (per model and per component)
    and append it to state[crm_calls] and the JSONL call log."""
    meta = meta or {}
    calls = list(state.get(K.CALLS) or [])
    record = {"t": round(time.time(), 3), "component": component, "model": model,
              "session": state.get(K.ORG), "session_id": meta.get("session_id"),
              **{k: v for k, v in meta.items() if k in ("attempts", "waited_s", "fallback_from", "repaired_tool_call")}}
    prompt = (usage.prompt_token_count or 0) if usage else 0
    cached = (usage.cached_content_token_count or 0) if usage else 0
    output = (usage.candidates_token_count or 0) if usage else 0
    thoughts = (usage.thoughts_token_count or 0) if usage else 0
    missing = usage is None
    cost = None if missing else cost_usd(model, prompt, cached, output, thoughts)
    unpriced = not missing and cost is None
    complete = cost is not None
    record.update(cost_usd=cost, cost_complete=complete,
                  cost_status="missing_usage" if missing else "unpriced" if unpriced else
                  "local" if model.startswith(("ollama/", "ollama_chat/")) else "priced")
    if missing:
        record["usage"] = None
    else:
        record.update(prompt=prompt, cached=cached, output=output, thoughts=thoughts)
    totals = dict(state.get(K.USAGE) or {})
    for bucket in (f"model:{model}", f"component:{component}"):
        entry = dict(totals.get(bucket) or {"calls": 0, "prompt": 0, "cached": 0, "output": 0, "thoughts": 0, "cost_usd": 0.0})
        entry["calls"] += 1
        entry["prompt"] += prompt
        entry["cached"] += cached
        entry["output"] += output
        entry["thoughts"] += thoughts
        entry["cost_usd"] = round(entry["cost_usd"] + (cost or 0.0), 8)
        entry["cost_complete"] = entry.get("cost_complete", True) and complete
        entry["missing_usage_calls"] = entry.get("missing_usage_calls", 0) + int(missing)
        entry["unpriced_model_calls"] = entry.get("unpriced_model_calls", 0) + int(unpriced)
        totals[bucket] = entry
    totals["total_cost_usd"] = round(totals.get("total_cost_usd", 0.0) + (cost or 0.0), 8)
    totals["model_calls"] = totals.get("model_calls", 0) + 1
    totals["cost_complete"] = totals.get("cost_complete", True) and complete
    totals["missing_usage_calls"] = totals.get("missing_usage_calls", 0) + int(missing)
    totals["unpriced_model_calls"] = totals.get("unpriced_model_calls", 0) + int(unpriced)
    state[K.USAGE] = totals
    calls.append(record)
    state[K.CALLS] = calls[-300:]
    log_call(record)


def usage_callback(model: str, component: str):
    """after_model_callback that records usage under the model that actually answered
    (set by the PinnedLlm / FallbackLlm wrappers), falling back to the configured name."""

    def after_model(callback_context: CallbackContext, llm_response: LlmResponse) -> None:
        # Streaming partials repeat usage; count only complete responses.
        if llm_response.partial:
            return None
        meta = dict(llm_response.custom_metadata or {})
        meta["session_id"] = getattr(getattr(callback_context, "session", None), "id", None)
        add_usage(callback_context.state, meta.get("answered_by") or model, component, llm_response.usage_metadata, meta)
        return None

    return after_model
