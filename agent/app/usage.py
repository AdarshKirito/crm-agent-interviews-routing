"""Token accounting and a per-call log, from each response's usage_metadata.

Runs use free tiers, so the money actually spent is $0. `cost_usd` is the Gemini
paid-tier list-price equivalent (Gemini API pricing page, checked 2026-09-27; 3.8
Flash prices double on 2027-01-01); models without a price here count 0 and are
reported by tokens. Output includes thinking tokens; cached input is billed at the
cache rate.
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
}
if os.getenv("CRMROUTE_PRICE_DATE", "") >= "2027-01-01":
    PRICES_PER_M["gemini-3.8-flash"] = {"input": 1.50, "cached": 0.15, "output": 7.50}


def cost_usd(model: str, prompt: int, cached: int, output: int, thoughts: int) -> float:
    price = PRICES_PER_M.get(model.split("/")[-1])
    if price is None:
        return 0.0
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
    record = {"t": round(time.time(), 3), "component": component, "model": model}
    if usage is None:
        record["usage"] = None
        calls.append(record)
        state[K.CALLS] = calls[-300:]
        log_call(record)
        return
    prompt = usage.prompt_token_count or 0
    cached = usage.cached_content_token_count or 0
    output = usage.candidates_token_count or 0
    thoughts = usage.thoughts_token_count or 0
    cost = cost_usd(model, prompt, cached, output, thoughts)
    totals = dict(state.get(K.USAGE) or {})
    for bucket in (f"model:{model}", f"component:{component}"):
        entry = dict(totals.get(bucket) or {"calls": 0, "prompt": 0, "cached": 0, "output": 0, "thoughts": 0, "cost_usd": 0.0})
        entry["calls"] += 1
        entry["prompt"] += prompt
        entry["cached"] += cached
        entry["output"] += output
        entry["thoughts"] += thoughts
        entry["cost_usd"] = round(entry["cost_usd"] + cost, 8)
        totals[bucket] = entry
    totals["total_cost_usd"] = round(totals.get("total_cost_usd", 0.0) + cost, 8)
    totals["model_calls"] = totals.get("model_calls", 0) + 1
    state[K.USAGE] = totals
    record.update(prompt=prompt, cached=cached, output=output, thoughts=thoughts,
                  **{k: v for k, v in meta.items() if k in ("attempts", "waited_s", "fallback_from", "repaired_tool_call")})
    calls.append(record)
    state[K.CALLS] = calls[-300:]
    log_call({**record, "session": state.get(K.ORG), "session_id": meta.get("session_id")})


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
