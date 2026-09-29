"""Token and cost accounting per session, from each response's usage_metadata.

Prices are list prices in USD per 1M tokens (Gemini API pricing page, checked
2026-09-27; 3.8 Flash prices double on 2027-01-01). Output includes thinking tokens.
Cached input tokens are billed at the cache rate instead of the input rate.
"""

import os
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from . import state_keys as K

PRICES_PER_M = {
    "gemini-3.8-flash": {"input": 0.75, "cached": 0.075, "output": 3.75},
    "gemini-3.1-flash-lite": {"input": 0.25, "cached": 0.025, "output": 1.50},
}
if os.getenv("CRMROUTE_PRICE_DATE", "") >= "2027-01-01":
    PRICES_PER_M["gemini-3.8-flash"] = {"input": 1.50, "cached": 0.15, "output": 7.50}


def cost_usd(model: str, prompt: int, cached: int, output: int, thoughts: int) -> float:
    price = PRICES_PER_M.get(model)
    if price is None:
        return 0.0
    billable_input = max(prompt - cached, 0)
    return (billable_input * price["input"] + cached * price["cached"] + (output + thoughts) * price["output"]) / 1e6


def add_usage(state: Any, model: str, component: str, usage: types.GenerateContentResponseUsageMetadata | None) -> None:
    """Accumulate one model call into state[crm_usage] (per model and per component)."""
    if usage is None:
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


def usage_callback(model: str, component: str):
    """after_model_callback that records usage for an agent running `model`."""

    def after_model(callback_context: CallbackContext, llm_response: LlmResponse) -> None:
        # Streaming partials repeat usage; count only complete responses.
        if llm_response.partial:
            return None
        add_usage(callback_context.state, model, component, llm_response.usage_metadata)
        return None

    return after_model
