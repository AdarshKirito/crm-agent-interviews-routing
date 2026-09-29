"""A stand-in model for workflow tests.

Each step is a function of the LlmRequest (so it can read real tool results) that
returns a function call or text. The test decides what the "model" does; everything
else -- workflow, MCP servers, Salesforce, guards, checker -- is real.
"""

import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

Step = Callable[[LlmRequest], dict[str, Any] | str]


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"call": name, "args": args}


def last_tool_text(req: LlmRequest) -> str:
    for content in reversed(req.contents):
        for part in content.parts or []:
            if part.function_response:
                resp = part.function_response.response or {}
                blocks = resp.get("content") or []
                return "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict)) or json.dumps(resp)
    return ""


class ScriptedLlm(BaseLlm):
    model: str = "scripted"
    steps: list = []
    requests: list = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        if not self.steps:
            raise AssertionError("scripted model ran out of steps")
        out = self.steps.pop(0)(llm_request)
        if isinstance(out, dict) and "call" in out:
            part = types.Part.from_function_call(name=out["call"], args=out["args"])
        else:
            part = types.Part.from_text(text=out if isinstance(out, str) else json.dumps(out))
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1000, candidates_token_count=50, thoughts_token_count=20, total_token_count=1070
            ),
        )
