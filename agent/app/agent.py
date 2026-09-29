"""crmroute: guarded, routed ADK workflow for CRMArena-Pro tasks.

    START -> intake -> screen --refuse--------------------------------------------> refuse
                          |--allow (employee)--> route_task --big--> solver_big --\
                          '--classify--> policy_check -> decide     '--small--> solver_small --> check --ok--> finalize
                                                           |--allow--> route_task       ^            |
                                                           '--refuse--> refuse          '--retry-----'
"""

from google.adk import Agent, Workflow
from google.adk.apps import App
from google.adk.models import BaseLlm, Gemini
from google.genai import types

from . import state_keys as K
from .config import BIG_MODEL, MAX_OUTPUT_TOKENS, SMALL_MODEL, TEMPERATURE, THINKING_LEVEL
from .guard.tool_guard import after_tool, before_tool
from .nodes import GuardVerdict, SolverOutput, check, decide, finalize, intake, refuse, route_task, screen
from .prompts import policy_instruction, solver_instruction
from .tools import build_toolsets
from .tracing import setup_tracing
from .usage import usage_callback

setup_tracing()


def _model(name: str) -> Gemini:
    return Gemini(model=name, retry_options=types.HttpRetryOptions(attempts=4, initial_delay=2.0))


def _config(max_tokens: int = MAX_OUTPUT_TOKENS) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel(THINKING_LEVEL.upper())),
    )


TOOLSETS = build_toolsets()


def _solver(name: str, model: BaseLlm, model_name: str) -> Agent:
    return Agent(
        name=name,
        model=model,
        description=f"Solves the CRM task with read-only tools on {model_name}.",
        instruction=solver_instruction,
        tools=TOOLSETS,
        output_schema=SolverOutput,
        output_key=K.DRAFT,
        generate_content_config=_config(),
        before_tool_callback=before_tool,
        after_tool_callback=after_tool,
        after_model_callback=usage_callback(model_name, name),
    )


def build_root_agent(big: BaseLlm | None = None, small: BaseLlm | None = None, policy: BaseLlm | None = None) -> Workflow:
    """Build the workflow. Models can be injected (tests use scripted ones); the
    Workflow copies its nodes, so they must be set here rather than patched later."""
    solver_big = _solver("solver_big", big or _model(BIG_MODEL), BIG_MODEL)
    solver_small = _solver("solver_small", small or _model(SMALL_MODEL), SMALL_MODEL)
    policy_check = Agent(
        name="policy_check",
        model=policy or _model(SMALL_MODEL),
        description="Applies the written confidentiality policy to a customer's request.",
        instruction=policy_instruction,
        output_schema=GuardVerdict,
        generate_content_config=_config(2048),
        after_model_callback=usage_callback(SMALL_MODEL, "policy_check"),
    )
    return Workflow(
        name="crmroute",
        edges=[
            ("START", intake, screen),
            (screen, {"refuse": refuse, "allow": route_task, "classify": policy_check}),
            (policy_check, decide),
            (decide, {"refuse": refuse, "allow": route_task}),
            (route_task, {"big": solver_big, "small": solver_small}),
            (solver_big, check),
            (solver_small, check),
            (check, {"retry_big": solver_big, "retry_small": solver_small, "ok": finalize}),
        ],
    )


root_agent = build_root_agent()
app = App(root_agent=root_agent, name="app")
