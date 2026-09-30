"""crmroute: guarded, routed ADK workflow for CRMArena-Pro tasks.

    START -> intake -> screen --refuse--------------------------------------------> refuse
                          |--allow (employee)--> route_task --big--> solver_big --\
                          '--classify--> policy_check -> decide     '--small--> solver_small --> check --ok--> finalize
                                                           |--allow--> route_task       ^            |
                                                           '--refuse--> refuse          '--retry-----'
"""

from google.adk import Agent, Workflow
from google.adk.apps import App
from google.adk.models import BaseLlm

from . import state_keys as K
from .config import BIG_MODEL, FALLBACK, MAX_OUTPUT_TOKENS, POLICY_MODEL, SMALL_MODEL
from .guard.tool_guard import after_tool, before_tool
from .models import build_model, generation_config
from .nodes import SolverOutput, check, decide, finalize, intake, make_policy_check, refuse, route_task, screen
from .prompts import solver_instruction
from .tools import build_toolsets
from .tracing import setup_tracing
from .usage import usage_callback

setup_tracing()


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
        generate_content_config=generation_config(model_name, MAX_OUTPUT_TOKENS),
        before_tool_callback=before_tool,
        after_tool_callback=after_tool,
        after_model_callback=usage_callback(model_name, name),
    )


def build_root_agent(big: BaseLlm | None = None, small: BaseLlm | None = None, policy: BaseLlm | None = None,
                     fallback: bool = FALLBACK) -> Workflow:
    """Build the workflow. Models can be injected (tests use scripted ones); the
    Workflow copies its nodes, so they must be set here rather than patched later.
    `fallback` is for the local demo only; measured runs keep one pinned model per role."""
    solver_big = _solver("solver_big", big or build_model(BIG_MODEL, fallback), BIG_MODEL)
    solver_small = _solver("solver_small", small or build_model(SMALL_MODEL, fallback), SMALL_MODEL)
    # a function node, not an agent: its verdict must stay out of the solver's conversation
    policy_check = make_policy_check(policy or build_model(POLICY_MODEL, fallback), POLICY_MODEL)
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
