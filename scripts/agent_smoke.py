"""Smoke-test the agent on a few real dev tasks with one pinned model per run.

Every role (both solvers and the policy classifier) uses the given model, so each
provider is exercised end to end: tools over MCP, structured output, guard, checker.
Needs the two MCP servers running (scripts/dev_up.sh or run_systems.sh start them).

  agent/.venv/Scripts/python scripts/agent_smoke.py --models gemini-3.1-flash-lite ollama_chat/qwen3:8b
"""
import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from google.adk.apps import App  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from app import state_keys as K  # noqa: E402
from app.agent import build_root_agent  # noqa: E402
from app.models import build_model  # noqa: E402


def grade(task: dict, final: str) -> int:
    """Quick check for the smoke report only; measured runs use the benchmark's graders."""
    if task["reward_metric"] == "privacy_rejection":
        return int(bool(re.search(r"confidential|privacy|private", final or "", re.I)))
    gold = [str(a) if a is not None else "None" for a in task["answer"]]
    got = [s.strip() for s in (final or "").split(",") if s.strip()]
    return int(sorted(got) == sorted(gold))


async def run_task(model: str, task: dict, timeout: float) -> dict:
    app = App(name="app", root_agent=build_root_agent(build_model(model), build_model(model), build_model(model), fallback=False))
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="app", user_id="smoke", session_id=uuid.uuid4().hex, state={
        K.ORG: task["org"], K.AUDIENCE: task["audience"], K.INTERACTIVE: False,
        K.TASK_CONTEXT: task["required"], K.EXTRA_CONTEXT: task["optional"],
    })
    start, error = time.time(), None
    try:
        async def go():
            async for _ in runner.run_async(user_id="smoke", session_id=session.id, new_message=types.Content(
                    role="user", parts=[types.Part.from_text(text=task["query"])])):
                pass
        await asyncio.wait_for(go(), timeout)
    except Exception as err:  # report and continue with the next task
        error = f"{type(err).__name__}: {str(err)[:300]}"
    s = await runner.session_service.get_session(app_name="app", user_id="smoke", session_id=session.id)
    calls = s.state.get(K.CALLS) or []
    final = s.state.get(K.FINAL)
    return {
        "model": model, "org": task["org"], "task_id": task["idx"], "task": task["task"], "gold": task["answer"],
        "final": final, "pass": grade(task, final) if final else 0, "error": error,
        "seconds": round(time.time() - start, 1), "model_calls": len(calls),
        "answered_by": sorted({c["model"] for c in calls}),
        "prompt_tokens": sum(c.get("prompt") or 0 for c in calls), "output_tokens": sum((c.get("output") or 0) + (c.get("thoughts") or 0) for c in calls),
        "tool_calls": len(s.state.get(K.NOTES) or []), "route": (s.state.get(K.ROUTE) or {}).get("task_type"),
        "guard": (s.state.get(K.GUARD) or {}).get("source"), "checker_retries": s.state.get(K.RETRIES),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tasks", default=str(ROOT / "data" / "smoke_tasks.json"))
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default=str(ROOT / "runs" / "smoke" / "agent_smoke.jsonl"))
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks).read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for task in tasks:
            row = await run_task(model, task, args.timeout)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("model", "task", "final", "gold", "pass", "error", "seconds", "model_calls",
                                                   "prompt_tokens", "output_tokens", "tool_calls", "answered_by")}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
