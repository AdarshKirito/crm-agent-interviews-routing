"""Estimate the model calls and tokens a run needs, and check them against free limits.

Per-task usage comes from measured runs (agent smoke/dev JSONL and run_tasks result
files); anything not yet measured (e.g. multi-turn) uses the stated assumption, which is
printed. Limits are the observed free-tier values listed in LIMITS (with their source).

  agent/.venv/Scripts/python scripts/estimate_budget.py --split data/test.json \
      --systems react react_privacy full routed --big mistral/mistral-medium-latest \
      --small mistral/mistral-small-latest --judge ollama_chat/qwen3:8b --user ollama_chat/qwen3:8b --days 2
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# requests/day, tokens/day, tokens/minute (None = no stated cap). Sources noted.
LIMITS = {
    "gemini-3.8-flash": {"rpd": 20, "tpd": None, "note": "forum report 2026-09-03; 503 overloads seen 2026-09-29"},
    "gemini-3.1-flash-lite": {"rpd": 500, "tpd": None, "note": "forum report 2026-09-03 (limits only shown in AI Studio)"},
    "mistral": {"rpd": 86_400, "tpd": None, "tokens_month": 1_000_000_000,
                "note": "~1 req/s and ~1B tokens/month (free Experiment plan); 0 req/min until activated"},
    "groq": {"rpd": 1_000, "tpd": 200_000, "tpm": 8_000, "note": "x-ratelimit headers 2026-09-29 + account page"},
    "openrouter": {"rpd": 50, "tpd": None, "note": "free models; upstream throttling seen 2026-09-29"},
    "ollama_chat": {"rpd": None, "tpd": None, "note": "local; throughput measured below"},
}


def limits_for(model: str) -> dict:
    if model in LIMITS:
        return LIMITS[model]
    return LIMITS.get(model.split("/")[0], {"rpd": None, "tpd": None, "note": "unknown"})


def measured_agent(paths: list[str]) -> dict:
    rows = [json.loads(l) for p in paths for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return {}
    by_model = defaultdict(list)
    for r in ok:
        by_model[r["model"]].append(r)
    calls = sum(r["model_calls"] for r in ok) / len(ok)
    tokens = sum(r["prompt_tokens"] + r["output_tokens"] for r in ok) / len(ok)
    secs = {m: sum(r["seconds"] for r in v) / max(1, sum(r["model_calls"] for r in v)) for m, v in by_model.items()}
    return {"n": len(ok), "calls": calls, "tokens": tokens, "sec_per_call": secs}


def measured_react(pattern: str) -> dict:
    rows = [r for f in glob.glob(pattern) for r in json.loads(Path(f).read_text()) if not r.get("error")]
    agent = [[c for c in r.get("llm_calls", []) if c["role"] == "agent"] for r in rows]
    agent = [a for a in agent if a]
    if not agent:
        return {}
    return {"n": len(agent), "calls": sum(len(a) for a in agent) / len(agent),
            "tokens": sum(sum((c["prompt_tokens"] or 0) + (c["completion_tokens"] or 0) for c in a) for a in agent) / len(agent)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(ROOT / "data" / "test.json"))
    ap.add_argument("--systems", nargs="+", default=["react", "react_privacy", "full", "routed"])
    ap.add_argument("--big", required=True)
    ap.add_argument("--small", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--small-share", type=float, default=0.0, help="share of routed tasks sent to the small model")
    ap.add_argument("--policy-share", type=float, default=0.3, help="share of tasks that are customer-facing (policy check)")
    ap.add_argument("--multi-user-turns", type=float, default=3.0, help="ASSUMED user turns per multi-turn task until measured")
    ap.add_argument("--agent-results", nargs="*", default=[str(ROOT / "runs" / "smoke" / "agent_smoke_v2.jsonl")])
    ap.add_argument("--react-results", default=str(ROOT / "runs" / "smoke" / "react" / "*.json"))
    args = ap.parse_args()

    split = json.loads(Path(args.split).read_text())
    single = sum(len(split[o]["single_turn"]) for o in ("b2b", "b2c"))
    multi = sum(len(split[o]["multi_turn"]) for o in ("b2b", "b2c"))
    agent, react = measured_agent(args.agent_results), measured_react(args.react_results)
    if not agent or not react:
        raise SystemExit("need measured agent and ReAct results first (scripts/agent_smoke.py, a small run_tasks run)")
    policy = args.policy or args.small
    print(f"tasks per system: {single} single-turn + {multi} multi-turn")
    print(f"measured per single-turn task: agent {agent['calls']:.1f} calls / {agent['tokens'] / 1000:.0f}K tokens (n={agent['n']}); "
          f"ReAct {react['calls']:.1f} calls / {react['tokens'] / 1000:.0f}K tokens (n={react['n']})")
    print(f"ASSUMED: multi-turn task = {args.multi_user_turns:g} user turns x single-turn usage; judge 1 call (~1K tokens) per task; "
          f"simulated user {args.multi_user_turns:g} calls (~2K tokens) per multi-turn task")

    use = defaultdict(lambda: {"calls": 0.0, "tokens": 0.0, "roles": set()})

    def add(model, calls, tokens, role):
        use[model]["calls"] += calls
        use[model]["tokens"] += tokens
        use[model]["roles"].add(role)

    for system in args.systems:
        per = react if system.startswith("react") else agent
        tasks = single + multi * args.multi_user_turns
        if system == "routed":
            add(args.big, per["calls"] * tasks * (1 - args.small_share), per["tokens"] * tasks * (1 - args.small_share), f"{system} solver")
            add(args.small, per["calls"] * tasks * args.small_share, per["tokens"] * tasks * args.small_share, f"{system} solver")
        elif system == "agent_small":
            add(args.small, per["calls"] * tasks, per["tokens"] * tasks, f"{system} solver")
        else:
            add(args.big, per["calls"] * tasks, per["tokens"] * tasks, f"{system} solver")
        if not system.startswith("react"):
            add(policy, (single + multi) * args.policy_share, (single + multi) * args.policy_share * 1_200, "policy check")
        add(args.judge, single + multi, (single + multi) * 1_000, "judge")
        add(args.user, multi * args.multi_user_turns, multi * args.multi_user_turns * 2_000, "simulated user")

    print(f"\n| model | roles | calls | tokens | calls/day needed ({args.days:g} days) | free limit/day | fits? |")
    print("|---|---|---|---|---|---|---|")
    for model, u in sorted(use.items(), key=lambda x: -x[1]["calls"]):
        lim = limits_for(model)
        per_day_calls, per_day_tokens = u["calls"] / args.days, u["tokens"] / args.days
        fits = True
        if lim.get("rpd") is not None and per_day_calls > lim["rpd"]:
            fits = False
        if lim.get("tpd") is not None and per_day_tokens > lim["tpd"]:
            fits = False
        if lim.get("tokens_month") is not None and u["tokens"] > lim["tokens_month"]:
            fits = False
        if lim.get("rpd") and model.startswith("mistral"):
            note = f"{u['calls'] / 3600:.1f} h at 1 req/s; " + lim.get("note", "")
        note = locals().get("note") if model.startswith("mistral") else lim.get("note", "")
        if model.startswith("ollama") and agent.get("sec_per_call", {}).get(model):
            hours = u["calls"] * agent["sec_per_call"][model] / 3600
            note += f"; ~{hours:.1f} GPU hours at {agent['sec_per_call'][model]:.0f}s/call"
            fits = hours <= 20 * args.days
        limit_txt = f"{lim.get('rpd') or '-'} req" + (f", {lim['tpd'] / 1e6:g}M tok" if lim.get("tpd") else "")             + (f", {lim['tokens_month'] / 1e9:g}B tok/month" if lim.get("tokens_month") else "")
        print(f"| {model} | {', '.join(sorted(u['roles']))} | {u['calls']:.0f} | {u['tokens'] / 1e6:.1f}M | "
              f"{per_day_calls:.0f} | {limit_txt} | {'yes' if fits else 'NO'} ({note}) |")


if __name__ == "__main__":
    main()
