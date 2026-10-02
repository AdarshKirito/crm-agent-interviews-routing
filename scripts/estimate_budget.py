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
import math
from collections import defaultdict
from datetime import date
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
    if model.startswith("vertex_ai/"):
        return {"note": "Vertex is billed; AI Studio free limits do not apply", "paid": True}
    if model.startswith("gemini/"):
        model = model.removeprefix("gemini/")
    if model in LIMITS:
        return LIMITS[model]
    return LIMITS.get(model.split("/")[0], {"rpd": None, "tpd": None, "note": "unknown"})


def measurement_rows(paths: list[str]) -> list[dict]:
    rows, files = [], set()
    for pattern in paths:
        path = Path(pattern)
        matches = list(path.glob("results_*.json")) if path.is_dir() else [Path(p) for p in glob.glob(pattern)]
        if not matches:
            raise ValueError(f"no measured results match {pattern}")
        for p in matches:
            if p.resolve() in files:
                continue
            files.add(p.resolve())
            text = p.read_text(encoding="utf-8")
            data = [json.loads(l) for l in text.splitlines() if l.strip()] if p.suffix == ".jsonl" else json.loads(text)
            if not isinstance(data, list):
                raise ValueError(f"expected task results in {p}")
            for r in data:
                if r.get("error") or r.get("interactive", p.name.endswith("_interactive.json")):
                    continue
                rows.append(r)
    return rows


def _usage(c: dict) -> tuple[float, float, float]:
    if c.get("usage", "present") is None:
        raise ValueError("a measured call has no token usage; use a complete measurement")
    if "prompt" in c:
        return c["prompt"] or 0, c.get("cached") or 0, (c.get("output") or 0) + (c.get("thoughts") or 0)
    if "prompt_tokens" in c:
        return c["prompt_tokens"] or 0, (c.get("cached_tokens") or 0), c.get("completion_tokens") or 0
    raise ValueError("call has no token counts")


def _summarise(samples: list[dict], models: set[str]) -> dict:
    if not samples:
        return {}
    summary = {key: sum(r[key] for r in samples) / len(samples) for key in ("calls", "prompt", "cached", "output")}
    return {**summary, "n": len(samples), "tokens": summary["prompt"] + summary["output"],
            "peak_prompt": max(r.get("peak_prompt", 0) for r in samples), "source_models": sorted(models)}


def measured_agent(paths: list[str]) -> dict:
    samples, models = [], set()
    for r in measurement_rows(paths):
        if "model_calls" in r:  # task-level output from agent_smoke.py
            samples.append({"calls": r["model_calls"], "prompt": r["prompt_tokens"], "cached": 0,
                            "output": r["output_tokens"], "peak_prompt": 0})
            models.add(r["model"])
        elif "agent_info" in r and "calls" in r["agent_info"]:
            calls = [c for c in r["agent_info"]["calls"] if c.get("component") != "policy_check"]
            counts = [_usage(c) for c in calls]
            samples.append({"calls": len(calls), "prompt": sum(c[0] for c in counts),
                            "cached": sum(c[1] for c in counts), "output": sum(c[2] for c in counts),
                            "peak_prompt": max((c[0] for c in counts), default=0)})
            models.update(c.get("model", "unknown") for c in calls)
        else:
            raise ValueError("--agent-results needs checkpoint JSON or task-level smoke JSONL, not raw per-call logs")
    return _summarise(samples, models)


def measured_react(pattern: str) -> dict:
    samples, models = [], set()
    for r in measurement_rows([pattern]):
        if "llm_calls" not in r:
            raise ValueError("ReAct result has no call accounting")
        calls = [c for c in r["llm_calls"] if c.get("role") == "agent"]
        counts = [_usage(c) for c in calls]
        samples.append({"calls": len(calls), "prompt": sum(c[0] for c in counts),
                        "cached": sum(c[1] for c in counts), "output": sum(c[2] for c in counts),
                        "peak_prompt": max((c[0] for c in counts), default=0)})
        models.update(c.get("answered_by") or c.get("requested", "unknown") for c in calls)
    return _summarise(samples, models)


def estimate_cost(model: str, prompt: float, cached: float, output: float, price_date: str) -> float | None:
    """Standard global text inference list prices, verified 2026-09-30.

    Sources: https://cloud.google.com/vertex-ai/generative-ai/pricing and
    https://ai.google.dev/gemini-api/docs/pricing. No cache-storage/grounding used.
    Unknown models stay unknown, never assumed free.
    """
    if model.startswith(("ollama/", "ollama_chat/")):
        return 0.0
    prices = {"gemini-3.8-flash": (0.75, 0.075, 3.75), "gemini-3.1-flash-lite": (0.25, 0.025, 1.50)}
    name = model.split("/")[-1]
    price = prices.get(name)
    if not price:
        return None
    if name == "gemini-3.8-flash" and price_date >= "2027-01-01":
        price = tuple(p * 2 for p in price)
    return (max(prompt - cached, 0) * price[0] + cached * price[1] + output * price[2]) / 1e6


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
    ap.add_argument("--backend", choices=("ai_studio", "vertex"), default="ai_studio")
    ap.add_argument("--headroom", type=float, default=1.5, help="multiply estimates to allow for longer tasks/retries")
    ap.add_argument("--max-cost-usd", type=float, help="fail when priced estimate exceeds this budget; not a runtime spending cap")
    ap.add_argument("--price-date", default=date.today().isoformat())
    ap.add_argument("--small-share", type=float, default=0.0, help="share of routed tasks sent to the small model")
    ap.add_argument("--policy-share", type=float, default=0.3, help="share of tasks that are customer-facing (policy check)")
    ap.add_argument("--multi-user-turns", type=float, default=3.0, help="ASSUMED user turns per multi-turn task until measured")
    ap.add_argument("--agent-results", nargs="+", default=[str(ROOT / "runs" / "dev_small" / "full")])
    ap.add_argument("--react-results", default=str(ROOT / "runs" / "test_small" / "react"))
    args = ap.parse_args()
    if (args.days <= 0 or args.headroom < 1 or args.multi_user_turns < 1
            or not 0 <= args.small_share <= 1 or not 0 <= args.policy_share <= 1):
        ap.error("days must be positive; headroom/turns >= 1; shares in [0,1]")
    if args.max_cost_usd is not None and args.max_cost_usd < 0:
        ap.error("--max-cost-usd must be nonnegative")
    try:
        date.fromisoformat(args.price_date)
    except ValueError:
        ap.error("--price-date must be YYYY-MM-DD")
    if not set(args.systems) <= {"react", "react_privacy", "full", "routed", "agent_small"}:
        ap.error("unknown system")
    if len(args.systems) != len(set(args.systems)):
        ap.error("duplicate systems")

    split = json.loads(Path(args.split).read_text())
    single = sum(len(split[o]["single_turn"]) for o in ("b2b", "b2c"))
    multi = sum(len(split[o]["multi_turn"]) for o in ("b2b", "b2c"))
    try:
        agent = measured_agent(args.agent_results) if any(not s.startswith("react") for s in args.systems) else {}
        react = measured_react(args.react_results) if any(s.startswith("react") for s in args.systems) else {}
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    if (not agent and any(not s.startswith("react") for s in args.systems)) or (not react and any(s.startswith("react") for s in args.systems)):
        ap.error("need completed single-turn measurements for the selected systems first")
    policy = args.policy or args.small
    print(f"tasks per system: {single} single-turn + {multi} multi-turn")
    for name, measured in (("agent", agent), ("ReAct", react)):
        if measured:
            print(f"measured {name}: {measured['calls']:.1f} solver calls / {measured['tokens'] / 1000:.0f}K tokens "
                  f"per single-turn task (n={measured['n']}; models={measured['source_models']})")
    print(f"ASSUMED: multi-turn task = {args.multi_user_turns:g} user turns x single-turn usage; judge 1 call (~1K tokens) per task; "
          f"simulated user {args.multi_user_turns:g} calls (~2K tokens) per multi-turn task")
    print(f"Headroom: {args.headroom:g}x. This is an estimate, not an invoice or enforced spending cap.")
    print("Policy usage is assumed separately; task-level smoke totals can already include policy calls (conservative overcount).")
    print("Prompt Guard has additional free-tier calls per agent user turn; no paid-price estimate is available here.")
    print("Cross-model usage is extrapolated, not measured on the proposed model; validate with dev before test.")

    use = defaultdict(lambda: {"calls": 0.0, "tokens": 0.0, "prompt": 0.0, "cached": 0.0, "output": 0.0, "peak_prompt": 0, "roles": set()})

    def add(model, calls, prompt, output, role, peak=0):
        if args.backend == "vertex" and model.startswith("gemini-"):
            model = "vertex_ai/" + model
        use[model]["calls"] += calls * args.headroom
        use[model]["tokens"] += (prompt + output) * args.headroom
        use[model]["prompt"] += prompt * args.headroom
        use[model]["output"] += output * args.headroom
        # No cache discounts forecast: cache hits are not guaranteed on a new model/run.
        use[model]["peak_prompt"] = max(use[model]["peak_prompt"], peak)
        use[model]["roles"].add(role)

    for system in args.systems:
        per = react if system.startswith("react") else agent
        tasks = single + multi * args.multi_user_turns
        if system == "routed":
            for model, fraction in ((args.big, 1 - args.small_share), (args.small, args.small_share)):
                if fraction:
                    add(model, per["calls"] * tasks * fraction, per["prompt"] * tasks * fraction,
                        per["output"] * tasks * fraction, f"{system} solver", per["peak_prompt"])
        elif system == "agent_small":
            add(args.small, per["calls"] * tasks, per["prompt"] * tasks, per["output"] * tasks, f"{system} solver", per["peak_prompt"])
        else:
            add(args.big, per["calls"] * tasks, per["prompt"] * tasks, per["output"] * tasks, f"{system} solver", per["peak_prompt"])
        if not system.startswith("react"):
            add(policy, tasks * args.policy_share, tasks * args.policy_share * 1_000, tasks * args.policy_share * 200, "policy check", 1_000)
        add(args.judge, single + multi, (single + multi) * 800, (single + multi) * 200, "judge", 800)
        add(args.user, multi * args.multi_user_turns, multi * args.multi_user_turns * 1_700,
            multi * args.multi_user_turns * 300, "simulated user", 1_700)

    print(f"\n| model | roles | calls | tokens | calls/day needed ({args.days:g} days) | free limit/day | fits? |")
    print("|---|---|---|---|---|---|---|")
    total_cost, unknown_cost, blocked = 0.0, [], []
    for model, u in sorted(use.items(), key=lambda x: -x[1]["calls"]):
        lim = limits_for(model)
        per_day_calls, per_day_tokens = u["calls"] / args.days, u["tokens"] / args.days
        fits = None if lim.get("note") == "unknown" else True
        if lim.get("rpd") is not None and per_day_calls > lim["rpd"]:
            fits = False
        if lim.get("tpd") is not None and per_day_tokens > lim["tpd"]:
            fits = False
        if lim.get("tokens_month") is not None and u["tokens"] > lim["tokens_month"]:
            fits = False
        if lim.get("tpm") and u["peak_prompt"] > lim["tpm"]:
            fits = False
        note = lim.get("note", "")
        if model.startswith("mistral"):
            note += f"; {u['calls'] / 3600:.1f}h at 1 req/s, activation/remaining quota unverified"
        cost = estimate_cost(model, u["prompt"], u["cached"], u["output"], args.price_date)
        if cost is None and u["calls"]:
            unknown_cost.append(model)
        else:
            total_cost += cost or 0
        note += "; estimated list-price " + (f"${cost:.2f}" if cost is not None else "unknown")
        limit_txt = f"{lim.get('rpd') or '-'} req" + (f", {lim['tpd'] / 1e6:g}M tok" if lim.get("tpd") else "")             + (f", {lim['tokens_month'] / 1e9:g}B tok/month" if lim.get("tokens_month") else "")
        verdict = "billed / verify project quota" if lim.get("paid") else "within recorded limits" if fits else "UNKNOWN" if fits is None else "NO"
        print(f"| {model} | {', '.join(sorted(u['roles']))} | {u['calls']:.0f} | {u['tokens'] / 1e6:.1f}M | "
              f"{per_day_calls:.0f} | {limit_txt} | {verdict} ({note}) |")
        if fits is False:
            needed = max(u["calls"] / lim["rpd"] if lim.get("rpd") else 0,
                         u["tokens"] / lim["tpd"] if lim.get("tpd") else 0)
            blocked.append(f"{model}: at least {math.ceil(needed)} quota days; a single prompt above TPM needs shorter context, not waiting")
    print(f"\nKnown standard global list-price subtotal: ${total_cost:.2f} ({args.price_date}, headroom included).")
    print("No cache discounts assumed. Remaining account quota, retries charged without usage, storage and hardware are not measured.")
    if unknown_cost:
        print("Unpriced providers (subtotal is not a total): " + ", ".join(unknown_cost))
    if args.max_cost_usd is not None and (unknown_cost or total_cost > args.max_cost_usd):
        raise SystemExit(f"budget not verified against ${args.max_cost_usd:.2f}; reduce task count or supply measurements/pricing first")
    if blocked:
        print("Reduce the run or spread it across resets:\n" + "\n".join(blocked))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
