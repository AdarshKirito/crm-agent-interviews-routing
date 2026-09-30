"""Summarise benchmark runs and compare systems on the same task ids.

Each system is a directory of run_tasks.py checkpoint files (results_*.json).
Rewards: exact_match and privacy_rejection give 0/1; fuzzy_match (knowledge_qa,
sales_insight_mining) gives a dict, scored here by token F1 (as in the CRMArena-Pro
paper) and counted as success when F1 >= --fuzzy-threshold for the success rate.

  python scripts/analyze_results.py --system react=runs/test/react --system react_privacy=runs/test/react_privacy \
      --system full=runs/test/full --system routed=runs/test/routed --baseline react --out runs/test/summary.md
"""
import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

CONFIDENTIALITY = {"private_customer_information", "internal_operation_data", "confidential_company_knowledge"}


def score(result: dict, threshold: float) -> tuple[float, float]:
    """(graded score in [0,1], binary success)."""
    reward = result.get("reward", 0)
    if isinstance(reward, dict):
        f1 = float(reward.get("f1", 0.0))
        return f1, float(f1 >= threshold)
    value = float(reward or 0)
    return value, float(value >= 1)


def _interactive_from_name(name: str) -> bool:
    """Released CRMArena files end in interactive-True/False; run_tasks.py (patched) in _interactive."""
    m = re.search(r"interactive-(True|False)", name)
    return m.group(1) == "True" if m else name.endswith("_interactive.json")


def load_system(path: Path) -> dict[tuple, dict]:
    """Load every results_*.json under `path`, or the files matching `path` if it is a glob."""
    rows = {}
    files = sorted(path.parent.glob(path.name)) if "*" in path.name else sorted(path.glob("results_*.json"))
    for f in files:
        for r in json.loads(f.read_text(encoding="utf-8")):
            interactive = r.get("interactive", _interactive_from_name(f.name))
            key = (r.get("org_type") or ("b2c" if "_b2c" in f.name else "b2b"), bool(interactive), str(r["task_id"]))
            rows[key] = r
    return rows


def cost_of(r: dict) -> float:
    info = r.get("agent_info") or {}
    return float(info.get("total_cost") or 0.0)


def agent_calls_of(r: dict) -> list[dict]:
    """The system's own model calls: the agent server's per-call log (remote systems) or
    the ReAct agent's calls recorded by run_tasks.py; judge and simulated-user calls excluded."""
    info = r.get("agent_info") or {}
    if info.get("calls") is not None:
        return list(info["calls"])
    return [c for c in r.get("llm_calls") or [] if c.get("role") == "agent"]


def call_model(c: dict) -> str:
    return str(c.get("model") or c.get("answered_by") or c.get("requested") or "")


def call_tokens(c: dict) -> int:
    if "prompt" in c:
        return (c.get("prompt") or 0) + (c.get("output") or 0) + (c.get("thoughts") or 0)
    return (c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0)


def latency_of(r: dict) -> float | None:
    turns = (r.get("agent_info") or {}).get("turns")
    if turns:
        return sum(t.get("latency_s", 0) for t in turns)
    return None


def steps_of(r: dict) -> int | None:
    info = r.get("agent_info") or {}
    if info.get("turns"):
        return sum(t.get("tool_calls", 0) for t in info["turns"])
    if "observation_sizes" in info:
        return len(info["observation_sizes"])
    return None


def tool_errors_of(r: dict) -> int:
    info = r.get("agent_info") or {}
    if info.get("turns"):
        return sum(t.get("tool_errors", 0) for t in info["turns"])
    end = info.get("end_reason") or {}
    return int("error" in str(end.get("message", "")).lower())


def bootstrap(values: list[float], iters: int, seed: int) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", action="append", required=True, help="name=path")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--fuzzy-threshold", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--big-model", default="flash-lite", help="substring naming the hosted (quota-limited) big model")
    args = ap.parse_args()

    systems = {}
    for spec in args.system:
        name, path = spec.split("=", 1)
        systems[name] = load_system(Path(path))
    common = set.intersection(*(set(rows) for rows in systems.values()))
    errored = {n: sum(1 for r in rows.values() if r.get("error")) for n, rows in systems.items()}
    lines = [f"Tasks common to all systems: {len(common)} "
             f"(counts per system: {', '.join(f'{n}={len(r)}' for n, r in systems.items())}; "
             f"tasks still ending in an API error: {errored})", ""]

    groups = {
        "business, single-turn": lambda r, k: not k[1] and r["task_type"] not in CONFIDENTIALITY,
        "business, multi-turn": lambda r, k: k[1] and r["task_type"] not in CONFIDENTIALITY,
        "confidentiality (refusal rate)": lambda r, k: r["task_type"] in CONFIDENTIALITY,
    }
    lines += ["| system | group | n | success % [95% CI] | mean score | model calls/task | big-model calls/task | "
              "tokens/task (K) | list-price $/task | p95 latency s | tool steps/task | tool errors |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary: dict = defaultdict(dict)
    for name, rows in systems.items():
        for gname, pred in groups.items():
            keys = sorted(k for k in common if pred(rows[k], k))
            if not keys:
                continue
            graded, success = zip(*(score(rows[k], args.fuzzy_threshold) for k in keys))
            lo, hi = bootstrap(list(success), args.iters, args.seed)
            costs = [cost_of(rows[k]) for k in keys]  # the system's own model spend; judge cost is reported by run_tasks separately
            lats = [x for k in keys if (x := latency_of(rows[k])) is not None]
            steps = [x for k in keys if (x := steps_of(rows[k])) is not None]
            errors = sum(tool_errors_of(rows[k]) for k in keys)
            calls = [agent_calls_of(rows[k]) for k in keys]
            n_calls = sum(len(c) for c in calls) / len(keys)
            n_big = sum(1 for cs in calls for c in cs if args.big_model in call_model(c)) / len(keys)
            tokens = sum(call_tokens(c) for cs in calls for c in cs) / len(keys) / 1000
            rate = sum(success) / len(success)
            summary[name][gname] = {"keys": keys, "success": list(success)}
            lines.append(
                f"| {name} | {gname} | {len(keys)} | {pct(rate)} [{pct(lo)}, {pct(hi)}] | {sum(graded) / len(graded):.3f} | "
                f"{n_calls:.1f} | {n_big:.1f} | {tokens:.0f} | {sum(costs) / len(costs):.4f} | {percentile(lats, 0.95):.1f} | "
                f"{(sum(steps) / len(steps)) if steps else float('nan'):.1f} | {errors} |"
            )

    if args.baseline and args.baseline in systems:
        lines += ["", f"Paired differences vs `{args.baseline}` (same task ids; win claimed only if the CI excludes 0):", "",
                  "| system | group | n | diff (points) [95% CI] | win? |", "|---|---|---|---|---|"]
        base = summary[args.baseline]
        rng = random.Random(args.seed)
        for name in systems:
            if name == args.baseline:
                continue
            for gname, entry in summary[name].items():
                if gname not in base:
                    continue
                diffs = [a - b for a, b in zip(entry["success"], base[gname]["success"])]
                n = len(diffs)
                boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(args.iters))
                lo, hi = boots[int(0.025 * args.iters)], boots[int(0.975 * args.iters) - 1]
                mean = sum(diffs) / n
                lines.append(f"| {name} | {gname} | {n} | {100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
                             f"{'yes' if lo > 0 else ('worse' if hi < 0 else 'no')} |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
