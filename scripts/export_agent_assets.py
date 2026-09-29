"""Write the data files the crmroute agent loads at start-up.

- agent/app/data/schema_<org>.md: the org schema text, built with the benchmark's own
  ChatAgent._build_schema, so the agent sees exactly what the ReAct baseline sees.
- agent/app/data/router_examples.jsonl: labelled example requests for the task-type
  router, drawn only from tasks that are NOT in test.json (single-turn and multi-turn
  lists are both excluded). Each example is the request plus its task context, the
  same text the agent receives at run time.

Run with the benchmark environment:
  vendor/CRMArena/.venv/Scripts/python scripts/export_agent_assets.py
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crmarena", default="vendor/CRMArena")
    ap.add_argument("--exclude", default="data/test.json")
    ap.add_argument("--out-dir", default="agent/app/data")
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    sys.path.insert(0, args.crmarena)
    from crm_sandbox.agents.chat_agent import ChatAgent
    from crm_sandbox.data.assets import B2B_SCHEMA, B2C_SCHEMA, TASKS_B2B, TASKS_B2C

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for org, schema in [("b2b", B2B_SCHEMA), ("b2c", B2C_SCHEMA)]:
        text = ChatAgent._build_schema(None, schema)
        (out / f"schema_{org}.md").write_text(text, encoding="utf-8")
        print(f"schema_{org}.md: {len(text)} chars")

    test = json.loads(Path(args.exclude).read_text())
    rng = random.Random(args.seed)
    examples = []
    for org, tasks in [("b2b", TASKS_B2B), ("b2c", TASKS_B2C)]:
        excluded = set(test[org]["single_turn"]) | set(test[org]["multi_turn"])
        by_type = defaultdict(list)
        for t in tasks:
            if t["idx"] not in excluded:
                by_type[t["task"]].append(t)
        for task_type, rows in sorted(by_type.items()):
            for t in rng.sample(rows, min(args.per_type, len(rows))):
                context = (t["metadata"] or {}).get("required") or ""
                examples.append({"org": org, "task": task_type, "idx": t["idx"],
                                 "text": f"{t['query'].strip()}\n{context.strip()}".strip()})
    path = out / "router_examples.jsonl"
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + "\n", encoding="utf-8")
    print(f"{path}: {len(examples)} examples, {len({e['task'] for e in examples})} task types")



def export_router_eval(test_path="data/test.json", out_path="data/router_eval_test.jsonl"):
    """Test-split requests with their labels, for measuring router accuracy only.
    The agent never loads this file."""
    from crm_sandbox.data.assets import TASKS_B2B, TASKS_B2B_INTERACTIVE, TASKS_B2C, TASKS_B2C_INTERACTIVE
    test = json.loads(Path(test_path).read_text())
    rows = []
    for org, single, multi in [("b2b", TASKS_B2B, TASKS_B2B_INTERACTIVE), ("b2c", TASKS_B2C, TASKS_B2C_INTERACTIVE)]:
        for mode, tasks in [("single_turn", single), ("multi_turn", multi)]:
            wanted = set(test[org][mode])
            for t in tasks:
                if t["idx"] in wanted:
                    context = (t["metadata"] or {}).get("required") or ""
                    rows.append({"org": org, "mode": mode, "idx": t["idx"], "task": t["task"],
                                 "text": f"{t['query'].strip()}\n{context.strip()}".strip()})
    Path(out_path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"{out_path}: {len(rows)} test requests (evaluation only)")


if __name__ == "__main__":
    main()
    export_router_eval()
