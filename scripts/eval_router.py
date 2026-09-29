"""Accuracy of the task-type router on test-split requests (offline, no LLM calls).

Two conditions: the full request + task context, and the task context alone (a
multi-turn user's first message can say almost nothing).
Run with the agent environment:  agent/.venv/Scripts/python scripts/eval_router.py
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from app.router import TaskRouter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/router_eval_test.jsonl")
    ap.add_argument("--out", default="runs/router/test_accuracy.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    router = TaskRouter()
    report = {}
    for condition in ["request+context", "context_only"]:
        correct, per_type, confusions, confidences = 0, defaultdict(lambda: [0, 0]), Counter(), []
        for r in rows:
            text = r["text"] if condition == "request+context" else r["text"][len(r["text"].split("\n")[0]):].strip() or r["text"]
            pred = router.predict(text)
            ok = pred.task_type == r["task"]
            correct += ok
            per_type[r["task"]][0] += ok
            per_type[r["task"]][1] += 1
            confidences.append((pred.confidence, ok))
            if not ok:
                confusions[(r["task"], pred.task_type)] += 1
        acc = correct / len(rows)
        low = [ok for c, ok in confidences if c < 0.5]
        report[condition] = {
            "n": len(rows),
            "accuracy": acc,
            "per_type": {t: c / n for t, (c, n) in sorted(per_type.items())},
            "top_confusions": [[a, b, n] for (a, b), n in confusions.most_common(8)],
            "low_confidence_share": len(low) / len(rows),
            "accuracy_when_low_confidence": (sum(low) / len(low)) if low else None,
        }
        print(f"\n[{condition}] accuracy {acc:.3f} on {len(rows)} test requests; "
              f"{len(low)} below 0.5 confidence")
        for (a, b), n in confusions.most_common(8):
            print(f"   {a} -> {b}: {n}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
