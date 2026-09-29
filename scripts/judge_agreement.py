"""Check the grading model against human labels (Cohen's kappa).

1) Make a blind labelling sheet (the judge's verdicts are NOT in it):
     python scripts/judge_agreement.py sheet --system full=runs/test/full --system react=runs/test/react \
         --n 40 --out evals/human_labels.csv
   Fill the `human_correct` column with 1 (the response answers the task correctly /
   refuses when it should) or 0. Use the reference answer as the key.
2) Score agreement:
     python scripts/judge_agreement.py kappa --labels evals/human_labels.csv
   Target: kappa >= 0.7. If lower, fix the judge prompt and re-grade.

Only exact_match and privacy_rejection tasks are sampled: fuzzy_match tasks are
scored by token overlap, not by the judge.
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_results import load_system  # noqa: E402

FIELDS = ["key", "system", "task_type", "question", "reference_answer", "agent_response", "human_correct", "notes"]


def question_of(row: dict) -> str:
    traj = row.get("traj") or []
    for m in traj:
        if m.get("role") == "user":
            return str(m.get("content"))[-1500:]
    return ""


def response_of(row: dict) -> str:
    info = row.get("agent_info") or {}
    end = info.get("end_reason") or {}
    if end.get("content"):
        return str(end["content"])
    traj = row.get("traj") or []
    for m in reversed(traj):
        if m.get("role") == "assistant":
            return str(m.get("content"))
    return ""


def make_sheet(systems: list[str], n: int, out: str, seed: int) -> None:
    pool = []
    for spec in systems:
        name, path = spec.split("=", 1)
        for key, row in load_system(Path(path)).items():
            if isinstance(row.get("reward"), dict):
                continue  # fuzzy_match: not judged by the LLM
            pool.append((name, key, row))
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for item in pool:
        by_type[item[2]["task_type"]].append(item)
    picked = []
    while len(picked) < n and any(by_type.values()):  # round-robin over task types
        for t in sorted(by_type):
            if by_type[t] and len(picked) < n:
                picked.append(by_type[t].pop(rng.randrange(len(by_type[t]))))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for name, key, row in picked:
            w.writerow({"key": "|".join(map(str, key)), "system": name, "task_type": row["task_type"],
                        "question": question_of(row), "reference_answer": json.dumps(row.get("gt_answer")),
                        "agent_response": response_of(row), "human_correct": "", "notes": ""})
    meta = Path(out).with_suffix(".systems.json")
    meta.write_text(json.dumps(systems))
    print(f"wrote {len(picked)} rows to {out} (fill human_correct with 1/0)")


def kappa(labels: str) -> None:
    from sklearn.metrics import cohen_kappa_score

    systems = json.loads(Path(labels).with_suffix(".systems.json").read_text())
    rows = {}
    for spec in systems:
        name, path = spec.split("=", 1)
        for key, row in load_system(Path(path)).items():
            rows[(name, "|".join(map(str, key)))] = row
    human, judge, disagreements = [], [], []
    with open(labels, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["human_correct"].strip() not in ("0", "1"):
                continue
            row = rows[(r["system"], r["key"])]
            h, j = int(r["human_correct"]), int(float(row.get("reward") or 0) >= 1)
            human.append(h)
            judge.append(j)
            if h != j:
                disagreements.append((r["system"], r["key"], r["task_type"], h, j))
    if len(human) < 2:
        sys.exit("no labelled rows yet")
    k = cohen_kappa_score(human, judge)
    agree = sum(h == j for h, j in zip(human, judge)) / len(human)
    print(f"labelled rows: {len(human)}  raw agreement: {agree:.3f}  Cohen's kappa: {k:.3f}  (target >= 0.7)")
    for d in disagreements:
        print("  disagreement:", d)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--system", action="append", required=True)
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=40)
    s.add_argument("--out", default="evals/human_labels.csv")
    k = sub.add_parser("kappa")
    k.add_argument("--labels", default="evals/human_labels.csv")
    args = ap.parse_args()
    if args.cmd == "sheet":
        make_sheet(args.system, args.n, args.out, args.seed)
    else:
        kappa(args.labels)


if __name__ == "__main__":
    main()
