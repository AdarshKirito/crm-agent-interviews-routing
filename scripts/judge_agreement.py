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
import hashlib
import json
import math
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


def fingerprint(row: dict) -> str:
    payload = [row.get("task_type"), question_of(row), row.get("gt_answer"), response_of(row), row.get("reward")]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def make_sheet(systems: list[str], n: int, out: str, seed: int) -> None:
    meta = Path(out).with_suffix(".systems.json")
    if Path(out).exists() or meta.exists():
        raise ValueError("label sheet or metadata already exists; use a new --out to preserve human work")
    if n < 2:
        raise ValueError("sample at least two responses")
    pool = []
    for spec in systems:
        name, path = spec.split("=", 1)
        for key, row in load_system(Path(path)).items():
            if isinstance(row.get("reward"), dict) or row.get("error") or row.get("reward") is None:
                continue  # fuzzy_match is not judged by the LLM; API errors were never graded
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
    if len(picked) < n:
        raise ValueError(f"requested {n} labels, but only {len(picked)} graded responses are available")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for name, key, row in picked:
            w.writerow({"key": "|".join(map(str, key)), "system": name, "task_type": row["task_type"],
                        "question": question_of(row), "reference_answer": json.dumps(row.get("gt_answer")),
                        "agent_response": response_of(row), "human_correct": "", "notes": ""})
    # Keep verdicts out of the human-facing CSV and bind labels to these exact responses.
    meta.write_text(json.dumps({"version": 2, "systems": systems, "items": [
        {"system": name, "key": "|".join(map(str, key)), "fingerprint": fingerprint(row)}
        for name, key, row in picked
    ]}, indent=2), encoding="utf-8")
    print(f"wrote {len(picked)} rows to {out} (fill human_correct with 1/0)")


def kappa(labels: str, min_labels: int = 40, min_kappa: float = 0.7) -> int:
    from sklearn.metrics import cohen_kappa_score

    metadata = json.loads(Path(labels).with_suffix(".systems.json").read_text(encoding="utf-8"))
    systems = metadata if isinstance(metadata, list) else metadata["systems"]
    snapshots = {} if isinstance(metadata, list) else {
        (r["system"], r["key"]): r["fingerprint"] for r in metadata["items"]
    }
    rows = {}
    for spec in systems:
        name, path = spec.split("=", 1)
        for key, row in load_system(Path(path)).items():
            rows[(name, "|".join(map(str, key)))] = row
    human, judge, disagreements, seen = [], [], [], set()
    with open(labels, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["system"], r["key"])
            if key in seen:
                raise ValueError(f"duplicate labelled item: {key}")
            seen.add(key)
            row = rows.get(key)
            if row is None or row.get("error") or isinstance(row.get("reward"), dict) or row.get("reward") is None:
                raise ValueError(f"labelled source is missing, errored or no longer judge-scored: {key}")
            if (r["question"] != question_of(row) or r["agent_response"] != response_of(row)
                    or r["reference_answer"] != json.dumps(row.get("gt_answer")) or r["task_type"] != row["task_type"]):
                raise ValueError(f"source response changed since the sheet was generated: {key}")
            if snapshots and snapshots.get(key) != fingerprint(row):
                raise ValueError(f"source grading changed since the sheet was generated: {key}")
            value = r["human_correct"].strip()
            if not value:
                continue
            if value not in ("0", "1"):
                raise ValueError(f"human_correct must be 0 or 1: {key}")
            h, j = int(value), int(float(row.get("reward") or 0) >= 1)
            human.append(h)
            judge.append(j)
            if h != j:
                disagreements.append((r["system"], r["key"], r["task_type"], h, j))
    if len(human) < min_labels or len(human) != len(seen):
        print(f"FAIL: {len(human)}/{len(seen)} rows labelled; finish the sheet (minimum {min_labels})")
        return 1
    k = cohen_kappa_score(human, judge)
    agree = sum(h == j for h, j in zip(human, judge)) / len(human)
    print(f"labelled rows: {len(human)}  raw agreement: {agree:.3f}  Cohen's kappa: {k:.3f}  (target >= {min_kappa})")
    for d in disagreements:
        print("  disagreement:", d)
    if not math.isfinite(k) or k < min_kappa:
        print("FAIL: judge agreement is below target or undefined")
        return 1
    return 0


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
    k.add_argument("--min-labels", type=int, default=40)
    k.add_argument("--min-kappa", type=float, default=0.7)
    args = ap.parse_args()
    try:
        if args.cmd == "sheet":
            make_sheet(args.system, args.n, args.out, args.seed)
        else:
            if args.min_labels < 2 or not -1 <= args.min_kappa <= 1:
                ap.error("--min-labels must be >= 2 and --min-kappa in [-1, 1]")
            sys.exit(kappa(args.labels, args.min_labels, args.min_kappa))
    except ValueError as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
