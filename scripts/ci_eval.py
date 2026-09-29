"""Pull-request eval: 15 fixed dev cases on the small model, compared with a stored baseline.

  python scripts/ci_eval.py make-cases            # writes evals/ci_cases.json from data/dev.json (once)
  python scripts/ci_eval.py run --out runs/ci      # runs the cases through the benchmark harness
  python scripts/ci_eval.py compare --run runs/ci  # fails (exit 1) if success or refusal rate drops
  python scripts/ci_eval.py compare --run runs/ci --update-baseline   # accept a new baseline

Case mix: 9 single-turn business tasks, 3 confidentiality (refusal) tasks and 3
multi-turn tasks, drawn with a fixed seed from the dev split. The agent server must
run with CRMROUTE_BIG_MODEL=gemini-3.1-flash-lite so every step uses the small model.
"""
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_results import CONFIDENTIALITY, load_system, score  # noqa: E402

CASES = ROOT / "evals" / "ci_cases.json"
BASELINE = ROOT / "evals" / "ci_baseline.json"


def make_cases(seed: int = 15) -> None:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/CRMArenaPro", "CRMArenaPro")
    dev = json.loads((ROOT / "data" / "dev.json").read_text())
    rng = random.Random(seed)
    types = {org: {t["idx"]: t["task"] for t in ds[org]} for org in ("b2b", "b2c")}
    cases = {org: {"single_turn": [], "multi_turn": []} for org in ("b2b", "b2c")}
    business = [(org, i) for org in ("b2b", "b2c") for i in dev[org]["single_turn"] if types[org][i] not in CONFIDENTIALITY]
    confidential = [(org, i) for org in ("b2b", "b2c") for i in dev[org]["single_turn"] if types[org][i] in CONFIDENTIALITY]
    multi = [(org, i) for org in ("b2b", "b2c") for i in dev[org]["multi_turn"]]
    chosen_types = set()
    for org, i in rng.sample(business, len(business)):  # spread over task types
        if len(cases["b2b"]["single_turn"]) + len(cases["b2c"]["single_turn"]) >= 9:
            break
        if types[org][i] not in chosen_types:
            chosen_types.add(types[org][i])
            cases[org]["single_turn"].append(i)
    for org, i in rng.sample(confidential, 3):
        cases[org]["single_turn"].append(i)
    for org, i in rng.sample(multi, 3):
        cases[org]["multi_turn"].append(i)
    CASES.parent.mkdir(parents=True, exist_ok=True)
    CASES.write_text(json.dumps({"seed": seed, **cases}, indent=1))
    print(f"wrote {CASES}: {sum(len(v) for o in ('b2b', 'b2c') for v in cases[o].values())} cases")


def run(out: str, url: str) -> None:
    bench = ROOT / "vendor" / "CRMArena"
    python = bench / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
    for org in ("b2b", "b2c"):
        for interactive in (False, True):
            cmd = [str(python), "-u", "run_tasks.py", "--agent_strategy", "remote", "--model", "crmroute-ci",
                   "--remote_url", url, "--task_category", "all", "--task_ids_file", str(CASES), "--org_type", org,
                   "--agent_eval_mode", "aided", "--log_dir", str(Path(out).resolve()),
                   "--judge_model", os.environ["CRMARENA_JUDGE_MODEL"], "--judge_provider", os.environ["CRMARENA_JUDGE_PROVIDER"],
                   "--user_model", os.getenv("CRMARENA_USER_MODEL", os.environ["CRMARENA_JUDGE_MODEL"]),
                   "--user_provider", os.getenv("CRMARENA_USER_PROVIDER", os.environ["CRMARENA_JUDGE_PROVIDER"])]
            if interactive:
                cmd += ["--interactive", "--max_user_turns", "6"]
            subprocess.run(cmd, cwd=bench, check=True)


def rates(run_dir: str) -> dict:
    rows = load_system(Path(run_dir))
    business = [score(r, 0.5)[1] for r in rows.values() if r["task_type"] not in CONFIDENTIALITY]
    refusals = [score(r, 0.5)[1] for r in rows.values() if r["task_type"] in CONFIDENTIALITY]
    return {
        "n": len(rows),
        "success_rate": sum(business) / len(business) if business else 0.0,
        "refusal_rate": sum(refusals) / len(refusals) if refusals else 0.0,
    }


def compare(run_dir: str, update: bool, tolerance: float) -> int:
    current = rates(run_dir)
    print("current:", json.dumps(current))
    if update or not BASELINE.exists():
        BASELINE.write_text(json.dumps(current, indent=1))
        print(f"baseline written to {BASELINE}")
        return 0
    base = json.loads(BASELINE.read_text())
    print("baseline:", json.dumps(base))
    failed = []
    for key in ("success_rate", "refusal_rate"):
        if current[key] < base[key] - tolerance:
            failed.append(f"{key} dropped from {base[key]:.2f} to {current[key]:.2f}")
    for msg in failed:
        print("FAIL:", msg)
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-cases")
    r = sub.add_parser("run")
    r.add_argument("--out", default="runs/ci")
    r.add_argument("--url", default=os.getenv("CRMROUTE_URL", "http://127.0.0.1:8000"))
    c = sub.add_parser("compare")
    c.add_argument("--run", default="runs/ci")
    c.add_argument("--update-baseline", action="store_true")
    # one of 15 cases is ~0.07-0.11 of a rate; allow one case of noise before failing
    c.add_argument("--tolerance", type=float, default=0.12)
    args = ap.parse_args()
    if args.cmd == "make-cases":
        make_cases()
    elif args.cmd == "run":
        run(args.out, args.url)
    else:
        sys.exit(compare(args.run, args.update_baseline, args.tolerance))


if __name__ == "__main__":
    main()
