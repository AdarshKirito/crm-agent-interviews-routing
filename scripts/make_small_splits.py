"""Seeded, stratified subsets of dev.json / test.json for runs that must fit free-tier quotas.

test_small: per org, 1 task per business type + 2 per confidentiality type (50 single-turn)
dev_small:  per org, 1 task per business type + 1 per confidentiality type (44 single-turn)
dev_run:    every dev single-turn task + per org 10 multi-turn tasks, at most one per task
            type (220 + 20). Multi-turn agent tasks cost ~10x a single-turn task, and the
            routing fit is per task type, so dev measures multi-turn on a sample only.
Subsets are drawn from the existing lists, so test tasks stay held out of dev.

  vendor/CRMArena/.venv/Scripts/python scripts/make_small_splits.py
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

CONFIDENTIALITY = {"private_customer_information", "internal_operation_data", "confidential_company_knowledge"}


def subset(split: dict, ds, per_business: int, per_conf: int, seed: int) -> dict:
    rng = random.Random(seed)
    out = {"seed": seed, "source": split.get("source")}
    for org in ("b2b", "b2c"):
        types = {t["idx"]: t["task"] for t in ds[org]}
        by_type = defaultdict(list)
        for idx in split[org]["single_turn"]:
            by_type[types[idx]].append(idx)
        picked = []
        for task_type in sorted(by_type):
            k = per_conf if task_type in CONFIDENTIALITY else per_business
            picked += rng.sample(sorted(by_type[task_type]), min(k, len(by_type[task_type])))
        out[org] = {"single_turn": sorted(picked), "multi_turn": []}
    out["counts"] = {org: len(out[org]["single_turn"]) for org in ("b2b", "b2c")}
    return out


def multi_sample(split: dict, ds, per_org: int, seed: int) -> dict:
    rng = random.Random(seed)
    out = {"seed": seed, "source": split.get("source")}
    for org in ("b2b", "b2c"):
        types = {t["idx"]: t["task"] for t in ds[f"{org}_interactive"]}
        by_type = defaultdict(list)
        for idx in split[org]["multi_turn"]:
            by_type[types[idx]].append(idx)
        order = sorted(by_type)
        rng.shuffle(order)
        picked = [rng.choice(sorted(by_type[t])) for t in order[:per_org]]
        out[org] = {"single_turn": sorted(split[org]["single_turn"]), "multi_turn": sorted(picked)}
    out["counts"] = {org: {m: len(out[org][m]) for m in ("single_turn", "multi_turn")} for org in ("b2b", "b2c")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()
    ds = load_dataset("Salesforce/CRMArenaPro", "CRMArenaPro")
    for name, src, per_b, per_c in [("test_small", "test", 1, 2), ("dev_small", "dev", 1, 1)]:
        split = json.loads(Path(f"data/{src}.json").read_text())
        split["source"] = f"data/{src}.json"
        small = subset(split, ds, per_b, per_c, args.seed)
        Path(f"data/{name}.json").write_text(json.dumps(small, indent=1))
        print(name, small["counts"])
    dev = json.loads(Path("data/dev.json").read_text())
    dev["source"] = "data/dev.json"
    run = multi_sample(dev, ds, 10, args.seed)
    Path("data/dev_run.json").write_text(json.dumps(run, indent=1))
    print("dev_run", run["counts"])


if __name__ == "__main__":
    main()
