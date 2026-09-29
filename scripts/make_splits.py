"""Write fixed dev/test task-id lists for CRMArena-Pro.

Test (defaults): 3 tasks x 19 business task types x 2 orgs (114 single-turn),
1 x 19 x 2 multi-turn (38), and 7 x 3 confidentiality types x 2 orgs (42).
Dev is drawn from what is left, so no task id is in both lists for an org. The
single-turn and interactive splits share ids (same task, different phrasing), so
the lists are made disjoint across both modes.

Usage:  python scripts/make_splits.py --out-dir data --seed 20260927
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

CONFIDENTIALITY = ["private_customer_information", "internal_operation_data", "confidential_company_knowledge"]


def by_type(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row["idx"])
    for ids in grouped.values():
        ids.sort(key=int)
    return grouped


def take(rng, grouped, per_type, types, used):
    picked = []
    for task_type in sorted(types):
        pool = [i for i in grouped[task_type] if i not in used]
        chosen = rng.sample(pool, per_type)
        used.update(chosen)
        picked.extend(chosen)
    return sorted(picked, key=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--test-per-type", type=int, default=3)
    ap.add_argument("--test-mt-per-type", type=int, default=1)
    ap.add_argument("--test-conf-per-type", type=int, default=7)
    ap.add_argument("--dev-per-type", type=int, default=5)
    ap.add_argument("--dev-mt-per-type", type=int, default=2)
    ap.add_argument("--dev-conf-per-type", type=int, default=5)
    args = ap.parse_args()

    ds = load_dataset("Salesforce/CRMArenaPro", "CRMArenaPro")
    rng = random.Random(args.seed)
    splits = {"dev": {"seed": args.seed}, "test": {"seed": args.seed}}
    for org in ["b2b", "b2c"]:
        single = by_type(ds[org])
        multi = by_type(ds[f"{org}_interactive"])
        business = [t for t in single if t not in CONFIDENTIALITY]
        assert len(business) == 19, business
        used = set()  # shared across modes and splits for this org
        test_single = take(rng, single, args.test_per_type, business, used)
        test_conf = take(rng, single, args.test_conf_per_type, CONFIDENTIALITY, used)
        test_multi = take(rng, multi, args.test_mt_per_type, business, used)
        dev_single = take(rng, single, args.dev_per_type, business, used)
        dev_conf = take(rng, single, args.dev_conf_per_type, CONFIDENTIALITY, used)
        dev_multi = take(rng, multi, args.dev_mt_per_type, business, used)
        splits["test"][org] = {"single_turn": sorted(test_single + test_conf, key=int), "multi_turn": test_multi}
        splits["dev"][org] = {"single_turn": sorted(dev_single + dev_conf, key=int), "multi_turn": dev_multi}

    for name in ["dev", "test"]:
        split = splits[name]
        split["counts"] = {
            org: {mode: len(ids) for mode, ids in split[org].items()} for org in ["b2b", "b2c"]
        }
        path = Path(args.out_dir) / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(split, indent=1))
        print(name, split["counts"], "->", path)

    for org in ["b2b", "b2c"]:
        dev_ids = {i for ids in splits["dev"][org].values() if isinstance(ids, list) for i in ids}
        test_ids = {i for ids in splits["test"][org].values() if isinstance(ids, list) for i in ids}
        assert not dev_ids & test_ids, f"dev/test overlap in {org}"
    print("dev and test are disjoint per org")


if __name__ == "__main__":
    main()
