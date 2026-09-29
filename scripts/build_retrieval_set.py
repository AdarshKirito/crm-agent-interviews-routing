"""Build a knowledge-retrieval evaluation set from CRMArena-Pro tasks outside test.json.

Gold labels:
- policy_violation_identification / quote_approval / invalid_config: the task answer
  is a knowledge-article Id (tasks whose answer is None are skipped). The search query
  is built from the record the task points at: the case's subject + description, or
  the quote's name + line items (product, quantity, discount, price).
- knowledge_qa (only with --include-silver): the answer is free text, so the gold article is a *silver* label: the
  article whose text contains the largest share of the reference answer's content
  words, kept only when that share is >= --min-support and clearly beats the runner-up.
  The query is the task question. A manual check of 6 such labels found none whose
  article actually contained the answer, so they are excluded by default.

Usage:  python scripts/build_retrieval_set.py --env vendor/CRMArena/.env --exclude data/test.json
"""
import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from simple_salesforce import Salesforce

ID_TASKS = {"policy_violation_identification": "Case", "quote_approval": "Quote", "invalid_config": "Quote"}
STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with".split())


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in STOP and len(w) > 2]


def connect(org: str) -> Salesforce:
    p = {"b2b": "SALESFORCE_B2B_", "b2c": "SALESFORCE_B2C_"}[org]
    return Salesforce(username=os.environ[p + "USERNAME"], password=os.environ[p + "PASSWORD"],
                      security_token=os.environ[p + "SECURITY_TOKEN"])


def batched(ids, n=150):
    ids = sorted(set(ids))
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def quoted(ids):
    return ",".join(f"'{i}'" for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="vendor/CRMArena/.env")
    ap.add_argument("--exclude", default="data/test.json")
    ap.add_argument("--knowledge-dir", default="data/knowledge")
    ap.add_argument("--out", default="data/retrieval/dev.jsonl")
    ap.add_argument("--min-support", type=float, default=0.6)
    ap.add_argument("--include-silver", action="store_true",
                    help="also write knowledge_qa rows with silver labels (off: a manual check found them unreliable)")
    args = ap.parse_args()
    load_dotenv(args.env)
    test = json.loads(Path(args.exclude).read_text())
    ds = load_dataset("Salesforce/CRMArenaPro", "CRMArenaPro")
    rows, skipped = [], {"none_answer": 0, "no_record": 0, "weak_silver": 0}

    for org in ["b2b", "b2c"]:
        excluded = set(test[org]["single_turn"]) | set(test[org]["multi_turn"])
        tasks = [t for t in ds[org] if t["idx"] not in excluded and (t["task"] in ID_TASKS or t["task"] == "knowledge_qa")]
        sf = connect(org)
        articles = [json.loads(l) for l in (Path(args.knowledge_dir) / f"{org}.jsonl").read_text(encoding="utf-8").splitlines()]
        art_words = {a["Id"]: set(words(" ".join(filter(None, [a["Title"], a["Summary"], a["FAQ_Answer__c"]])))) for a in articles}

        ref_ids = {"Case": [], "Quote": []}
        for t in tasks:
            if t["task"] in ID_TASKS:
                m = re.search(r"(?:Case|Quote) Id to be considered is:\s*([A-Za-z0-9]{15,18})", t["metadata"]["required"] or "")
                if m:
                    ref_ids[ID_TASKS[t["task"]]].append(m.group(1))
        cases, quote_text = {}, {}
        for chunk in batched(ref_ids["Case"]):
            for r in sf.query_all(f"SELECT Id, Subject, Description FROM Case WHERE Id IN ({quoted(chunk)})")["records"]:
                cases[r["Id"]] = f"{r['Subject'] or ''}. {r['Description'] or ''}".strip()
        for chunk in batched(ref_ids["Quote"]):
            for r in sf.query_all(f"SELECT Id, Name, Description FROM Quote WHERE Id IN ({quoted(chunk)})")["records"]:
                quote_text[r["Id"]] = [f"{r['Name'] or ''} {r['Description'] or ''}".strip()]
            items = sf.query_all(
                "SELECT QuoteId, Product2.Name, Quantity, Discount, UnitPrice FROM QuoteLineItem "
                f"WHERE QuoteId IN ({quoted(chunk)})")["records"]
            for it in items:
                name = (it.get("Product2") or {}).get("Name") or "product"
                quote_text.setdefault(it["QuoteId"], []).append(
                    f"{name}: quantity {it['Quantity']}, discount {it['Discount'] or 0}%, unit price {it['UnitPrice']}")

        for t in tasks:
            answer = t["answer"][0] if t["answer"] else None
            base = {"org": org, "task_id": t["idx"], "task": t["task"]}
            if t["task"] in ID_TASKS:
                if not answer or answer == "None":
                    skipped["none_answer"] += 1
                    continue
                m = re.search(r"(?:Case|Quote) Id to be considered is:\s*([A-Za-z0-9]{15,18})", t["metadata"]["required"] or "")
                ref = m.group(1) if m else None
                text = cases.get(ref) if ID_TASKS[t["task"]] == "Case" else "; ".join(quote_text.get(ref, []))
                if not text:
                    skipped["no_record"] += 1
                    continue
                rows.append({**base, "query": text[:1500], "gold": [answer], "label": "gold"})
            elif args.include_silver:
                ans = set(words(answer))
                if not ans:
                    skipped["weak_silver"] += 1
                    continue
                support = sorted(((len(ans & w) / len(ans), aid) for aid, w in art_words.items()), reverse=True)
                best, runner = support[0], support[1]
                if best[0] < args.min_support or best[0] - runner[0] < 0.1:
                    skipped["weak_silver"] += 1
                    continue
                rows.append({**base, "query": t["query"], "gold": [best[1]], "label": "silver", "support": round(best[0], 3)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    from collections import Counter
    print(f"{len(rows)} queries -> {out}")
    print("by task:", dict(Counter((r["org"], r["task"]) for r in rows)))
    print("skipped:", skipped)


if __name__ == "__main__":
    main()
