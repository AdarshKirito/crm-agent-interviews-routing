"""One-time export of the knowledge articles (Knowledge__kav) from the CRMArena-Pro orgs.

Writes data/knowledge/<org>.jsonl with one article per line. Uses the benchmark's
own credentials (.env with SALESFORCE_B2B_* / SALESFORCE_B2C_*).

Usage:  python scripts/export_knowledge.py --env vendor/CRMArena/.env --out-dir data/knowledge
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from simple_salesforce import Salesforce

FIELDS = ["Id", "Title", "UrlName", "Summary", "FAQ_Answer__c", "LastModifiedDate"]


def connect(org: str) -> Salesforce:
    prefix = {"b2b": "SALESFORCE_B2B_", "b2c": "SALESFORCE_B2C_"}[org]
    return Salesforce(
        username=os.environ[f"{prefix}USERNAME"],
        password=os.environ[f"{prefix}PASSWORD"],
        security_token=os.environ[f"{prefix}SECURITY_TOKEN"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="vendor/CRMArena/.env")
    ap.add_argument("--out-dir", default="data/knowledge")
    ap.add_argument("--orgs", default="b2b,b2c")
    args = ap.parse_args()
    load_dotenv(args.env)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for org in args.orgs.split(","):
        sf = connect(org)
        records = sf.query_all(f"SELECT {', '.join(FIELDS)} FROM Knowledge__kav")["records"]
        path = out_dir / f"{org}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({k: r.get(k) for k in FIELDS}, ensure_ascii=False) + "\n")
        lengths = sorted(len((r.get("FAQ_Answer__c") or "").split()) for r in records)
        print(f"{org}: {len(records)} articles -> {path}; words per article "
              f"min={lengths[0]} median={lengths[len(lengths) // 2]} max={lengths[-1]}")


if __name__ == "__main__":
    main()
