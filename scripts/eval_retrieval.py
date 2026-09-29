"""Recall@k / MRR of knowledge search on the set from build_retrieval_set.py.

Compares dense, BM25, hybrid (RRF) and hybrid + rerank from the search fork, and
optionally the Salesforce MCP server's SOSL `search_knowledge` tool (--sosl-url).
Run with the search fork's environment:

  search/.venv/Scripts/python scripts/eval_retrieval.py --qdrant-path data/qdrant \
      --sosl-url http://127.0.0.1:3333/mcp
"""
import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from mcp_server_qdrant.hybrid import SEARCH_MODES, HybridKnowledgeIndex, HybridSettings


def rank_of(gold: set[str], ids: list[str]) -> int | None:
    for i, aid in enumerate(ids, 1):
        if aid in gold:
            return i
    return None


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


async def sosl_search(url: str, org: str, query: str, k: int) -> list[str]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, headers={"x-crm-org": org}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("search_knowledge", {"query": query, "top_k": k})
            if res.isError:
                return []
            return [a["Id"] for a in json.loads(res.content[0].text)["articles"]]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/retrieval/dev.jsonl")
    ap.add_argument("--qdrant-path", default="data/qdrant")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--modes", default=",".join(SEARCH_MODES))
    ap.add_argument("--sosl-url", default=None, help="Salesforce MCP server URL to include SOSL search")
    ap.add_argument("--out", default="runs/retrieval/dev_results.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    client = AsyncQdrantClient(path=args.qdrant_path)
    index = HybridKnowledgeIndex(client, HybridSettings())
    systems = args.modes.split(",") + (["sosl"] if args.sosl_url else [])
    ranks: dict[str, list[int | None]] = defaultdict(list)
    try:
        for i, row in enumerate(rows):
            gold = set(row["gold"])
            for system in systems:
                if system == "sosl":
                    # SOSL query terms are OR-joined keywords; long record text is cut to what SOSL accepts
                    ids = await sosl_search(args.sosl_url, row["org"], row["query"][:400], 10)
                else:
                    hits = await index.search(index.collection_for(row["org"]), row["query"], top_k=10, mode=system)
                    ids = [h.article_id for h in hits]
                ranks[system].append(rank_of(gold, ids))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)}", file=sys.stderr)
    finally:
        await client.close()

    groups = {"all": list(range(len(rows)))}
    for i, row in enumerate(rows):
        groups.setdefault(row["task"], []).append(i)
    report = {"n": len(rows), "k": args.k, "systems": {}}
    header = f"{'system':<14}" + "".join(f"{g[:26]:>28}" for g in groups)
    print(f"\nrecall@{args.k} (n) by task type; 'all' adds 95% bootstrap CI and MRR@10\n" + header)
    for system in systems:
        cells, report["systems"][system] = [], {}
        for g, idx in groups.items():
            hits = [1.0 if ranks[system][i] and ranks[system][i] <= args.k else 0.0 for i in idx]
            r1 = sum(1.0 for i in idx if ranks[system][i] == 1) / len(idx)
            mrr = sum(1.0 / ranks[system][i] for i in idx if ranks[system][i]) / len(idx)
            recall = sum(hits) / len(hits)
            entry = {"n": len(idx), f"recall@{args.k}": recall, "recall@1": r1, "mrr@10": mrr}
            if g == "all":
                lo, hi = bootstrap_ci(hits)
                entry["ci95"] = [lo, hi]
                cells.append(f"{recall:.3f} [{lo:.2f},{hi:.2f}] m{mrr:.2f}")
            else:
                cells.append(f"{recall:.3f} ({len(idx)})")
            report["systems"][system][g] = entry
        print(f"{system:<14}" + "".join(f"{c:>28}" for c in cells))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
