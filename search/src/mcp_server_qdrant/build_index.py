"""Build the per-org hybrid knowledge collections from exported articles.

    crm-knowledge-index --knowledge-dir ../data/knowledge --qdrant-path ../data/qdrant

Reads <knowledge-dir>/<org>.jsonl (written by scripts/export_knowledge.py) and
creates collection knowledge_<org> in a local Qdrant store.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from mcp_server_qdrant.hybrid import HybridKnowledgeIndex, HybridSettings


async def build(knowledge_dir: Path, qdrant_path: str, orgs: list[str]) -> None:
    client = AsyncQdrantClient(path=qdrant_path)
    index = HybridKnowledgeIndex(client, HybridSettings())
    try:
        for org in orgs:
            src = knowledge_dir / f"{org}.jsonl"
            articles = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
            start = time.time()
            n = await index.build(index.collection_for(org), articles)
            print(f"{org}: {len(articles)} articles -> {n} chunks in {index.collection_for(org)} ({time.time() - start:.1f}s)")
    finally:
        await client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knowledge-dir", default="data/knowledge")
    ap.add_argument("--qdrant-path", default="data/qdrant")
    ap.add_argument("--orgs", default="b2b,b2c")
    args = ap.parse_args()
    asyncio.run(build(Path(args.knowledge_dir), args.qdrant_path, args.orgs.split(",")))


if __name__ == "__main__":
    main()
