# Fork notice

This directory is a fork of [qdrant/mcp-server-qdrant](https://github.com/qdrant/mcp-server-qdrant)
(Apache-2.0, see LICENSE) at commit `c56ae5adf62bb78d852bf7bbcbc5d7b75e2bbe41`.

Changes for crmroute:

- `src/mcp_server_qdrant/hybrid.py` (new): per-article chunking, a dense vector
  (`BAAI/bge-small-en-v1.5`) and an IDF-weighted BM25 sparse vector (`Qdrant/bm25`)
  per chunk; search modes `dense`, `sparse`, `hybrid` (RRF), `hybrid_dbsf` (DBSF) and
  `hybrid_rerank` (RRF + FastEmbed cross-encoder); results collapsed to one hit per article.
- `src/mcp_server_qdrant/build_index.py` (new): `crm-knowledge-index` CLI that builds
  `knowledge_<org>` collections from exported `Knowledge__kav` rows.
- `src/mcp_server_qdrant/mcp_server.py`: optional read-only `search_knowledge` tool
  (`HYBRID_ENABLED=true`) that picks the collection from the `x-crm-org` request
  header; `HYBRID_ONLY=true` hides the original `qdrant-find`/`qdrant-store` tools.
  With `HYBRID_ENABLED` unset the server behaves as upstream (its tests still pass).
- `pyproject.toml`: version `0.8.1+crmroute.1` and the new CLI entry point.
- `.python-version`: 3.12. `kiro-power/` and the `.github` workflows were removed (the
  monorepo CI runs this test suite).

The default search mode is `sparse` (BM25) because it had the highest recall@5 on the
dev retrieval set (see the project README, "Knowledge search").
