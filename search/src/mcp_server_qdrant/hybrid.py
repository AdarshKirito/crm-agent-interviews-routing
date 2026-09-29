"""Hybrid (dense + BM25) search with reciprocal-rank fusion and cross-encoder reranking.

Added to the mcp-server-qdrant fork for crmroute. Articles are split into
overlapping word windows, each prefixed with the article title. A query runs a
dense (bge-small) and a sparse (Qdrant/bm25, IDF-weighted) search, Qdrant fuses
the two rankings with RRF, a FastEmbed cross-encoder reranks the fused pool, and
results are collapsed to one hit per article (its best chunk).
"""
import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Literal

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pydantic import Field
from pydantic_settings import BaseSettings
from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger(__name__)

SearchMode = Literal["dense", "sparse", "hybrid", "hybrid_dbsf", "hybrid_rerank"]
SEARCH_MODES: tuple[str, ...] = ("dense", "sparse", "hybrid", "hybrid_dbsf", "hybrid_rerank")

DENSE = "dense"
SPARSE = "bm25"
ID_NAMESPACE = uuid.UUID("5d0c7f3e-8a51-4f3c-9d1e-2b7a1c0e6f42")


class HybridSettings(BaseSettings):
    enabled: bool = Field(default=False, validation_alias="HYBRID_ENABLED")
    only: bool = Field(default=True, validation_alias="HYBRID_ONLY")
    tool_description: str = Field(
        default=(
            "Search the company's knowledge articles (policies, product guides, competitor notes) by meaning and "
            "keywords. Returns the best-matching articles with Id, Title, Summary and full text."
        ),
        validation_alias="HYBRID_TOOL_DESCRIPTION",
    )
    dense_model: str = Field(default="BAAI/bge-small-en-v1.5", validation_alias="HYBRID_DENSE_MODEL")
    sparse_model: str = Field(default="Qdrant/bm25", validation_alias="HYBRID_SPARSE_MODEL")
    rerank_model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2", validation_alias="HYBRID_RERANK_MODEL")
    candidates: int = Field(default=40, validation_alias="HYBRID_CANDIDATES")
    rerank_pool: int = Field(default=24, validation_alias="HYBRID_RERANK_POOL")
    chunk_words: int = Field(default=180, validation_alias="HYBRID_CHUNK_WORDS")
    chunk_overlap: int = Field(default=40, validation_alias="HYBRID_CHUNK_OVERLAP")
    collection_prefix: str = Field(default="knowledge_", validation_alias="HYBRID_COLLECTION_PREFIX")
    default_org: str = Field(default="b2b", validation_alias="CRM_DEFAULT_ORG")
    max_article_chars: int = Field(default=3500, validation_alias="HYBRID_MAX_ARTICLE_CHARS")
    # Chosen on the dev retrieval set (scripts/eval_retrieval.py); see README "Knowledge search".
    search_mode: str = Field(default="sparse", validation_alias="HYBRID_SEARCH_MODE")


@dataclass
class Hit:
    article_id: str
    title: str
    score: float
    chunk: str
    payload: dict


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def article_chunks(article: dict, size: int, overlap: int) -> list[str]:
    title = (article.get("Title") or "").strip()
    body = " ".join(
        part.strip() for part in (article.get("Summary"), article.get("FAQ_Answer__c")) if part and part.strip()
    )
    return [f"{title}. {chunk}" if title else chunk for chunk in chunk_words(body, size, overlap)]


class HybridKnowledgeIndex:
    def __init__(self, client: AsyncQdrantClient, settings: HybridSettings | None = None):
        self.client = client
        self.settings = settings or HybridSettings()
        self._dense: TextEmbedding | None = None
        self._sparse: SparseTextEmbedding | None = None
        self._reranker: TextCrossEncoder | None = None

    # Models load lazily: the first query pays the load time, not server start-up.
    @property
    def dense(self) -> TextEmbedding:
        if self._dense is None:
            self._dense = TextEmbedding(self.settings.dense_model)
        return self._dense

    @property
    def sparse(self) -> SparseTextEmbedding:
        if self._sparse is None:
            self._sparse = SparseTextEmbedding(self.settings.sparse_model)
        return self._sparse

    @property
    def reranker(self) -> TextCrossEncoder | None:
        if self._reranker is None and self.settings.rerank_model:
            self._reranker = TextCrossEncoder(self.settings.rerank_model)
        return self._reranker

    def collection_for(self, org: str | None) -> str:
        return f"{self.settings.collection_prefix}{(org or self.settings.default_org).lower()}"

    async def build(self, collection: str, articles: list[dict]) -> int:
        """(Re)create `collection` from exported Knowledge__kav rows. Point ids are
        derived from article id + chunk number, so rebuilding is deterministic."""
        texts, payloads, ids = [], [], []
        for article in articles:
            for n, chunk in enumerate(article_chunks(article, self.settings.chunk_words, self.settings.chunk_overlap)):
                texts.append(chunk)
                ids.append(str(uuid.uuid5(ID_NAMESPACE, f"{article['Id']}:{n}")))
                payloads.append(
                    {
                        "document": chunk,
                        "article_id": article["Id"],
                        "chunk": n,
                        "title": article.get("Title"),
                        "summary": article.get("Summary"),
                        "url_name": article.get("UrlName"),
                        "body": article.get("FAQ_Answer__c"),
                    }
                )
        loop = asyncio.get_running_loop()
        dense_vecs = await loop.run_in_executor(None, lambda: list(self.dense.passage_embed(texts)))
        sparse_vecs = await loop.run_in_executor(None, lambda: list(self.sparse.passage_embed(texts)))

        if await self.client.collection_exists(collection):
            await self.client.delete_collection(collection)
        await self.client.create_collection(
            collection_name=collection,
            vectors_config={DENSE: models.VectorParams(size=len(dense_vecs[0]), distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        points = [
            models.PointStruct(
                id=pid,
                vector={
                    DENSE: d.tolist(),
                    SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                },
                payload=p,
            )
            for pid, d, s, p in zip(ids, dense_vecs, sparse_vecs, payloads)
        ]
        for start in range(0, len(points), 256):
            await self.client.upsert(collection_name=collection, points=points[start : start + 256])
        return len(points)

    async def _embed_query(self, query: str):
        loop = asyncio.get_running_loop()
        dense = await loop.run_in_executor(None, lambda: next(iter(self.dense.query_embed([query]))))
        sparse = await loop.run_in_executor(None, lambda: next(iter(self.sparse.query_embed([query]))))
        return dense.tolist(), models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())

    async def search(self, collection: str, query: str, top_k: int = 5, mode: SearchMode | None = None) -> list[Hit]:
        mode = mode or self.settings.search_mode
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}")
        if not await self.client.collection_exists(collection):
            return []
        dense, sparse = await self._embed_query(query)
        n = self.settings.candidates
        if mode == "dense":
            res = await self.client.query_points(collection, query=dense, using=DENSE, limit=n, with_payload=True)
        elif mode == "sparse":
            res = await self.client.query_points(collection, query=sparse, using=SPARSE, limit=n, with_payload=True)
        else:
            res = await self.client.query_points(
                collection,
                prefetch=[
                    models.Prefetch(query=dense, using=DENSE, limit=n),
                    models.Prefetch(query=sparse, using=SPARSE, limit=n),
                ],
                query=models.FusionQuery(fusion=models.Fusion.DBSF if mode == "hybrid_dbsf" else models.Fusion.RRF),
                limit=n,
                with_payload=True,
            )
        ranked = [(float(p.score), p.payload or {}) for p in res.points]
        if mode == "hybrid_rerank" and self.reranker is not None and ranked:
            pool = ranked[: self.settings.rerank_pool]
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                None, lambda: list(self.reranker.rerank(query, [p["document"] for _, p in pool]))
            )
            ranked = sorted(((float(s), p) for s, (_, p) in zip(scores, pool)), key=lambda x: x[0], reverse=True)
        hits: list[Hit] = []
        seen: set[str] = set()
        for score, payload in ranked:
            article_id = payload.get("article_id")
            if not article_id or article_id in seen:
                continue
            seen.add(article_id)
            hits.append(Hit(article_id, payload.get("title") or "", score, payload.get("document") or "", payload))
            if len(hits) >= top_k:
                break
        return hits

    def to_result(self, hit: Hit) -> dict:
        body = hit.payload.get("body") or ""
        if len(body) > self.settings.max_article_chars:
            body = body[: self.settings.max_article_chars] + " ..."
        return {
            "Id": hit.article_id,
            "Title": hit.title,
            "Summary": hit.payload.get("summary"),
            "FAQ_Answer__c": body,
            "score": round(hit.score, 4),
        }
