"""Offline regressions for the crmroute knowledge-search path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from qdrant_client import models

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.hybrid import HybridKnowledgeIndex, HybridSettings


def index_with_vectors():
    client = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True),
        query_points=AsyncMock(return_value=SimpleNamespace(points=[])),
    )
    index = HybridKnowledgeIndex(client, HybridSettings())
    index._dense = Mock()
    index._dense.query_embed.return_value = [np.array([0.25, 0.75])]
    index._sparse = Mock()
    index._sparse.query_embed.return_value = [
        SimpleNamespace(indices=np.array([1, 3]), values=np.array([0.5, 1.0]))
    ]
    return index


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sparse", "dense", "hybrid", "hybrid_dbsf"])
async def test_search_embeds_only_the_requested_modes(mode):
    index = index_with_vectors()
    await index.search("knowledge_b2b", "refund policy", mode=mode)
    assert index._dense.query_embed.call_count == (mode != "sparse")
    assert index._sparse.query_embed.call_count == (mode != "dense")
    kwargs = index.client.query_points.call_args.kwargs
    if mode == "sparse":
        assert kwargs["using"] == "bm25"
        assert isinstance(kwargs["query"], models.SparseVector)
    elif mode == "dense":
        assert kwargs["using"] == "dense"
    else:
        assert len(kwargs["prefetch"]) == 2


def test_generic_embedding_provider_does_not_load_a_model_until_used(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr("mcp_server_qdrant.embeddings.fastembed.TextEmbedding", constructor)
    provider = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
    constructor.assert_not_called()
    assert provider.embedding_model is constructor.return_value
    assert provider.embedding_model is constructor.return_value
    constructor.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")


def test_collection_selection_validates_org_and_keeps_orgs_separate():
    index = index_with_vectors()
    assert index.collection_for(" B2C ") == "knowledge_b2c"
    assert index.collection_for("b2b") == "knowledge_b2b"
    assert index.collection_for(None) == "knowledge_b2b"
    for org in ("", "prod", "../b2b", "b2b,b2c"):
        with pytest.raises(ValueError, match="org"):
            index.collection_for(org)


@pytest.mark.asyncio
async def test_article_collapse_keeps_best_chunk_and_order():
    index = index_with_vectors()
    index.client.query_points.return_value.points = [
        SimpleNamespace(score=score, payload={"article_id": aid, "title": aid, "document": chunk})
        for score, aid, chunk in [(4, "a", "best"), (3, "a", "second"), (2, "b", "third")]
    ]
    hits = await index.search("knowledge_b2b", "query", top_k=2)
    assert [(hit.article_id, hit.chunk) for hit in hits] == [("a", "best"), ("b", "third")]


@pytest.mark.asyncio
async def test_missing_collection_is_an_operational_error():
    index = index_with_vectors()
    index.client.collection_exists.return_value = False
    with pytest.raises(ValueError, match="collection"):
        await index.search("knowledge_b2c", "policy")
    index._sparse.query_embed.assert_not_called()


@pytest.mark.asyncio
async def test_empty_rebuild_does_not_delete_existing_collection():
    index = index_with_vectors()
    index.client.delete_collection = AsyncMock()
    with pytest.raises(ValueError, match="empty"):
        await index.build("knowledge_b2b", [])
    index.client.delete_collection.assert_not_called()


def test_truncated_article_is_marked_and_exposes_the_matched_passage():
    from mcp_server_qdrant.hybrid import Hit

    index = index_with_vectors()
    index.settings.max_article_chars = 20
    hit = Hit("a", "Policy", 1.0, "Important matching ending", {"body": "x" * 50})
    result = index.to_result(hit)
    assert result["truncated"] is True
    assert result["matched_passage"] == hit.chunk
    assert len(result["FAQ_Answer__c"]) <= 20


def test_knowledge_response_budget_counts_metadata_and_never_loses_truncation():
    import json
    from mcp_server_qdrant.hybrid import Hit

    index = index_with_vectors()
    index.settings.max_response_chars = 450
    hits = [Hit(str(i), "Policy", 1, "passage", {"body": "x" * 100}) for i in range(5)]
    response = index.to_response(hits)
    assert len(json.dumps(response)) <= 450
    assert response["truncated"] is True
    assert 0 < response["returned"] < len(hits)
    assert response["articles"][0]["Id"] == "0"
    assert response["note"]
    huge_title = Hit("a", "x" * 5000, 1, "passage", {})
    response = index.to_response([huge_title])
    assert response["returned"] == 0
    assert response["truncated"] is True
    assert len(json.dumps(response)) <= 450

