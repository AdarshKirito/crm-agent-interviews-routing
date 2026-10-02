"""Exercise the read-only CRM tool through the MCP protocol without models/network."""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastmcp import Client

from mcp_server_qdrant.hybrid import Hit, HybridSettings
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import EmbeddingProviderSettings, QdrantSettings, ToolSettings


@pytest.mark.asyncio
async def test_knowledge_mcp_org_binding_and_read_only_tools(monkeypatch):
    constructor = Mock(side_effect=AssertionError("Unused dense model must not load"))
    monkeypatch.setattr("mcp_server_qdrant.embeddings.fastembed.TextEmbedding", constructor)
    server = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(QDRANT_URL=":memory:", QDRANT_READ_ONLY=True),
        embedding_provider_settings=EmbeddingProviderSettings(),
        hybrid_settings=HybridSettings(HYBRID_ENABLED=True, HYBRID_ONLY=True),
    )
    server.hybrid_index.search = AsyncMock(return_value=[Hit(
        "article1", "Policy", 1.0, "Matched passage", {"body": "Policy text"}
    )])
    headers = {"x-crm-org": "b2c"}
    monkeypatch.setattr("mcp_server_qdrant.mcp_server.get_http_headers", lambda **kwargs: headers)
    try:
        async with Client(server) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools] == ["search_knowledge"]
            assert tools[0].annotations.readOnlyHint is True
            result = await client.call_tool_mcp("search_knowledge", {"query": "refunds"})
            assert not result.isError
            payload = json.loads(result.content[0].text)
            assert payload["returned"] == 1
            assert payload["articles"][0]["Id"] == "article1"
            server.hybrid_index.search.assert_awaited_once_with("knowledge_b2c", "refunds", top_k=5)
            headers["x-crm-org"] = "other-org"
            result = await client.call_tool_mcp("search_knowledge", {"query": "refunds"})
            assert result.isError
            assert "Unknown org" in result.content[0].text
            assert server.hybrid_index.search.await_count == 1
            headers["x-crm-org"] = "b2b"
            server.hybrid_index.search.side_effect = ValueError("Knowledge collection is missing")
            result = await client.call_tool_mcp("search_knowledge", {"query": "refunds"})
            assert result.isError
            assert "collection is missing" in result.content[0].text
        constructor.assert_not_called()
    finally:
        await server.qdrant_connector._client.close()
