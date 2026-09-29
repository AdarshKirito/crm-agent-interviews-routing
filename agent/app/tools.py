"""MCP toolsets: the read-only Salesforce server and the hybrid knowledge-search server.

Both are reached over Streamable HTTP. Each request carries the session's org in the
`x-crm-org` header, so one server instance serves both CRMArena-Pro orgs.
"""

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import BaseTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

from . import state_keys as K
from .config import KNOWLEDGE_BACKEND, MCP_BEARER_TOKEN, SALESFORCE_MCP_URL, SEARCH_MCP_URL

SALESFORCE_TOOLS = ("soql_query", "sosl_search", "describe_object", "get_record", "search_knowledge")


def _headers(ctx: ReadonlyContext) -> dict[str, str]:
    headers = {"x-crm-org": str(ctx.state.get(K.ORG) or "b2b")}
    if MCP_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_BEARER_TOKEN}"
    return headers


def _salesforce_filter(tool: BaseTool, readonly_context: ReadonlyContext | None = None) -> bool:
    # Knowledge search comes from exactly one backend so the model never sees two
    # tools with the same name.
    if tool.name == "search_knowledge":
        return KNOWLEDGE_BACKEND == "sosl"
    return tool.name in SALESFORCE_TOOLS


def _connection(url: str) -> StreamableHTTPConnectionParams:
    return StreamableHTTPConnectionParams(url=url, timeout=60.0, sse_read_timeout=300.0)


def build_toolsets() -> list[McpToolset]:
    toolsets = [
        McpToolset(
            connection_params=_connection(SALESFORCE_MCP_URL),
            tool_filter=_salesforce_filter,
            header_provider=_headers,
        )
    ]
    if KNOWLEDGE_BACKEND == "hybrid":
        toolsets.append(
            McpToolset(
                connection_params=_connection(SEARCH_MCP_URL),
                tool_filter=["search_knowledge"],
                header_provider=_headers,
            )
        )
    return toolsets
