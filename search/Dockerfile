# crmroute fork: installs THIS source (not the PyPI package) and serves the hybrid
# knowledge search over Streamable HTTP. Mount or bake a Qdrant local store at /data/qdrant
# (build it with crm-knowledge-index). The project's root Dockerfile bundles this server
# with the agent for Cloud Run.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev
ENV QDRANT_LOCAL_PATH=/data/qdrant HYBRID_ENABLED=true HYBRID_ONLY=true QDRANT_READ_ONLY=true \
    FASTMCP_SERVER_HOST=0.0.0.0 FASTMCP_SERVER_PORT=8765
EXPOSE 8765
CMD ["/app/.venv/bin/mcp-server-qdrant", "--transport", "streamable-http"]
