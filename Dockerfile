# One container: the ADK agent (port $PORT) plus its two MCP servers on
# localhost (Salesforce :3333, knowledge search :8765).
# Build after exporting the knowledge articles (scripts/export_knowledge.py) so the
# search index can be built into the image:
#   docker build -t crmroute .

# ---- Salesforce MCP server (TypeScript) ------------------------------------------
FROM node:24-slim AS mcp-salesforce
WORKDIR /build
COPY mcp-salesforce/package.json mcp-salesforce/package-lock.json ./
RUN npm ci
COPY mcp-salesforce/tsconfig.json ./
COPY mcp-salesforce/src ./src
RUN npx tsc -p tsconfig.json && npm prune --omit=dev

# ---- Python runtime --------------------------------------------------------------
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/
COPY --from=node:24-slim /usr/local/bin/node /usr/local/bin/node
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 FASTEMBED_CACHE_PATH=/models PYTHONUNBUFFERED=1
WORKDIR /srv

# Salesforce MCP server
COPY --from=mcp-salesforce /build/dist /srv/mcp-salesforce/dist
COPY --from=mcp-salesforce /build/node_modules /srv/mcp-salesforce/node_modules
COPY mcp-salesforce/package.json /srv/mcp-salesforce/

# Knowledge-search MCP server (mcp-server-qdrant fork) + index built at image build time
COPY search/pyproject.toml search/uv.lock search/README.md search/LICENSE /srv/search/
COPY search/src /srv/search/src
RUN cd /srv/search && uv sync --frozen --no-dev
COPY data/knowledge /srv/data/knowledge
RUN /srv/search/.venv/bin/crm-knowledge-index --knowledge-dir /srv/data/knowledge --qdrant-path /srv/data/qdrant

# Agent
COPY agent/pyproject.toml agent/uv.lock agent/README.md /srv/agent/
RUN cd /srv/agent && uv sync --frozen --no-dev --no-install-project
COPY agent/app /srv/agent/app
RUN cd /srv/agent && uv sync --frozen --no-dev \
 && /srv/agent/.venv/bin/python -c "from app.router import get_router; get_router().warm_up(); from app.guard.pii import warm_up; warm_up()"

COPY deploy/start.sh /srv/start.sh
RUN chmod +x /srv/start.sh
ARG AGENT_VERSION=0.0.0
ENV AGENT_VERSION=${AGENT_VERSION} PORT=8080
EXPOSE 8080
CMD ["/srv/start.sh"]
