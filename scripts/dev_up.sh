#!/usr/bin/env bash
# Start the local stack: Salesforce MCP (:3333), knowledge search MCP (:8765) and the
# agent API (:8000). Ctrl+C stops everything.
#   ./scripts/dev_up.sh            # routing mode
#   CRMROUTE_MODE=no_route ./scripts/dev_up.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
py() { if [ -x "$1/.venv/Scripts/python" ]; then echo "$1/.venv/Scripts"; else echo "$1/.venv/bin"; fi; }
LOGS="$ROOT/runs/dev_logs"; mkdir -p "$LOGS"
PIDS=()
trap 'for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done' EXIT

(cd "$ROOT/mcp-salesforce" && SF_CACHE_DIR="$ROOT/data/cache/sf" node dist/index.js --http --port 3333 \
   --env "$ROOT/vendor/CRMArena/.env" > "$LOGS/mcp_salesforce.log" 2>&1) & PIDS+=($!)
(cd "$ROOT/search" && QDRANT_LOCAL_PATH="$ROOT/data/qdrant" HYBRID_ENABLED=true HYBRID_ONLY=true QDRANT_READ_ONLY=true \
   FASTMCP_SERVER_PORT=8765 FASTMCP_SERVER_LOG_LEVEL=WARNING "$(py "$ROOT/search")/mcp-server-qdrant" --transport streamable-http \
   > "$LOGS/mcp_search.log" 2>&1) & PIDS+=($!)
(cd "$ROOT/agent" && "$(py "$ROOT/agent")/python" -m uvicorn app.fast_api_app:app --host 127.0.0.1 --port 8000 \
   > "$LOGS/agent.log" 2>&1) & PIDS+=($!)

for _ in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8000/list-apps >/dev/null && break; sleep 1; done
echo "agent:            http://127.0.0.1:8000  (ADK web UI at /dev-ui/)"
echo "salesforce MCP:   http://127.0.0.1:3333/mcp"
echo "knowledge MCP:    http://127.0.0.1:8765/mcp"
echo "logs:             $LOGS"
wait
