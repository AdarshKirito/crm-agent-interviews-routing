#!/usr/bin/env bash
# Start the local stack: Salesforce MCP (:3333), knowledge search MCP (:8765) and the
# agent API (:8000). Ctrl+C stops everything.
#   ./scripts/dev_up.sh                      # demo: automatic provider fallback on
#   CRMROUTE_FALLBACK=0 ./scripts/dev_up.sh  # one pinned model per role
#   CRMROUTE_MODE=no_route ./scripts/dev_up.sh
# Keys come from the repo-root .env. Every model call, with the model that answered it,
# is appended to runs/dev_logs/calls.jsonl.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a
export PYTHONIOENCODING=utf-8
export CRMROUTE_FALLBACK="${CRMROUTE_FALLBACK:-1}"
py() { if [ -x "$1/.venv/Scripts/python" ]; then echo "$1/.venv/Scripts"; else echo "$1/.venv/bin"; fi; }
LOGS="$ROOT/runs/dev_logs"; mkdir -p "$LOGS"
export CRMROUTE_CALL_LOG="${CRMROUTE_CALL_LOG:-$LOGS/calls.jsonl}"
PIDS=()
trap 'for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done' EXIT

up() {
  local status
  status="$(curl -s -m 2 -o /dev/null -w '%{http_code}' "$1")" || return 1
  if [[ "$1" == */mcp ]]; then
    [[ "$status" == 200 || "$status" == 400 || "$status" == 405 || "$status" == 406 ]]
  else
    [[ "$status" == 2?? ]]
  fi
}

if ! up http://127.0.0.1:3333/healthz; then
  (cd "$ROOT/mcp-salesforce" && SF_CACHE_DIR="$ROOT/data/cache/sf" node dist/index.js --http --port 3333 \
     --env "$ROOT/vendor/CRMArena/.env" > "$LOGS/mcp_salesforce.log" 2>&1) & PIDS+=($!)
fi
if ! up http://127.0.0.1:8765/mcp; then
  (cd "$ROOT/search" && QDRANT_LOCAL_PATH="$ROOT/data/qdrant" HYBRID_ENABLED=true HYBRID_ONLY=true QDRANT_READ_ONLY=true \
     FASTMCP_SERVER_PORT=8765 FASTMCP_SERVER_LOG_LEVEL=WARNING "$(py "$ROOT/search")/mcp-server-qdrant" --transport streamable-http \
     > "$LOGS/mcp_search.log" 2>&1) & PIDS+=($!)
fi
(cd "$ROOT/agent" && "$(py "$ROOT/agent")/python" -m uvicorn app.fast_api_app:app --host 127.0.0.1 --port 8000 \
   > "$LOGS/agent.log" 2>&1) & PIDS+=($!)

for _ in $(seq 1 90); do up http://127.0.0.1:8000/list-apps && break; sleep 1; done
up http://127.0.0.1:8000/list-apps || { echo "agent did not start; see $LOGS/agent.log" >&2; exit 1; }
echo "agent:            http://127.0.0.1:8000  (ADK web UI at /dev-ui/)  fallback=$CRMROUTE_FALLBACK"
echo "salesforce MCP:   http://127.0.0.1:3333/mcp"
echo "knowledge MCP:    http://127.0.0.1:8765/mcp"
echo "call log:         $CRMROUTE_CALL_LOG"
wait
