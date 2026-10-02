#!/usr/bin/env bash
# Container entrypoint: start both MCP servers on localhost, then the agent on $PORT.
# Salesforce credentials come from the environment (Cloud Run: Secret Manager).
set -euo pipefail

cd /srv/mcp-salesforce
SF_CACHE_DIR=/tmp/sf-cache node dist/index.js --http --host 127.0.0.1 --port 3333 &

cd /srv/search
QDRANT_LOCAL_PATH=/srv/data/qdrant HYBRID_ENABLED=true HYBRID_ONLY=true QDRANT_READ_ONLY=true \
FASTMCP_SERVER_HOST=127.0.0.1 FASTMCP_SERVER_PORT=8765 FASTMCP_SERVER_LOG_LEVEL=WARNING \
  /srv/search/.venv/bin/mcp-server-qdrant --transport streamable-http &

/srv/agent/.venv/bin/python - <<'PY'
import time, urllib.request, urllib.error
for url, protocol in (("http://127.0.0.1:3333/healthz", False), ("http://127.0.0.1:8765/mcp", True)):
    for attempt in range(60):
        try:
            with urllib.request.urlopen(url, timeout=1):
                break
        except urllib.error.HTTPError as error:
            if protocol and error.code in (400, 405, 406):
                break
        except OSError:
            pass
        time.sleep(0.5)
    else:
        raise SystemExit(f"Required MCP service did not become ready: {url}")
PY

cd /srv/agent
exec /srv/agent/.venv/bin/uvicorn app.fast_api_app:app --host 0.0.0.0 --port "${PORT:-8080}"
