"""Tracing must not break start-up, and spans must reach the collector.

The instrumentor patches ADK internals that move between releases, ADK's API server
installs its own global TracerProvider, and the exporter settings decide whether a
local Phoenix receives anything at all.
"""

import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

PROBE = ("from opentelemetry import trace\n"
         "with trace.get_tracer('test').start_as_current_span('probe'):\n    pass\n"
         "assert trace.get_tracer_provider().force_flush()\n")


@pytest.mark.parametrize("start", [
    # the server module, as uvicorn loads it (tracing must be set up before ADK's app)
    "import app.fast_api_app\nfrom app.tracing import setup_tracing\nassert setup_tracing()\n",
    # another library installed the global provider first
    "from opentelemetry import trace\nfrom opentelemetry.sdk.trace import TracerProvider\n"
    "trace.set_tracer_provider(TracerProvider())\nimport app.agent\n"
    "from app.tracing import setup_tracing\nassert setup_tracing()\n",
], ids=["api_server", "provider_already_set"])
def test_agent_starts_with_tracing_and_exports_spans_over_http(start):
    received = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server naming
            received.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {**os.environ, "PHOENIX_COLLECTOR_ENDPOINT": f"http://127.0.0.1:{server.server_port}"}
    try:
        subprocess.run([sys.executable, "-c", start + PROBE], env=env, check=True, timeout=300)
    finally:
        server.shutdown()
    assert "/v1/traces" in received
