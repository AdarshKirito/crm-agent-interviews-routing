"""Tracing must not break start-up: the instrumentor patches ADK internals that move between releases."""

import os
import subprocess
import sys


def test_agent_starts_with_phoenix_tracing_enabled():
    # Nothing listens on port 9; spans are exported in the background and simply dropped.
    env = {**os.environ, "PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:9"}
    code = "import app.agent\nfrom app.tracing import setup_tracing\nassert setup_tracing()"
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=300)
