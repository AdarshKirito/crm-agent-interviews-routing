"""Send ADK traces to Arize Phoenix when PHOENIX_COLLECTOR_ENDPOINT is set.

Local:  uvx arize-phoenix serve   (UI at http://localhost:6006)
        PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
"""

import logging
import os

logger = logging.getLogger(__name__)
_done = False


def setup_tracing() -> bool:
    global _done
    if _done or not os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        return _done
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    from phoenix.otel import register

    provider = register(project_name=os.getenv("PHOENIX_PROJECT_NAME", "crmroute"), batch=True)
    GoogleADKInstrumentor().instrument(tracer_provider=provider)
    _done = True
    logger.info("Phoenix tracing enabled -> %s", os.getenv("PHOENIX_COLLECTOR_ENDPOINT"))
    return True
