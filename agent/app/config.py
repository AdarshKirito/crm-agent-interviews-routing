"""Runtime settings for the crmroute agent (all overridable by environment variables)."""

import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

BIG_MODEL = os.getenv("CRMROUTE_BIG_MODEL", "gemini-3.8-flash")
SMALL_MODEL = os.getenv("CRMROUTE_SMALL_MODEL", "gemini-3.1-flash-lite")
# One thinking level everywhere; it must match the baselines (CRMARENA_THINKING_LEVEL).
THINKING_LEVEL = os.getenv("CRMROUTE_THINKING_LEVEL", "low")
# Google recommends temperature 1.0 for Gemini 3; the baselines use the same value.
TEMPERATURE = float(os.getenv("CRMROUTE_TEMPERATURE", "1.0"))
MAX_OUTPUT_TOKENS = int(os.getenv("CRMROUTE_MAX_OUTPUT_TOKENS", "8192"))

# "no_route": every task on the big model (system 3). "route": the small model
# where the routing table allows (system 4). "all_small": every task on the small
# model, used on dev to fit the routing table.
MODE = os.getenv("CRMROUTE_MODE", "route")

SALESFORCE_MCP_URL = os.getenv("CRMROUTE_SALESFORCE_MCP_URL", "http://127.0.0.1:3333/mcp")
SEARCH_MCP_URL = os.getenv("CRMROUTE_SEARCH_MCP_URL", "http://127.0.0.1:8765/mcp")
MCP_BEARER_TOKEN = os.getenv("CRMROUTE_MCP_BEARER_TOKEN", "")
# "hybrid": the search fork's search_knowledge; "sosl": the Salesforce server's.
KNOWLEDGE_BACKEND = os.getenv("CRMROUTE_KNOWLEDGE_BACKEND", "hybrid")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PROMPT_GUARD_MODEL = os.getenv("CRMROUTE_PROMPT_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")
PROMPT_GUARD_THRESHOLD = float(os.getenv("CRMROUTE_PROMPT_GUARD_THRESHOLD", "0.9"))

MAX_CHECKER_RETRIES = int(os.getenv("CRMROUTE_MAX_CHECKER_RETRIES", "1"))
MAX_CLARIFYING_QUESTIONS = int(os.getenv("CRMROUTE_MAX_CLARIFYING_QUESTIONS", "3"))
ROUTER_MIN_CONFIDENCE = float(os.getenv("CRMROUTE_ROUTER_MIN_CONFIDENCE", "0.5"))
