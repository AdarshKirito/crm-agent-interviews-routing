"""Runtime settings for the crmroute agent (all overridable by environment variables).

Keys come from the repository-root .env (never committed). Existing environment
variables win over the file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)
def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


# Either flag turns Vertex on. google-genai uses GOOGLE_GENAI_USE_ENTERPRISE when the two
# disagree, so both are set to the same value.
USE_VERTEX = _flag("GOOGLE_GENAI_USE_VERTEXAI") or _flag("GOOGLE_GENAI_USE_ENTERPRISE")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "true" if USE_VERTEX else "false"
if USE_VERTEX:
    # Vertex AI (application default credentials, GOOGLE_CLOUD_PROJECT / _LOCATION).
    # google-genai silently prefers an API key over Vertex when one is set, so the AI
    # Studio keys are removed from this process.
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("GEMINI_API_KEY", None)
elif os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    # AI Studio free tier: ADK / google-genai read GOOGLE_API_KEY.
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

DATA_DIR = Path(__file__).parent / "data"

# Model per role. A bare "gemini-..." name uses the AI Studio key natively; any
# "provider/model" string goes through LiteLLM (see app/models.py).
BIG_MODEL = os.getenv("CRMROUTE_BIG_MODEL", "gemini-3.8-flash")
SMALL_MODEL = os.getenv("CRMROUTE_SMALL_MODEL", "gemini-3.1-flash-lite")
POLICY_MODEL = os.getenv("CRMROUTE_POLICY_MODEL", SMALL_MODEL)
# One thinking level everywhere; it must match the baselines (CRMARENA_THINKING_LEVEL).
THINKING_LEVEL = os.getenv("CRMROUTE_THINKING_LEVEL", "low")
MAX_OUTPUT_TOKENS = int(os.getenv("CRMROUTE_MAX_OUTPUT_TOKENS", "8192"))

# Measured runs pin one model per role and wait out rate limits (up to this long per
# call). CRMROUTE_FALLBACK=1 is for the local demo only: on a provider error the call
# moves along FALLBACK_CHAIN.
MAX_RETRY_WAIT_S = float(os.getenv("CRMROUTE_MAX_RETRY_WAIT_S", "900"))
FALLBACK = os.getenv("CRMROUTE_FALLBACK", "0").lower() in ("1", "true", "yes")
FALLBACK_CHAIN = [m.strip() for m in os.getenv(
    "CRMROUTE_FALLBACK_CHAIN",
    "gemini-3.8-flash,gemini-3.1-flash-lite,mistral/mistral-small-latest,groq/openai/gpt-oss-120b,"
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free,ollama_chat/qwen3:8b",
).split(",") if m.strip()]
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("CRMROUTE_OLLAMA_NUM_CTX", "16384"))
OLLAMA_THINK = os.getenv("CRMROUTE_OLLAMA_THINK", "0").lower() in ("1", "true", "yes")
# Every model call is appended here as JSON lines (model that answered, tokens, waits).
CALL_LOG = os.getenv("CRMROUTE_CALL_LOG", str(REPO_ROOT / "runs" / "calls.jsonl"))

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
# Tool calls allowed per user turn (the ReAct baseline stops at 20 turns in total).
MAX_TOOL_CALLS_PER_TURN = int(os.getenv("CRMROUTE_MAX_TOOL_CALLS_PER_TURN", "10"))
MAX_CLARIFYING_QUESTIONS = int(os.getenv("CRMROUTE_MAX_CLARIFYING_QUESTIONS", "3"))
ROUTER_MIN_CONFIDENCE = float(os.getenv("CRMROUTE_ROUTER_MIN_CONFIDENCE", "0.5"))
