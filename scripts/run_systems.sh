#!/usr/bin/env bash
# Run the four systems (and the dev-only small-model run) on a fixed task list,
# using explicitly pinned providers (Vertex, direct APIs and/or local Ollama).
#
#   BIG_MODEL=... SMALL_MODEL=... CRMARENA_JUDGE_MODEL=... CRMARENA_JUDGE_PROVIDER=... \
#   SPLIT=dev  ./scripts/run_systems.sh full agent_small
#   SPLIT=test ./scripts/run_systems.sh react react_privacy full routed
#
# Systems:
#   react          1. benchmark ReAct baseline (on BIG_MODEL)
#   react_privacy  2. same agent with --privacy_aware_prompt true
#   full           3. crmroute agent, every task on the big model (CRMROUTE_MODE=no_route)
#   routed         4. crmroute agent with the routing table (CRMROUTE_MODE=route)
#   agent_small    dev only: crmroute agent with every task on the small model
#
# Everything that must stay equal across systems is set once here: models, thinking
# level, eval mode, judge, simulated user and task ids. Each role is pinned to one
# model for the whole run; rate limits are waited out and never cause a switch, and a
# used-up daily quota stops the run cleanly. Runs resume (--reuse_results): tasks that
# ended in an API error are redone. Every system writes a manifest of its pins.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# API keys from the repo-root .env (git-ignored; values are never printed)
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a
export PYTHONIOENCODING=utf-8
SPLIT="${SPLIT:-dev}"
TASK_IDS="${TASK_IDS:-$ROOT/data/$SPLIT.json}"
OUT="${OUT:-$ROOT/runs/$SPLIT}"
EVAL_MODE="${EVAL_MODE:-aided}"
ORGS="${ORGS:-b2b b2c}"
MODES="${MODES:-single multi}"
MAX_USER_TURNS="${MAX_USER_TURNS:-10}"
MAX_TURNS="${MAX_TURNS:-20}"
export CRMARENA_THINKING_LEVEL="${CRMARENA_THINKING_LEVEL:-low}"
export CRMROUTE_THINKING_LEVEL="$CRMARENA_THINKING_LEVEL"
: "${BIG_MODEL:?set BIG_MODEL, e.g. gemini-3.1-flash-lite or mistral/mistral-medium-latest}"
: "${SMALL_MODEL:?set SMALL_MODEL, e.g. ollama_chat/qwen3:8b or mistral/mistral-small-latest}"
POLICY_MODEL="${POLICY_MODEL:-$SMALL_MODEL}"
: "${CRMARENA_JUDGE_MODEL:?set CRMARENA_JUDGE_MODEL (the same for every system)}"
: "${CRMARENA_JUDGE_PROVIDER:?set CRMARENA_JUDGE_PROVIDER}"
export CRMARENA_USER_MODEL="${CRMARENA_USER_MODEL:-$CRMARENA_JUDGE_MODEL}"
export CRMARENA_USER_PROVIDER="${CRMARENA_USER_PROVIDER:-$CRMARENA_JUDGE_PROVIDER}"
export CRMROUTE_FALLBACK=0  # automatic fallback is for the local demo only
# docker (default): the agent runs in the crmroute image with the full guard. On some
# Windows hosts Smart App Control blocks spaCy, which Presidio's name detection needs.
AGENT_RUNTIME="${AGENT_RUNTIME:-docker}"
IMAGE="${IMAGE:-crmroute:local}"
# Bare gemini names use Vertex AI when GOOGLE_GENAI_USE_VERTEXAI/_ENTERPRISE is true, else the AI Studio key.
is_true() { case "${1,,}" in true|1) return 0 ;; *) return 1 ;; esac; }
if is_true "${GOOGLE_GENAI_USE_VERTEXAI:-}" || is_true "${GOOGLE_GENAI_USE_ENTERPRISE:-}"; then USE_VERTEX=1; else USE_VERTEX=0; fi
if [[ "$BIG_MODEL" == */* ]]; then BASE_PROVIDER="${BIG_MODEL%%/*}"
elif [ "$USE_VERTEX" = 1 ]; then BASE_PROVIDER=vertex_ai; else BASE_PROVIDER=gemini; fi
# Parallel streams of one system (e.g. ORGS=b2c MODES=multi) need their own agent port.
PORT_OFFSET="${PORT_OFFSET:-0}"
STREAM="$(echo "$ORGS-$MODES" | tr ' ' '_')"

py() {  # python of a venv on Windows (Scripts) or Linux/macOS (bin)
  if [ -x "$1/.venv/Scripts/python" ]; then echo "$1/.venv/Scripts/python"; else echo "$1/.venv/bin/python"; fi
}
hostpath() { cygpath -m "$1" 2>/dev/null || echo "$1"; }  # Git Bash -> E:/... for docker -v
BENCH_PY="$(py "$ROOT/vendor/CRMArena")"
AGENT_PY="$(py "$ROOT/agent")"
SEARCH_BIN="$(dirname "$(py "$ROOT/search")")"
PIDS=()
CONTAINERS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  for c in "${CONTAINERS[@]:-}"; do
    if [ -n "$c" ]; then
      # Keep the server traceback even after this run's disposable container stops.
      docker logs "$c" > "$OUT/logs/${c}.log" 2>&1 || true
      docker rm -f "$c" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT
mkdir -p "$OUT/logs"
OUT="$(cd "$OUT" && pwd)"  # docker -v and run_tasks.py need absolute paths
TASK_IDS="$(cd "$(dirname "$TASK_IDS")" && pwd)/$(basename "$TASK_IDS")"

wait_http() {  # url
  for _ in $(seq 1 180); do curl -fsS -m 2 -o /dev/null "$1" 2>/dev/null && return 0; sleep 1; done
  echo "service at $1 did not come up" >&2; return 1
}

mcp_ready() {
  local status
  status="$(curl -s -m 2 -o /dev/null -w '%{http_code}' "$1")" || return 1
  # A plain GET lacks the MCP Accept/session headers. These responses still prove
  # the protocol handler is ready; connection failures and HTTP 5xx do not.
  [[ "$status" == 200 || "$status" == 400 || "$status" == 405 || "$status" == 406 ]]
}

start_mcp() {  # host MCP servers (only needed when the agent runs on the host)
  [ "$AGENT_RUNTIME" = docker ] && return 0
  if ! curl -fsS -m 2 http://127.0.0.1:3333/healthz >/dev/null 2>&1; then
    (cd "$ROOT/mcp-salesforce" && SF_CACHE_DIR="$ROOT/data/cache/sf" node dist/index.js --http --port 3333 \
        --env "$ROOT/vendor/CRMArena/.env" > "$OUT/logs/mcp_salesforce.log" 2>&1) & PIDS+=($!)
    wait_http http://127.0.0.1:3333/healthz
  fi
  if ! mcp_ready http://127.0.0.1:8765/mcp; then
    (cd "$ROOT/search" && QDRANT_LOCAL_PATH="$ROOT/data/qdrant" HYBRID_ENABLED=true HYBRID_ONLY=true QDRANT_READ_ONLY=true \
        FASTMCP_SERVER_PORT=8765 FASTMCP_SERVER_LOG_LEVEL=WARNING "$SEARCH_BIN/mcp-server-qdrant" --transport streamable-http \
        > "$OUT/logs/mcp_search.log" 2>&1) & PIDS+=($!)
    for _ in $(seq 1 90); do mcp_ready http://127.0.0.1:8765/mcp && break; sleep 1; done
    mcp_ready http://127.0.0.1:8765/mcp || { echo "knowledge MCP did not start" >&2; return 1; }
  fi
}

start_agent() {  # mode port
  local mode="$1" port="$2"
  if [ "$AGENT_RUNTIME" = docker ]; then
    # the image bundles both MCP servers; Ollama is reached on the host
    local name="crmroute-$mode-$port-$$"
    local adc="${GOOGLE_APPLICATION_CREDENTIALS:-${APPDATA:-$HOME/.config}/gcloud/application_default_credentials.json}" vertex_args=()
    if [ "$USE_VERTEX" = 1 ]; then  # gcloud application default credentials, read-only
      [ -f "$adc" ] || { echo "Vertex credentials file missing: set GOOGLE_APPLICATION_CREDENTIALS" >&2; return 1; }
      vertex_args=(-v "$(hostpath "$adc"):/secrets/adc.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/adc.json)
    fi
    mkdir -p "$ROOT/data/cache/sf-docker"
    MSYS_NO_PATHCONV=1 docker run -d --name "$name" -p "127.0.0.1:$port:8080" \
      --env-file "$(hostpath "$ROOT/.env")" --env-file "$(hostpath "$ROOT/vendor/CRMArena/.env")" "${vertex_args[@]}" \
      -e OLLAMA_API_BASE=http://host.docker.internal:11434 -e CRMROUTE_MODE="$mode" \
      -e CRMROUTE_BIG_MODEL="$BIG_MODEL" -e CRMROUTE_SMALL_MODEL="$SMALL_MODEL" -e CRMROUTE_POLICY_MODEL="$POLICY_MODEL" \
      -e CRMROUTE_FALLBACK=0 -e CRMROUTE_THINKING_LEVEL="$CRMROUTE_THINKING_LEVEL" \
      -e CRMROUTE_CALL_LOG="/srv/runs/agent_calls_${mode}_${STREAM}.jsonl" \
      -v "$(hostpath "$OUT/logs"):/srv/runs" \
      -v "$(hostpath "$ROOT/agent/app/data/routing.yaml"):/srv/agent/app/data/routing.yaml:ro" \
      -v "$(hostpath "$ROOT/data/cache/sf-docker"):/tmp/sf-cache" \
      "$IMAGE" >/dev/null
    CONTAINERS+=("$name")
  else
    (cd "$ROOT/agent" && env CRMROUTE_MODE="$mode" CRMROUTE_BIG_MODEL="$BIG_MODEL" CRMROUTE_SMALL_MODEL="$SMALL_MODEL" \
        CRMROUTE_POLICY_MODEL="$POLICY_MODEL" CRMROUTE_CALL_LOG="$OUT/logs/agent_calls_${mode}_${STREAM}.jsonl" \
        "$AGENT_PY" -m uvicorn app.fast_api_app:app --host 127.0.0.1 --port "$port" \
        > "$OUT/logs/agent_${mode}_${port}.log" 2>&1) & PIDS+=($!)
  fi
  wait_http "http://127.0.0.1:$port/list-apps"
}

manifest() {  # system: record every pin so a result can be audited later
  local image_id=""
  if [ "$AGENT_RUNTIME" = docker ]; then image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"; fi
  CRMARENA_RUN_FINGERPRINT="$("$BENCH_PY" "$ROOT/scripts/run_manifest.py" \
    --root "$ROOT" --output "$OUT/manifest_${1}_${STREAM}.json" --system "$1" \
    --split "$SPLIT" --task-ids "$TASK_IDS" --eval-mode "$EVAL_MODE" --orgs "$ORGS" --modes "$MODES" \
    --big-model "$BIG_MODEL" --small-model "$SMALL_MODEL" --policy-model "$POLICY_MODEL" \
    --judge-model "$CRMARENA_JUDGE_MODEL" --judge-provider "$CRMARENA_JUDGE_PROVIDER" \
    --user-model "$CRMARENA_USER_MODEL" --user-provider "$CRMARENA_USER_PROVIDER" \
    --thinking-level "$CRMARENA_THINKING_LEVEL" \
    --backend "$([ "$USE_VERTEX" = 1 ] && echo "vertex:${GOOGLE_CLOUD_PROJECT:-}" || echo ai_studio)" \
    --agent-runtime "$AGENT_RUNTIME" --image "$IMAGE" --image-id "$image_id" \
    --max-user-turns "$MAX_USER_TURNS" --max-turns "$MAX_TURNS")"
  export CRMARENA_RUN_FINGERPRINT
}

bench() {  # system strategy-args...
  local system="$1"; shift
  export CRMARENA_CALL_LOG="$OUT/logs/bench_calls_${system}_${STREAM}.jsonl"
  for org in $ORGS; do
    for mode in $MODES; do
      local flags=()
      [ "$mode" = multi ] && flags+=(--interactive --max_user_turns "$MAX_USER_TURNS")
      echo "== $system | $SPLIT | $org | $mode"
      (cd "$ROOT/vendor/CRMArena" && "$BENCH_PY" -u run_tasks.py --task_category all --task_ids_file "$TASK_IDS" \
          --org_type "$org" --agent_eval_mode "$EVAL_MODE" --max_turns "$MAX_TURNS" --reuse_results --log_dir "$OUT/$system" \
          --judge_model "$CRMARENA_JUDGE_MODEL" --judge_provider "$CRMARENA_JUDGE_PROVIDER" \
          --user_model "$CRMARENA_USER_MODEL" --user_provider "$CRMARENA_USER_PROVIDER" \
          "${flags[@]}" "$@") 2>&1 | tee -a "$OUT/logs/${system}_${org}_${mode}.log"
    done
  done
}

[ "$#" -gt 0 ] || { echo "usage: SPLIT=dev|test $0 system..." >&2; exit 2; }
for org in $ORGS; do case "$org" in b2b|b2c) ;; *) echo "unknown org: $org" >&2; exit 2 ;; esac; done
for mode in $MODES; do case "$mode" in single|multi) ;; *) echo "unknown mode: $mode" >&2; exit 2 ;; esac; done
for system in "$@"; do
  case "$system" in
    react|react_privacy|full|routed) ;;
    agent_small) [ "$SPLIT" = dev ] || { echo "agent_small is for dev only" >&2; exit 2; } ;;
    *) echo "unknown system: $system" >&2; exit 2 ;;
  esac
  # Validate the previous run's pins before starting a container or a model call.
  manifest "$system"
  case "$system" in
    react)
      bench react --agent_strategy react --model "$BIG_MODEL" --llm_provider "$BASE_PROVIDER" --privacy_aware_prompt false ;;
    react_privacy)
      bench react_privacy --agent_strategy react --model "$BIG_MODEL" --llm_provider "$BASE_PROVIDER" --privacy_aware_prompt true ;;
    full)
      p=$((8001 + PORT_OFFSET)); start_mcp; start_agent no_route "$p"
      bench full --agent_strategy remote --model crmroute-full --remote_url "http://127.0.0.1:$p" ;;
    routed)
      p=$((8002 + PORT_OFFSET)); start_mcp; start_agent route "$p"
      bench routed --agent_strategy remote --model crmroute-routed --remote_url "http://127.0.0.1:$p" ;;
    agent_small)
      p=$((8003 + PORT_OFFSET)); start_mcp; start_agent all_small "$p"
      bench agent_small --agent_strategy remote --model crmroute-small --remote_url "http://127.0.0.1:$p" ;;
    *) echo "unknown system: $system" >&2; exit 2 ;;
  esac
done
echo "results in $OUT"
