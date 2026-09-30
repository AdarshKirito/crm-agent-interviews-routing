#!/usr/bin/env bash
# Run the four systems (and the dev-only small-model run) on a fixed task list,
# using free API tiers and/or local Ollama.
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
export CRMARENA_THINKING_LEVEL="${CRMARENA_THINKING_LEVEL:-low}"
export CRMROUTE_THINKING_LEVEL="$CRMARENA_THINKING_LEVEL"
: "${BIG_MODEL:?set BIG_MODEL, e.g. gemini-3.1-flash-lite or mistral/mistral-medium-latest}"
: "${SMALL_MODEL:?set SMALL_MODEL, e.g. ollama_chat/qwen3:8b or mistral/mistral-small-latest}"
POLICY_MODEL="${POLICY_MODEL:-$SMALL_MODEL}"
: "${CRMARENA_JUDGE_MODEL:?set CRMARENA_JUDGE_MODEL (a free model, the same for every system)}"
: "${CRMARENA_JUDGE_PROVIDER:?set CRMARENA_JUDGE_PROVIDER (gemini, mistral, groq or ollama_chat)}"
export CRMARENA_USER_MODEL="${CRMARENA_USER_MODEL:-$CRMARENA_JUDGE_MODEL}"
export CRMARENA_USER_PROVIDER="${CRMARENA_USER_PROVIDER:-$CRMARENA_JUDGE_PROVIDER}"
export CRMROUTE_FALLBACK=0  # automatic fallback is for the local demo only
# docker (default): the agent runs in the crmroute image with the full guard. On some
# Windows hosts Smart App Control blocks spaCy, which Presidio's name detection needs.
AGENT_RUNTIME="${AGENT_RUNTIME:-docker}"
IMAGE="${IMAGE:-crmroute:local}"
# The ReAct baselines run the big model; bare gemini names use the AI Studio key.
if [[ "$BIG_MODEL" == */* ]]; then BASE_PROVIDER="${BIG_MODEL%%/*}"; else BASE_PROVIDER=gemini; fi

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
  for c in "${CONTAINERS[@]:-}"; do [ -n "$c" ] && docker rm -f "$c" >/dev/null 2>&1 || true; done
}
trap cleanup EXIT
mkdir -p "$OUT/logs"
OUT="$(cd "$OUT" && pwd)"  # docker -v and run_tasks.py need absolute paths
TASK_IDS="$(cd "$(dirname "$TASK_IDS")" && pwd)/$(basename "$TASK_IDS")"

wait_http() {  # url
  for _ in $(seq 1 180); do curl -s -m 2 -o /dev/null "$1" && return 0; sleep 1; done
  echo "service at $1 did not come up" >&2; return 1
}

start_mcp() {  # host MCP servers (only needed when the agent runs on the host)
  [ "$AGENT_RUNTIME" = docker ] && return 0
  if ! curl -s -m 2 http://127.0.0.1:3333/healthz >/dev/null; then
    (cd "$ROOT/mcp-salesforce" && SF_CACHE_DIR="$ROOT/data/cache/sf" node dist/index.js --http --port 3333 \
        --env "$ROOT/vendor/CRMArena/.env" > "$OUT/logs/mcp_salesforce.log" 2>&1) & PIDS+=($!)
    wait_http http://127.0.0.1:3333/healthz
  fi
  if ! curl -s -m 2 -o /dev/null http://127.0.0.1:8765/mcp; then
    (cd "$ROOT/search" && QDRANT_LOCAL_PATH="$ROOT/data/qdrant" HYBRID_ENABLED=true HYBRID_ONLY=true QDRANT_READ_ONLY=true \
        FASTMCP_SERVER_PORT=8765 FASTMCP_SERVER_LOG_LEVEL=WARNING "$SEARCH_BIN/mcp-server-qdrant" --transport streamable-http \
        > "$OUT/logs/mcp_search.log" 2>&1) & PIDS+=($!)
    sleep 8
  fi
}

start_agent() {  # mode port
  local mode="$1" port="$2"
  if [ "$AGENT_RUNTIME" = docker ]; then
    # the image bundles both MCP servers; Ollama is reached on the host
    local name="crmroute-$mode"
    docker rm -f "$name" >/dev/null 2>&1 || true
    mkdir -p "$ROOT/data/cache/sf-docker"
    MSYS_NO_PATHCONV=1 docker run -d --name "$name" -p "127.0.0.1:$port:8080" \
      --env-file "$(hostpath "$ROOT/.env")" --env-file "$(hostpath "$ROOT/vendor/CRMArena/.env")" \
      -e OLLAMA_API_BASE=http://host.docker.internal:11434 -e CRMROUTE_MODE="$mode" \
      -e CRMROUTE_BIG_MODEL="$BIG_MODEL" -e CRMROUTE_SMALL_MODEL="$SMALL_MODEL" -e CRMROUTE_POLICY_MODEL="$POLICY_MODEL" \
      -e CRMROUTE_FALLBACK=0 -e CRMROUTE_THINKING_LEVEL="$CRMROUTE_THINKING_LEVEL" \
      -e CRMROUTE_CALL_LOG="/srv/runs/agent_calls_${mode}.jsonl" \
      -v "$(hostpath "$OUT/logs"):/srv/runs" \
      -v "$(hostpath "$ROOT/agent/app/data/routing.yaml"):/srv/agent/app/data/routing.yaml:ro" \
      -v "$(hostpath "$ROOT/data/cache/sf-docker"):/tmp/sf-cache" \
      "$IMAGE" >/dev/null
    CONTAINERS+=("$name")
  else
    (cd "$ROOT/agent" && env CRMROUTE_MODE="$mode" CRMROUTE_BIG_MODEL="$BIG_MODEL" CRMROUTE_SMALL_MODEL="$SMALL_MODEL" \
        CRMROUTE_POLICY_MODEL="$POLICY_MODEL" CRMROUTE_CALL_LOG="$OUT/logs/agent_calls_${mode}.jsonl" \
        "$AGENT_PY" -m uvicorn app.fast_api_app:app --host 127.0.0.1 --port "$port" \
        > "$OUT/logs/agent_${mode}_${port}.log" 2>&1) & PIDS+=($!)
  fi
  wait_http "http://127.0.0.1:$port/list-apps"
}

manifest() {  # system: record every pin so a result can be audited later
  local image_id=""
  [ "$AGENT_RUNTIME" = docker ] && image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null | cut -c8-19)"
  cat > "$OUT/manifest_$1.json" <<JSON
{"system": "$1", "split": "$SPLIT", "task_ids": "$(hostpath "$TASK_IDS")", "eval_mode": "$EVAL_MODE", "orgs": "$ORGS", "modes": "$MODES",
 "big_model": "$BIG_MODEL", "small_model": "$SMALL_MODEL", "policy_model": "$POLICY_MODEL",
 "judge_model": "$CRMARENA_JUDGE_MODEL", "judge_provider": "$CRMARENA_JUDGE_PROVIDER", "user_sim_model": "$CRMARENA_USER_MODEL",
 "thinking_level": "$CRMARENA_THINKING_LEVEL", "agent_runtime": "$AGENT_RUNTIME", "image": "$IMAGE", "image_id": "$image_id",
 "commit": "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)", "uncommitted_changes": $( [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && echo true || echo false ), "started": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
}

bench() {  # system strategy-args...
  local system="$1"; shift
  manifest "$system"
  export CRMARENA_CALL_LOG="$OUT/logs/bench_calls_${system}.jsonl"
  for org in $ORGS; do
    for mode in $MODES; do
      local flags=()
      [ "$mode" = multi ] && flags+=(--interactive --max_user_turns "$MAX_USER_TURNS")
      echo "== $system | $SPLIT | $org | $mode"
      (cd "$ROOT/vendor/CRMArena" && "$BENCH_PY" -u run_tasks.py --task_category all --task_ids_file "$TASK_IDS" \
          --org_type "$org" --agent_eval_mode "$EVAL_MODE" --reuse_results --log_dir "$OUT/$system" \
          --judge_model "$CRMARENA_JUDGE_MODEL" --judge_provider "$CRMARENA_JUDGE_PROVIDER" \
          --user_model "$CRMARENA_USER_MODEL" --user_provider "$CRMARENA_USER_PROVIDER" \
          "${flags[@]}" "$@") 2>&1 | tee -a "$OUT/logs/${system}_${org}_${mode}.log"
    done
  done
}

[ "$#" -gt 0 ] || { echo "usage: SPLIT=dev|test $0 system..." >&2; exit 2; }
for system in "$@"; do
  case "$system" in
    react)
      bench react --agent_strategy react --model "$BIG_MODEL" --llm_provider "$BASE_PROVIDER" --privacy_aware_prompt false ;;
    react_privacy)
      bench react_privacy --agent_strategy react --model "$BIG_MODEL" --llm_provider "$BASE_PROVIDER" --privacy_aware_prompt true ;;
    full)
      start_mcp; start_agent no_route 8001
      bench full --agent_strategy remote --model crmroute-full --remote_url http://127.0.0.1:8001 ;;
    routed)
      start_mcp; start_agent route 8002
      bench routed --agent_strategy remote --model crmroute-routed --remote_url http://127.0.0.1:8002 ;;
    agent_small)
      start_mcp; start_agent all_small 8003
      bench agent_small --agent_strategy remote --model crmroute-small --remote_url http://127.0.0.1:8003 ;;
    *) echo "unknown system: $system" >&2; exit 2 ;;
  esac
done
echo "results in $OUT"
