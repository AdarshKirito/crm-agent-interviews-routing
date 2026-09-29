#!/usr/bin/env bash
# Run the four systems (and the dev-only small-model run) on a fixed task list.
#
#   SPLIT=dev  ./scripts/run_systems.sh react react_privacy full agent_small
#   SPLIT=test ./scripts/run_systems.sh react react_privacy full routed
#
# Systems:
#   react          1. benchmark ReAct baseline
#   react_privacy  2. same agent with --privacy_aware_prompt true
#   full           3. crmroute agent, every task on the big model (CRMROUTE_MODE=no_route)
#   routed         4. crmroute agent with the routing table (CRMROUTE_MODE=route)
#   agent_small    dev only: crmroute agent with every task on the small model
#
# Everything that must stay equal across systems is set once here: model, thinking
# level, eval mode, judge model, simulated-user model and task ids. Runs resume
# (--reuse_results), so an interrupted run can be restarted.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPLIT="${SPLIT:-dev}"
TASK_IDS="$ROOT/data/$SPLIT.json"
OUT="$ROOT/runs/$SPLIT"
EVAL_MODE="${EVAL_MODE:-aided}"
BIG_MODEL="${BIG_MODEL:-gemini-3.8-flash}"
PROVIDER="${PROVIDER:-vertex_ai}"
ORGS="${ORGS:-b2b b2c}"
MODES="${MODES:-single multi}"
MAX_USER_TURNS="${MAX_USER_TURNS:-10}"
export CRMARENA_THINKING_LEVEL="${CRMARENA_THINKING_LEVEL:-low}"
export CRMROUTE_THINKING_LEVEL="$CRMARENA_THINKING_LEVEL"
: "${CRMARENA_JUDGE_MODEL:?set CRMARENA_JUDGE_MODEL (e.g. gpt-4o-2024-08-06, or gemini-3.8-flash)}"
: "${CRMARENA_JUDGE_PROVIDER:?set CRMARENA_JUDGE_PROVIDER (openai or vertex_ai)}"
export CRMARENA_USER_MODEL="${CRMARENA_USER_MODEL:-$CRMARENA_JUDGE_MODEL}"
export CRMARENA_USER_PROVIDER="${CRMARENA_USER_PROVIDER:-$CRMARENA_JUDGE_PROVIDER}"

py() {  # python of a venv on Windows (Scripts) or Linux/macOS (bin)
  if [ -x "$1/.venv/Scripts/python" ]; then echo "$1/.venv/Scripts/python"; else echo "$1/.venv/bin/python"; fi
}
BENCH_PY="$(py "$ROOT/vendor/CRMArena")"
AGENT_PY="$(py "$ROOT/agent")"
SEARCH_BIN="$(dirname "$(py "$ROOT/search")")"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
mkdir -p "$OUT/logs"

wait_http() {  # url
  for _ in $(seq 1 60); do curl -s -m 2 -o /dev/null "$1" && return 0; sleep 1; done
  echo "service at $1 did not come up" >&2; return 1
}

start_mcp() {
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

start_agent() {  # mode port [extra env...]
  local mode="$1" port="$2"; shift 2
  (cd "$ROOT/agent" && env CRMROUTE_MODE="$mode" "$@" "$AGENT_PY" -m uvicorn app.fast_api_app:app --host 127.0.0.1 --port "$port" \
      > "$OUT/logs/agent_${mode}_${port}.log" 2>&1) & PIDS+=($!)
  wait_http "http://127.0.0.1:$port/list-apps"
}

bench() {  # system log_dir strategy-args...
  local system="$1"; shift
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
      bench react --agent_strategy react --model "$BIG_MODEL" --llm_provider "$PROVIDER" --privacy_aware_prompt false ;;
    react_privacy)
      bench react_privacy --agent_strategy react --model "$BIG_MODEL" --llm_provider "$PROVIDER" --privacy_aware_prompt true ;;
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
