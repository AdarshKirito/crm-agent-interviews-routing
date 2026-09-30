#!/usr/bin/env bash
# Resume the free-tier test run after the daily Gemini quota resets (midnight Pacific).
#   scripts/resume_test_small.sh agent   # crmroute: full, then routed
#   scripts/resume_test_small.sh react   # ReAct baseline, then ReAct + privacy prompt
# Completed tasks are skipped; tasks that ended in an API error are redone. The pins
# are the ones used for every measured run (see README, "Run the benchmark").
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
group="${1:?usage: $0 agent|react}"
export BIG_MODEL=gemini-3.1-flash-lite SMALL_MODEL=ollama_chat/qwen3:8b POLICY_MODEL=groq/openai/gpt-oss-20b
export CRMARENA_JUDGE_MODEL=ollama_chat/qwen3:8b CRMARENA_JUDGE_PROVIDER=ollama_chat
export TASK_IDS=data/test_small.json OUT=runs/test_small SPLIT=test_small MODES=single
export PATH="$ROOT/.tools/node-v24.19.0-win-x64:$PATH"
log="runs/test_small_resume_${group}_$(date -u +%Y%m%dT%H%MZ).out"
mkdir -p runs
# Docker Desktop and Ollama must be up for the agent systems and the judge.
for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && curl -s -m 2 -o /dev/null http://localhost:11434/api/tags && break; sleep 10; done
case "$group" in
  agent) bash scripts/run_systems.sh full routed > "$log" 2>&1 ;;
  react) bash scripts/run_systems.sh react react_privacy > "$log" 2>&1 ;;
  *) echo "unknown group $group" >&2; exit 2 ;;
esac
echo "exit=$?" >> "$log"
