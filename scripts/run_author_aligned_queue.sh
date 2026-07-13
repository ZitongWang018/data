#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/agentic_ttt_repro}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs/table1_author_admissible}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MIN_FREE_MIB="${MIN_FREE_MIB:-28000}"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "${LOG_DIR}"
exec 9>"${LOG_DIR}/author_queue.lock"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) another author-aligned queue is already active"
  exit 0
fi

echo "$$" >"${LOG_DIR}/author_queue.pid"
trap 'status=$?; echo "$(date --iso-8601=seconds) queue exited status=${status}"; exit "${status}"' EXIT

cd "${REPO_DIR}"
export USE_NETWORK_TURBO=0
export ALFWORLD_DATA=/root/autodl-tmp/data_cache/alfworld
export PYTHONPATH=src
export TRANSFORMERS_OFFLINE=1
source scripts/env_autodl.sh

echo "$(date --iso-8601=seconds) waiting for GPU: required_free_mib=${MIN_FREE_MIB}"
while true; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  active_run="$(pgrep -f '[r]un_(react|attt)_alfworld.py' || true)"
  if [[ -z "${active_run}" && "${free_mib}" =~ ^[0-9]+$ && "${free_mib}" -ge "${MIN_FREE_MIB}" ]]; then
    sleep 10
    stable_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "${stable_free_mib}" =~ ^[0-9]+$ && "${stable_free_mib}" -ge "${MIN_FREE_MIB}" ]]; then
      echo "$(date --iso-8601=seconds) GPU ready: free_mib=${stable_free_mib}"
      break
    fi
  fi
  echo "$(date --iso-8601=seconds) still waiting: free_mib=${free_mib} active_agentic_pid=${active_run:-none}"
  sleep "${POLL_SECONDS}"
done

COMMON_ARGS=(
  --episodes 134
  --max-steps 50
  --max-new-tokens 2048
  --prompt-mode author_admissible
  --repeat-action-stop 3
  --device-map cuda
  --game-order interleaved
  --order-seed 0
  --seed 0
  --resume
)

run_stage() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  echo "$(date --iso-8601=seconds) starting ${name}"
  "$@" >>"${log_file}" 2>&1
  echo "$(date --iso-8601=seconds) completed ${name}"
}

run_stage react_134 \
  "${PYTHON_BIN}" scripts/run_react_alfworld.py \
  "${COMMON_ARGS[@]}" \
  --backend hf \
  --output "${LOG_DIR}/react_134.json"

run_stage env_no_filter_134 \
  "${PYTHON_BIN}" scripts/run_attt_alfworld.py \
  "${COMMON_ARGS[@]}" \
  --signal env \
  --no-token-reweight \
  --output "${LOG_DIR}/env_no_filter_134.json"

run_stage env_attt_134 \
  "${PYTHON_BIN}" scripts/run_attt_alfworld.py \
  "${COMMON_ARGS[@]}" \
  --signal env \
  --output "${LOG_DIR}/env_attt_134.json"

run_stage env_progress_134 \
  "${PYTHON_BIN}" scripts/run_attt_alfworld.py \
  "${COMMON_ARGS[@]}" \
  --signal env \
  --progress-buffered-env \
  --output "${LOG_DIR}/env_progress_134.json"

touch "${LOG_DIR}/author_queue.complete"
echo "$(date --iso-8601=seconds) all author-aligned stages completed"
