#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/agentic_ttt_repro
export USE_NETWORK_TURBO=0
source scripts/env_autodl.sh
export ALFWORLD_DATA=/root/autodl-tmp/data_cache/alfworld
export PYTHONPATH=src
export TRANSFORMERS_OFFLINE=1

MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen3.5-4B/snapshots/master
LOGS=/root/autodl-tmp/logs
COMMON=(--model-path "$MODEL" --episodes 134 --max-steps 50 --seed 0 --resume)

/root/miniconda3/bin/python scripts/run_react_alfworld.py \
  "${COMMON[@]}" \
  --output "$LOGS/react_paper_seed0_134.json"

/root/miniconda3/bin/python scripts/run_attt_alfworld.py \
  "${COMMON[@]}" --cadence 5 --signal env --no-token-reweight \
  --output "$LOGS/env_no_filter_paper_seed0_134.json"

/root/miniconda3/bin/python scripts/run_attt_alfworld.py \
  "${COMMON[@]}" --cadence 5 --signal env \
  --output "$LOGS/env_attt_paper_seed0_134.json"

/root/miniconda3/bin/python scripts/run_attt_alfworld.py \
  "${COMMON[@]}" --cadence 5 --signal env --progress-buffered-env \
  --output "$LOGS/env_progress_attt_seed0_134.json"

/root/miniconda3/bin/python scripts/summarize_table1_results.py \
  "$LOGS/react_paper_seed0_134.json" \
  "$LOGS/env_no_filter_paper_seed0_134.json" \
  "$LOGS/env_attt_paper_seed0_134.json" \
  "$LOGS/env_progress_attt_seed0_134.json"
