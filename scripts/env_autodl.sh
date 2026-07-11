#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT=/root/autodl-tmp/agentic_ttt_repro
export AUTODL_TMP=/root/autodl-tmp

export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export HF_DATASETS_CACHE=/root/autodl-tmp/data_cache/hf_datasets
export HF_HUB_DISABLE_XET=1
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
export TORCH_HOME=/root/autodl-tmp/torch_cache
export XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TRITON_CACHE_DIR=/root/autodl-tmp/triton_cache
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions
export CUDA_CACHE_PATH=/root/autodl-tmp/cuda_cache

export WANDB_DIR=/root/autodl-tmp/logs/wandb
export WANDB_CACHE_DIR=/root/autodl-tmp/logs/wandb_cache
export TMPDIR=/root/autodl-tmp/tmp

mkdir -p \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$MODELSCOPE_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$CUDA_CACHE_PATH" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$TMPDIR"

if [ "${USE_NETWORK_TURBO:-1}" = "1" ] && [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
fi
