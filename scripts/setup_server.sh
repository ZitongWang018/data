#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/agentic_ttt_repro
source scripts/env_autodl.sh
export PATH=/root/miniconda3/bin:$PATH

/root/miniconda3/bin/python -m pip install --upgrade pip
/root/miniconda3/bin/python -m pip install -r requirements.txt
