# Agentic TTT Reproduction

Minimal Qwen3.5-4B reproduction of *No Time Like the Present: Agentic
Test-Time Training for LLM Agents* on ALFWorld.

## Implemented comparisons

- ReAct without TTT.
- Online TTT without repetition filtering.
- Sequence-level update filtering.
- Agentic TTT with tokenizer-level 3-gram exposure weights.
- Progress-buffered Env aTTT plus optional adaptive and loop-control extensions.

The default `react_fewshot` prompt uses the original task-specific two-shot
ALFWorld demonstrations from the MIT-licensed ReAct repository. It does not
show the model `admissible_commands`. The old action-list prompt remains
available as `--prompt-mode admissible` for diagnostics only.

## Remote layout

- Worktree: `/root/autodl-tmp/agentic_ttt_repro`
- ModelScope model cache: `/root/autodl-tmp/modelscope_cache`
- ALFWorld data: `/root/autodl-tmp/data_cache/alfworld`
- Experiment output: `/root/autodl-tmp/logs`

## Run

```bash
cd /root/autodl-tmp/agentic_ttt_repro
export USE_NETWORK_TURBO=0
source scripts/env_autodl.sh
export ALFWORLD_DATA=/root/autodl-tmp/data_cache/alfworld
export PYTHONPATH=src
export TRANSFORMERS_OFFLINE=1

python scripts/run_react_alfworld.py \
  --episodes 134 --max-steps 50 --resume \
  --output /root/autodl-tmp/logs/react_paper_134.json

python scripts/run_attt_alfworld.py \
  --episodes 134 --max-steps 50 --cadence 5 --signal env --resume \
  --output /root/autodl-tmp/logs/env_attt_paper_134.json
```

Both runners sort the complete `valid_unseen` game list before slicing, record
the selected gamefile, and atomically checkpoint after every episode. By
default, they interleave the six ALFWorld task families with a fixed within-
family seed, preventing a resumable prefix from being dominated by one task
family. Use `--game-order sorted` only when reproducing a legacy diagnostic.
## Interaction note

 actions are prompt-only: they append  to the trajectory and do
not call . Agent-turn budget still increments.
