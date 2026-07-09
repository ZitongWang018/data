# Agentic TTT Reproduction

This repository contains a minimal reproduction scaffold for:

`No Time Like the Present: Agentic Test-Time Training for LLM Agents`

## What is implemented

- Repetition-aware token weighting used by aTTT.
- A baseline runner that behaves like no-TTT / ReAct.
- A small smoke-test episode runner that validates the online update flow.
- Server-side cache and workspace conventions for `/root/autodl-tmp`.

## What is not yet fully implemented

- Full ALFWorld / SWE-bench Lite evaluation.
- Full multi-GPU concurrent serving stack.
- Large-model throughput benchmarking.

## Suggested local / server layout

- Local mirror: `./agentic_ttt_repro`
- Server worktree: `/root/autodl-tmp/agentic_ttt_repro`
- Model cache: `/root/autodl-tmp/hf_cache`
- Data cache: `/root/autodl-tmp/data_cache`
- Logs: `/root/autodl-tmp/logs`

## Default Reproduction Target

The first server run targets the smallest paper model:

- Model: `Qwen3.5-4B`
- Baseline: no-TTT / ReAct
- Method: aTTT
- Cadence: `K=5`
- LoRA: rank `8`, alpha `16`
- Learning rate: `5e-4`
- Repetition weighting: 3-gram, `w_min=0.05`

Before downloading models or datasets on the server, run:

```bash
source /root/autodl-tmp/agentic_ttt_repro/scripts/env_autodl.sh
```
