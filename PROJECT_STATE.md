# Project State

Updated: 2026-07-12

## Current target

Produce a deterministic Qwen3.5-4B ALFWorld table containing ReAct, Env
no-filter, Env aTTT, and a new repetition-control idea, then compare the
observed deltas with the paper's Table 1.

## Working infrastructure

- Remote repo: `/root/autodl-tmp/agentic_ttt_repro`
- Model: `/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen3.5-4B/snapshots/master`
- ALFWorld: `/root/autodl-tmp/data_cache/alfworld`
- Logs: `/root/autodl-tmp/logs`
- GitHub: `ZitongWang018/data`, branch `main`

## Verified behavior

- ALFWorld exposes live `won`, `admissible_commands`, and `extra.gamefile`.
- The weak oracle can complete tasks, so the reward path is functional.
- Game files are now fully collected, sorted, and sliced deterministically.
- The default baseline uses the original task-specific two-shot ReAct prompts
  without exposing admissible actions.
- A one-episode paper-style smoke run produced a 30% invalid-action rate,
  matching the paper's reported Qwen3.5-4B failure signature much better than
  the old admissible-action prompt.
- Env aTTT performs non-zero LoRA updates and resets the adapter between
  episodes while loading the base model only once per run.
- Results are atomically checkpointed after every episode and can resume.

## In progress

- Deterministic larger-sample ReAct, Env no-filter, and Env aTTT evaluation.
- Motivation-driven extension and matched-slice comparison.
- Final table and discrepancy analysis.

## Invalidated evidence

- Earlier five-episode 0% runs were smoke tests only.
- The interrupted 47-episode baseline reached 7/47, but it used an
  admissible-action prompt and nondeterministic file ordering, so it is retained
  only as evidence that the old harness inflated performance, not as a paper
  reproduction result.
