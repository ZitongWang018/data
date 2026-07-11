# Agentic TTT Reproduction Rules

## Scope

Reproduce the smallest-model ALFWorld comparison from *No Time Like the
Present: Agentic Test-Time Training for LLM Agents* using Qwen3.5-4B. The
minimum result set is ReAct, Env no-filter online TTT, Env aTTT, and one
motivation-driven extension.

## Storage boundary

- Execute experiments only on the remote server.
- Keep the remote worktree, models, datasets, caches, temporary files, and
  logs below `/root/autodl-tmp`.
- Source `scripts/env_autodl.sh` before running setup or experiments.
- Download models with ModelScope after unsetting HTTP(S) proxy variables.
- Do not sync model, dataset, cache, or full trajectory artifacts to the local
  mirror or GitHub.

## Reproducibility

- Use the deterministic sorted ALFWorld game order and record each gamefile.
- Treat runs below 30 episodes as diagnostics, not paper-level evidence.
- Use `--resume` for long runs; runners checkpoint atomically after each
  episode.
- Keep the paper defaults unless the result name records the change: 50 steps,
  cadence 5, LoRA rank 8, alpha 16, learning rate 5e-4, two update steps,
  tokenizer 3-grams, and minimum weight 0.05.
- The paper-aligned Qwen3.5-4B update source is Env.

## Synchronization

- Canonical remote worktree: `/root/autodl-tmp/agentic_ttt_repro`.
- Canonical branch: `main` in `ZitongWang018/data`.
- Commit and push source changes from the server, then synchronize code to the
  local `agentic_ttt_repro` mirror.
- Update `PROJECT_STATE.md` and append `MEMORY.md` after material changes.
