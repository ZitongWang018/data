# Project State

Updated: 2026-07-12

## Current target

Produce a deterministic Qwen3.5-4B ALFWorld table under the author-feedback
first-version setting: observation text plus current admissible action list,
zero-shot `Thought:`/`Action:` output, temperature 0 / greedy decoding,
max_tokens 2048, max_steps 50, and early stop after three identical repeated
actions. The HF/PEFT fallback path uses `--device-map cuda` to avoid invalid
CPU offload for Qwen3.5 linear-attention kernels. Compare ReAct-style baseline,
Env no-filter, Env aTTT, and progress-buffered Env aTTT against the paper's
Table 1, while noting this is not the paper's standard few-shot ReAct prompt.

## Working infrastructure

- Remote repo: `/root/autodl-tmp/agentic_ttt_repro`
- Model: `/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen3.5-4B/snapshots/master`
- ALFWorld: `/root/autodl-tmp/data_cache/alfworld`
- Logs: `/root/autodl-tmp/logs`
- Paper table logs: `/root/autodl-tmp/logs/table1_thinkfix`
- GitHub: `ZitongWang018/data`, branch `main`

## Verified behavior

- ALFWorld exposes live `won`, `admissible_commands`, and `extra.gamefile`.
- The weak oracle can complete tasks, so the reward path is functional.
- Game files are now fully collected, sorted, and sliced deterministically.
- The current default baseline uses `--prompt-mode author_admissible`, exposing
  live `admissible_commands` and asking for `Thought:` plus exact-list
  `Action:` with no shots.
- `react_fewshot` remains available for paper-style prompt diagnostics without
  exposing admissible actions.
- `think:` actions no longer call `env.step`; they only append `OK.` to the
  prompt trajectory, matching standard ReAct semantics.
- Env aTTT performs non-zero LoRA updates and resets the adapter between
  episodes while loading the base model only once per run.
- Results are atomically checkpointed after every episode and can resume.

## In progress

- Relaunch the full interleaved seed-0 author-aligned matrix, preferably under
  `/root/autodl-tmp/logs/table1_author_admissible`: baseline, Env no-filter,
  Env aTTT, and Env progress-buffered. Use vLLM for the baseline if installed;
  keep online TTT on HF/PEFT for LoRA updates.
- Final table and discrepancy analysis after the author-aligned matrix
  completes.

## Invalidated / superseded evidence

- Earlier five-episode 0% runs were smoke tests only.
- The interrupted 47-episode baseline reached 7/47, but it used an
  admissible-action prompt and nondeterministic file ordering.
- The 13-episode `react_fewshot_fixed_look13` result reached 8/13, but it is
  invalid for table comparison because lexical file ordering concentrated its
  prefix in the easy `look_at_obj_in_light` task family.
- `react_table1_interleaved_134.json` (7.46%) and
  `env_no_filter_table1_interleaved_134.json` (6.72%) used the pre-fix
  `think:`?`env.step` path, so they are retained only as pre-fix diagnostics,
  not as the final paper-comparison table.
- Any `react_fewshot` table should be reported separately from the
  author-feedback setting because the author setting exposes legal actions and
  has no shots.
