# Project State

Updated: 2026-07-12

## Current target

Produce a deterministic Qwen3.5-4B ALFWorld table containing ReAct, Env
no-filter, Env sequence filter, Env aTTT, and progress-buffered Env aTTT,
then compare the observed deltas with the paper's Table 1.

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
- The default baseline uses the original task-specific two-shot ReAct prompts
  without exposing admissible actions.
- `think:` actions no longer call `env.step`; they only append `OK.` to the
  prompt trajectory, matching standard ReAct semantics.
- Env aTTT performs non-zero LoRA updates and resets the adapter between
  episodes while loading the base model only once per run.
- Results are atomically checkpointed after every episode and can resume.

## In progress

- Relaunching the full interleaved seed-0 paper matrix under
  `/root/autodl-tmp/logs/table1_thinkfix` after the think-step fix:
  ReAct, Env no-filter, Env sequence filter, Env aTTT, Env progress-buffered.
- Final table and discrepancy analysis after the matrix completes.

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
