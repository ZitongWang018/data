# Project Memory

## 2026-07-10

- Built the Qwen3.5-4B ModelScope and ALFWorld environment entirely below
  `/root/autodl-tmp`.
- Added tokenizer-ID n-gram aTTT weighting, Env/Self update sources, slice-aware
  runners, sanity checks, and GitHub deploy-key synchronization.
- Tiny five-episode 0% runs were found to be misleading; a longer legacy ReAct
  run produced wins.

## 2026-07-12

- Migrated to the replacement SSH endpoint and verified that repo, model, data,
  and logs survived.
- Recovered 7 wins from 47 completed legacy baseline episodes, but invalidated
  the run for final comparison because its prompt exposed admissible actions.
- Replaced the legacy prompt with the original ALFWorld two-shot ReAct prompt,
  removed action-list leakage, sorted the full game set before slicing, and
  recorded gamefile identities.
- Added per-episode atomic checkpoints, resume support, fixed seeds, and LoRA
  reset without reloading the base model.
- Corrected update text to use Env observations and Self model text verbatim.
- Added progress-buffered Env aTTT, which selects the latest valid state-changing
  observation in each cadence window and skips windows containing only no-ops.
- Replaced lexical-prefix evaluation with a deterministic interleaved task-family
  order (fixed seed within each family) after discovering that a 13-episode
  sorted prefix was almost entirely `look_at_obj_in_light` and produced an
  unusable 61.5% apparent ReAct score.
- Started the first formal 134-episode interleaved ReAct checkpoint run; all
  follow-up methods must use the same order and seed.

## 2026-07-12 (think-fix relaunch)

- Fixed ReAct/`aTTT` runners so `think:` does not call `env.step`; prompt still
  records `OK.` and the agent turn still consumes one step budget.
- Changed `run_attt_alfworld.py` default `--signal` to `env` for the 4B table.
- Superseded pre-fix interleaved 134 ReAct / Env no-filter logs; relaunched the
  matched paper matrix under `/root/autodl-tmp/logs/table1_thinkfix`.
