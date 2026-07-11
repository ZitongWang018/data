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
