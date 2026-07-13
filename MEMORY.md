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

## 2026-07-13 (author-feedback alignment)

- Author feedback says the first-version setting is not standard few-shot
  ReAct: the model sees the environment observation plus the current legal
  action list, uses a zero-shot `Thought:` / `Action:` prompt, temperature 0,
  `max_tokens=2048`, at most 50 steps, and early-stops after three identical
  repeated actions.
- Added `--prompt-mode author_admissible` as the default and kept
  `react_fewshot` only as a separate diagnostic mode.
- Added `--repeat-action-stop 3`; baseline can optionally use `--backend vllm`
  when vLLM is installed, while online TTT keeps HF/PEFT because LoRA updates
  require a trainable model.
- HF/PEFT now defaults to `--device-map cuda`; `device_map=auto` can offload
  Qwen3.5 linear-attention/causal-conv layers to CPU under memory pressure and
  crash with `Expected x.is_cuda() to be true`.
