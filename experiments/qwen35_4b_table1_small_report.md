# Qwen3.5-4B ALFWorld Small-Sample Reproduction

Date: 2026-07-10

## Scope

This report records the current small-sample reproduction and sanity checks for the minimal Qwen3.5-4B ALFWorld setup. It is not a statistically reliable replacement for the paper's 140-episode Table 1. The goal here is to verify that the code path is sane before spending the much larger compute budget required for a faithful table.

## Environment

- Remote project: `/root/autodl-tmp/agentic_ttt_repro`
- Model: `/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen3.5-4B/snapshots/master`
- Data: `/root/autodl-tmp/data_cache/alfworld`
- Split: `eval_out_of_distribution`
- Step budget: `50`
- Offline model loading: `TRANSFORMERS_OFFLINE=1`
- All logs/results: `/root/autodl-tmp/logs`

## Sanity Checks

- ALFWorld reset exposes `admissible_commands`, `extra.gamefile`, and `won`.
- `check_alfworld_infos.py` verifies action stepping and `won` field access.
- `check_oracle_alfworld.py` uses the stored ALFRED high-level plan and a weak admissible-action matcher. It can trigger `won=True` on solvable matched episodes, confirming the environment/reward path is live.
- Result JSON success rates were manually recomputed from `results[*].won`; they match the stored `success_rate`.
- Raw generations are non-empty. Parser failures are mostly caused by generated text containing actions not in the current admissible set, which then safely falls back to `look`.
- `--start-index` was added and smoke-tested to avoid relying only on the first five games.

## Results

| Method | Start index | Episodes | Wins | Success rate | Paper target in config |
|---|---:|---:|---:|---:|---:|
| ReAct | 0 | 5 | 0 | 0.0% | 1.9% |
| ReAct | 20 | 5 | 0 | 0.0% | 1.9% |
| Self-aTTT, tokenizer-level weights | 0 | 5 | 0 | 0.0% | 3.6% |
| Adaptive Self-aTTT, tokenizer-level weights | 0 | 5 | 0 | 0.0% | 4.8% summary-aTTT target |
| New idea: adaptive aTTT + action n-gram gate + skip loop updates | 0 | 5 | 0 | 0.0% | n/a |
| New idea: adaptive aTTT + action n-gram gate + skip loop updates + action-only update text | 0 | 5 | 0 | 0.0% | n/a |

## Interpretation

The repeated 0.0% result should not be over-interpreted as a faithful paper mismatch. With only five episodes per method, a paper-level success rate around 2-5% can easily produce zero observed successes. However, the logs show a real behavioral issue: Qwen3.5-4B repeatedly falls into loops such as `look`, object examine loops, receptacle open/close loops, or take/move cycles.

The first Self-aTTT implementation had a bug-risk: whitespace-token repetition weights often did not align with Qwen tokenizer tokens and could fall back to all-ones. This was fixed with tokenizer-id n-gram weighting. Even after that fix, Self-aTTT still tended to reinforce bad trajectories. The new gate/skip/action-only variants reduce obvious self-training damage but did not solve tasks in the current small sample.

## Concrete Reflections

- Code path is not obviously broken: oracle can trigger `won=True`, JSON stats match manual counts, and model actions are admissible when executed.
- The prompt/parser path is imperfect but not silently empty: Qwen often starts with a valid action, then continues with extra hallucinated text.
- Self-training on raw model text is risky because raw generations contain hallucinated future observations/actions.
- Action-only update text is cleaner but still not enough for success in the tested sample.
- The model may need stronger instruction tuning, better ALFWorld-specific prompting, environment-summary memory, or a larger evaluation sample to observe the low expected baseline rate.

## Recommended Next Run

Run a larger sample before judging reproduction quality:

```bash
TRANSFORMERS_OFFLINE=1 ALFWORLD_DATA=/root/autodl-tmp/data_cache/alfworld PYTHONPATH=src \
/root/miniconda3/bin/python scripts/run_react_alfworld.py \
  --model-path /root/autodl-tmp/modelscope_cache/models/Qwen--Qwen3.5-4B/snapshots/master \
  --episodes 50 --max-steps 50 \
  --output /root/autodl-tmp/logs/react_50.json
```

Then run the same 50-episode slice for `self_attt`, `self_adaptive_attt`, and the action-only gate/skip idea. The full table should ultimately use all 134 available `valid_unseen` games or the configured 140-equivalent protocol if the exact paper split is recovered.
