# Progress-Buffered Env aTTT

## Motivation

Qwen3.5-4B has a high invalid-action rate. A cadence boundary can therefore
land on a `think:` message, an unparseable command, `look`, or another no-op.
Vanilla Env aTTT then trains on `OK.`, `Nothing happens`, or an observation that
does not represent progress. Repetition weighting limits repeated tokens, but
it does not decide whether the selected observation is useful.

## Method

Within each five-step cadence window, retain the latest observation produced by
a valid, non-thought, non-`look`, non-`inventory` action that changes the
observation. At the cadence boundary, apply the standard paper aTTT loss to that
buffered observation. Skip the update if the window contains no such
transition. Reset the buffer and LoRA adapter on the same schedule as the paper
method.

This changes source selection only; LoRA rank, alpha, learning rate, gradient
steps, token 3-grams, weight floor, prompt, episode order, and step budget stay
matched to Env aTTT.

## Evaluation

Compare on identical sorted ALFWorld games:

- ReAct
- Env no-filter
- Env aTTT
- Env progress-buffered aTTT

Primary metric: success rate. Diagnostics: invalid-action rate, fraction of
skipped no-progress updates, paired rescues/regressions relative to Env aTTT,
and success by task type.

The idea is considered promising only if it improves the matched full-slice
success rate or produces a credible positive paired delta. A tiny-slice gain is
not sufficient.
