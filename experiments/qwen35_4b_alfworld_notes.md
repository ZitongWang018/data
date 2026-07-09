# Qwen3.5-4B ALFWorld Notes

## Setup

- Model: `/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen3.5-4B/snapshots/master`
- Data: `/root/autodl-tmp/data_cache/alfworld`
- Split: `eval_out_of_distribution`
- Step budget: 50
- Prompt: action-first ReAct prompt with admissible actions

## Paper Table 1 Targets

For Qwen3.5-4B on ALFWorld, the paper reports:

- ReAct: 1.9
- Self aTTT: 3.6
- Env aTTT: 4.5
- Summary aTTT: 4.8

## Small-Sample Sanity Results

| Method | Episodes | Success Rate | Notes |
| --- | ---: | ---: | --- |
| ReAct | 5 | 0.0 | Directionally consistent with the low 1.9 paper baseline; trajectories show repeated loops. |
| Loop suppression v1 | 5 | 0.0 | Single-action suppression does not break multi-action loops. |
| Action n-gram gating | 5 | 0.0 | Breaks some repeated action templates and explores more broadly, but still does not solve tasks. |

## Observation

The failed ReAct trajectories repeatedly enter short action templates, e.g. moving an object to a receptacle, taking it back, and moving between the same locations. This supports the paper motivation that harmful repetition is a trajectory-level signal, but a single-action ban is too weak because the loop is usually a repeated action n-gram rather than one repeated action.

## Next Idea

Tried action n-gram novelty gating:

- Track recent 2-gram and 3-gram action patterns.
- If choosing an action would complete a previously repeated action n-gram, down-rank or temporarily mask that action in the prompt.
- Keep the intervention local and reversible so it does not block necessary repeated actions such as opening a fridge after revisiting it.

The first result did not improve success on 5 examples. It did change behavior: instead of repeating one short template until the step limit, the model explores more alternatives. The failure suggests that action-space gating alone is not enough for Qwen3.5-4B because the model often loses the task object or goal after the loop is broken. The next reproduction step should therefore implement the paper's weight-update path, starting with Self-signal aTTT, because it may preserve goal-relevant trajectory information rather than only changing the admissible action list.
