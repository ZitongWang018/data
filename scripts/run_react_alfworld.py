from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from agentic_ttt.alfworld_env import build_alfworld_config, make_alfworld_env
from agentic_ttt.llm_policy import LocalCausalPolicy
from agentic_ttt.results import atomic_write_json, load_completed_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen3.5-4B/snapshots/master")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prompt-mode", choices=["react_fewshot", "admissible"], default="react_fewshot")
    parser.add_argument("--loop-suppress", action="store_true")
    parser.add_argument("--ngram-gate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default="/root/autodl-tmp/logs/react_alfworld_result.json")
    args = parser.parse_args()

    output = Path(args.output)
    if not str(output).startswith("/root/autodl-tmp"):
        raise RuntimeError(f"Refusing to write outside /root/autodl-tmp: {output}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    method = "react"
    if args.loop_suppress:
        method += "_loop_suppress"
    if args.ngram_gate:
        method += "_ngram_gate"
    results = load_completed_results(
        output,
        resume=args.resume,
        expected={"method": method, "start_index": args.start_index, "seed": args.seed, "prompt_mode": args.prompt_mode},
    )
    if len(results) > args.episodes:
        raise ValueError(f"Output already has {len(results)} episodes, more than requested {args.episodes}")
    remaining = args.episodes - len(results)
    if remaining == 0:
        print(f"Result already complete: {output}", flush=True)
        return

    config = build_alfworld_config(num_eval_games=0, max_steps=args.max_steps)
    env = make_alfworld_env(
        config,
        batch_size=1,
        start_index=args.start_index + len(results),
        num_games=remaining,
    )
    policy = LocalCausalPolicy(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        prompt_mode=args.prompt_mode,
    )

    for episode_id in range(len(results), args.episodes):
        obs_batch, infos = env.reset()
        obs = obs_batch[0]
        initial_observation = obs
        gamefile = infos["extra.gamefile"][0]
        task = obs.split("\n\n")[-1].strip()
        history: list[tuple[str, str]] = []
        done = False
        won = False
        steps = 0
        trajectory = []
        invalid_actions = 0

        while not done and steps < args.max_steps:
            admissible = infos["admissible_commands"][0]
            prompt_admissible = admissible
            if args.loop_suppress:
                prompt_admissible = suppress_loop_actions(history, prompt_admissible)
            if args.ngram_gate:
                prompt_admissible = suppress_repeated_action_ngrams(history, prompt_admissible)
            gen = policy.generate_action(
                task=task,
                observation=obs,
                history=history,
                admissible_actions=prompt_admissible,
                initial_observation=initial_observation,
                gamefile=gamefile,
            )
            is_thought = gen.action.lower().startswith("think:")
            if args.prompt_mode == "admissible" and gen.action not in admissible:
                gen.action = "look" if "look" in admissible else admissible[0]
            elif not is_thought and gen.action not in admissible:
                invalid_actions += 1
            print(f"episode={episode_id} step={steps} action={gen.action}", flush=True)
            next_obs_batch, _scores, dones, infos = env.step([gen.action])
            next_obs = "OK." if is_thought else next_obs_batch[0]
            done = bool(dones[0])
            won = bool(infos["won"][0])
            trajectory.append({"obs": obs, "raw": gen.text, "action": gen.action, "won": won})
            history.append((obs, gen.action))
            obs = next_obs
            steps += 1

        results.append(
            {
                "episode": episode_id,
                "game_index": args.start_index + episode_id,
                "gamefile": gamefile,
                "won": won,
                "steps": steps,
                "invalid_actions": invalid_actions,
                "trajectory": trajectory,
            }
        )
        print(f"episode={episode_id} won={won} steps={steps}", flush=True)
        write_summary(
            output=output,
            method=method,
            args=args,
            results=results,
            complete=len(results) == args.episodes,
        )

    env.close()


def write_summary(*, output: Path, method: str, args: argparse.Namespace, results: list[dict], complete: bool) -> None:
    summary = {
        "method": method,
        "prompt_mode": args.prompt_mode,
        "seed": args.seed,
        "episodes": args.episodes,
        "completed_episodes": len(results),
        "complete": complete,
        "start_index": args.start_index,
        "success_rate": sum(1 for row in results if row["won"]) / max(1, len(results)) * 100.0,
        "invalid_action_rate": sum(row["invalid_actions"] for row in results)
        / max(1, sum(row["steps"] for row in results))
        * 100.0,
        "results": results,
    }
    atomic_write_json(output, summary)
    print(
        f"checkpoint method={method} completed={len(results)}/{args.episodes} "
        f"success_rate={summary['success_rate']:.3f}",
        flush=True,
    )


def suppress_loop_actions(
    history: list[tuple[str, str]],
    admissible: list[str],
    *,
    window: int = 6,
    threshold: int = 3,
) -> list[str]:
    recent = [action for _obs, action in history[-window:]]
    banned = {action for action in set(recent) if recent.count(action) >= threshold}
    filtered = [action for action in admissible if action not in banned]
    return filtered if filtered else admissible


def suppress_repeated_action_ngrams(
    history: list[tuple[str, str]],
    admissible: list[str],
    *,
    ngram_sizes: tuple[int, ...] = (2, 3),
    min_count: int = 2,
) -> list[str]:
    actions = [action for _obs, action in history]
    if len(actions) < min(ngram_sizes):
        return admissible

    banned: set[str] = set()
    for ngram_size in ngram_sizes:
        if len(actions) < ngram_size - 1:
            continue
        prefix = tuple(actions[-(ngram_size - 1) :])
        counts: dict[tuple[str, ...], int] = {}
        for start in range(0, len(actions) - ngram_size + 1):
            gram = tuple(actions[start : start + ngram_size])
            counts[gram] = counts.get(gram, 0) + 1
        for action in admissible:
            candidate = prefix + (action,)
            if counts.get(candidate, 0) >= min_count:
                banned.add(action)

    filtered = [action for action in admissible if action not in banned]
    return filtered if filtered else admissible


if __name__ == "__main__":
    main()
