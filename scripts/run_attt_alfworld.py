from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from agentic_ttt.alfworld_env import build_alfworld_config, make_alfworld_env
from agentic_ttt.repetition import max_jaccard_repetition
from agentic_ttt.results import atomic_write_json, load_completed_results
from agentic_ttt.trainable_policy import ATrainConfig, TrainableCausalPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen3.5-4B/snapshots/master")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prompt-mode", choices=["react_fewshot", "admissible"], default="react_fewshot")
    parser.add_argument("--cadence", type=int, default=5)
    parser.add_argument("--signal", choices=["self", "env"], default="self")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--no-token-reweight", action="store_true")
    parser.add_argument("--sequence-filter-threshold", type=float)
    parser.add_argument("--ngram-gate", action="store_true")
    parser.add_argument("--skip-loop-updates", action="store_true")
    parser.add_argument("--update-action-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default="/root/autodl-tmp/logs/self_attt_alfworld_result.json")
    args = parser.parse_args()

    output = Path(args.output)
    if not str(output).startswith("/root/autodl-tmp"):
        raise RuntimeError(f"Refusing to write outside /root/autodl-tmp: {output}")
    if args.sequence_filter_threshold is not None and not 0.0 <= args.sequence_filter_threshold <= 1.0:
        raise ValueError("sequence-filter-threshold must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    method = build_method_name(args)
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
    train_config = ATrainConfig(
        adaptive=args.adaptive,
        use_token_reweighting=not args.no_token_reweight and args.sequence_filter_threshold is None,
    )
    policy = TrainableCausalPolicy(
        args.model_path,
        train_config=train_config,
        max_new_tokens=args.max_new_tokens,
        prompt_mode=args.prompt_mode,
    )

    for episode_id in range(len(results), args.episodes):
        policy.reset_episode()
        obs_batch, infos = env.reset()
        obs = obs_batch[0]
        initial_observation = obs
        gamefile = infos["extra.gamefile"][0]
        task = obs.split("\n\n")[-1].strip()
        history: list[tuple[str, str]] = []
        updates: list[dict[str, float | int | str]] = []
        candidate_update_history: list[str] = []
        done = False
        won = False
        steps = 0
        trajectory = []
        invalid_actions = 0

        while not done and steps < args.max_steps:
            admissible = infos["admissible_commands"][0]
            prompt_admissible = admissible
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
            next_obs_batch, _scores, dones, infos = env.step([gen.action])
            next_obs = "OK." if is_thought else next_obs_batch[0]
            done = bool(dones[0])
            won = bool(infos["won"][0])
            update_text = build_update_text(
                signal=args.signal,
                obs=obs,
                next_obs=next_obs,
                action=gen.action,
                model_text=gen.text,
                update_action_only=args.update_action_only,
            )
            if (steps + 1) % args.cadence == 0:
                repetition = max_jaccard_repetition(update_text, candidate_update_history)
                novelty = 1.0 - repetition
                if args.sequence_filter_threshold is not None and novelty < args.sequence_filter_threshold:
                    updates.append(
                        {
                            "step": steps + 1,
                            "loss": 0.0,
                            "skipped": "sequence_filter",
                            "novelty": novelty,
                            "text": update_text[:300],
                        }
                    )
                    print(f"episode={episode_id} update_step={steps + 1} skipped=sequence_filter novelty={novelty:.4f}", flush=True)
                elif args.skip_loop_updates and is_recent_loop(history + [(obs, gen.action)]):
                    updates.append({"step": steps + 1, "loss": 0.0, "skipped": "loop", "text": update_text[:300]})
                    print(f"episode={episode_id} update_step={steps + 1} skipped=loop", flush=True)
                else:
                    loss = policy.update_on_text(update_text)
                    updates.append({"step": steps + 1, "loss": loss, "text": update_text[:300]})
                    print(f"episode={episode_id} update_step={steps + 1} loss={loss:.4f}", flush=True)
                candidate_update_history.append(update_text)
            print(f"episode={episode_id} step={steps} action={gen.action}", flush=True)
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
                "updates": updates,
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


def build_method_name(args: argparse.Namespace) -> str:
    if args.sequence_filter_threshold is not None:
        method = f"{args.signal}_sequence_filter"
    elif args.no_token_reweight:
        method = f"{args.signal}_no_filter"
    else:
        method = f"{args.signal}_adaptive_attt" if args.adaptive else f"{args.signal}_attt"
    if args.ngram_gate:
        method += "_ngram_gate"
    if args.skip_loop_updates:
        method += "_skip_loop_updates"
    if args.update_action_only:
        method += "_action_only"
    return method


def write_summary(*, output: Path, method: str, args: argparse.Namespace, results: list[dict], complete: bool) -> None:
    summary = {
        "method": method,
        "signal": args.signal,
        "token_reweighting": not args.no_token_reweight and args.sequence_filter_threshold is None,
        "sequence_filter_threshold": args.sequence_filter_threshold,
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


def build_update_text(
    *,
    signal: str,
    obs: str,
    next_obs: str,
    action: str,
    model_text: str,
    update_action_only: bool,
) -> str:
    if signal == "env":
        return next_obs.strip()
    if update_action_only:
        return action.strip()
    return model_text.strip() or action.strip()


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


def is_recent_loop(history: list[tuple[str, str]], *, window: int = 6) -> bool:
    actions = [action for _obs, action in history[-window:]]
    if len(actions) < 4:
        return False
    if any(actions.count(action) >= 4 for action in set(actions)):
        return True
    bigrams = [tuple(actions[i : i + 2]) for i in range(len(actions) - 1)]
    return any(bigrams.count(gram) >= 2 for gram in set(bigrams))


if __name__ == "__main__":
    main()
