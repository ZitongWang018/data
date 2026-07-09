from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_ttt.alfworld_env import build_alfworld_config, make_alfworld_env
from agentic_ttt.llm_policy import LocalCausalPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen3.5-4B/snapshots/master")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--loop-suppress", action="store_true")
    parser.add_argument("--output", default="/root/autodl-tmp/logs/react_alfworld_result.json")
    args = parser.parse_args()

    output = Path(args.output)
    if not str(output).startswith("/root/autodl-tmp"):
        raise RuntimeError(f"Refusing to write outside /root/autodl-tmp: {output}")

    config = build_alfworld_config(num_eval_games=args.episodes, max_steps=args.max_steps)
    env = make_alfworld_env(config, batch_size=1)
    policy = LocalCausalPolicy(args.model_path, max_new_tokens=args.max_new_tokens)
    results = []

    for episode_id in range(args.episodes):
        obs_batch, infos = env.reset()
        obs = obs_batch[0]
        task = obs.split("\n\n")[-1].strip()
        history: list[tuple[str, str]] = []
        done = False
        won = False
        steps = 0
        trajectory = []

        while not done and steps < args.max_steps:
            admissible = infos["admissible_commands"][0]
            prompt_admissible = suppress_loop_actions(history, admissible) if args.loop_suppress else admissible
            gen = policy.generate_action(
                task=task,
                observation=obs,
                history=history,
                admissible_actions=prompt_admissible,
            )
            if gen.action not in admissible:
                gen.action = "look" if "look" in admissible else admissible[0]
            print(f"episode={episode_id} step={steps} action={gen.action}", flush=True)
            next_obs_batch, _scores, dones, infos = env.step([gen.action])
            next_obs = next_obs_batch[0]
            done = bool(dones[0])
            won = bool(infos["won"][0])
            trajectory.append({"obs": obs, "raw": gen.text, "action": gen.action, "won": won})
            history.append((obs, gen.action))
            obs = next_obs
            steps += 1

        results.append({"episode": episode_id, "won": won, "steps": steps, "trajectory": trajectory})
        print(f"episode={episode_id} won={won} steps={steps}", flush=True)

    env.close()
    summary = {
        "method": "react",
        "episodes": args.episodes,
        "success_rate": sum(1 for row in results if row["won"]) / max(1, len(results)) * 100.0,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"success_rate": summary["success_rate"], "episodes": args.episodes}, indent=2), flush=True)


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


if __name__ == "__main__":
    main()
