from __future__ import annotations

import json
from pathlib import Path

from agentic_ttt.alfworld_env import build_alfworld_config, make_alfworld_env


def choose_action(admissible: list[str], high_action: dict) -> str | None:
    action = high_action["discrete_action"]["action"]
    args = [str(arg).lower() for arg in high_action["discrete_action"].get("args", [])]
    if action == "GotoLocation":
        matches = [cmd for cmd in admissible if cmd.startswith("go to ") and args[0] in cmd]
    elif action == "PickupObject":
        matches = [cmd for cmd in admissible if cmd.startswith("take ") and args[0] in cmd]
    elif action in {"CoolObject", "HeatObject", "CleanObject"}:
        verb = {"CoolObject": "cool", "HeatObject": "heat", "CleanObject": "clean"}[action]
        matches = [cmd for cmd in admissible if cmd.startswith(f"{verb} ") and args[0] in cmd]
    elif action == "PutObject":
        obj, recep = args[0], args[1]
        matches = [cmd for cmd in admissible if cmd.startswith("move ") and obj in cmd and recep in cmd]
    else:
        matches = []
    return matches[0] if matches else None


def main() -> None:
    config = build_alfworld_config(num_eval_games=5, max_steps=50)
    env = make_alfworld_env(config, batch_size=1)
    wins = 0
    for episode_id in range(5):
        _obs, infos = env.reset()
        gamefile = Path(infos["extra.gamefile"][0])
        traj = json.loads(gamefile.with_name("traj_data.json").read_text(encoding="utf-8"))
        won = False
        steps = 0
        for high_action in traj["plan"]["high_pddl"]:
            action = choose_action(infos["admissible_commands"][0], high_action)
            if action is None:
                continue
            _obs, _scores, dones, infos = env.step([action])
            won = bool(infos["won"][0])
            steps += 1
            if bool(dones[0]):
                break
        wins += int(won)
        print({"episode": episode_id, "won": won, "steps": steps, "gamefile": str(gamefile)})
    env.close()
    print({"weak_oracle_success_rate": wins / 5 * 100.0})


if __name__ == "__main__":
    main()
