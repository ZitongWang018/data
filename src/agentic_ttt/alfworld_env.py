from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def build_alfworld_config(
    *,
    data_root: str = "/root/autodl-tmp/data_cache/alfworld",
    split: str = "eval_out_of_distribution",
    num_eval_games: int = 1,
    max_steps: int = 50,
) -> dict[str, Any]:
    root = Path(data_root)
    if not str(root).startswith("/root/autodl-tmp"):
        raise RuntimeError(f"Refusing to use ALFWorld data outside /root/autodl-tmp: {root}")

    return {
        "general": {
            "training_method": "dqn",
        },
        "dataset": {
            "data_path": str(root / "json_2.1.1/train"),
            "eval_id_data_path": str(root / "json_2.1.1/valid_seen"),
            "eval_ood_data_path": str(root / "json_2.1.1/valid_unseen"),
            "num_train_games": 0,
            "num_eval_games": num_eval_games,
        },
        "env": {
            "type": "AlfredTWEnv",
            "task_types": [1, 2, 3, 4, 5, 6],
            "goal_desc_human_anns_prob": 0,
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "logic": {
            "domain": str(root / "logic/alfred.pddl"),
            "grammar": str(root / "logic/alfred.twl2"),
        },
        "rl": {
            "training": {
                "max_nb_steps_per_episode": max_steps,
            }
        },
        "dagger": {
            "training": {
                "max_nb_steps_per_episode": max_steps,
            }
        },
        "split": split,
    }


def make_alfworld_env(config: dict[str, Any], *, batch_size: int = 1, start_index: int = 0, num_games: int | None = None):
    from alfworld.agents.environment import get_environment

    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    # ALFWorld truncates game_files during construction. Collect all games first so
    # sorting and start_index select the same episodes on every filesystem.
    runtime_config = deepcopy(config)
    if start_index or num_games is not None:
        runtime_config["dataset"]["num_eval_games"] = 0
    env_cls = get_environment(runtime_config["env"]["type"])
    env = env_cls(runtime_config, train_eval=runtime_config.get("split", "eval_out_of_distribution"))
    env.game_files = sorted(env.game_files)
    if start_index or num_games is not None:
        end_index = None if num_games is None else start_index + num_games
        env.game_files = env.game_files[start_index:end_index]
        if not env.game_files:
            raise ValueError(f"No ALFWorld games selected for start_index={start_index}, num_games={num_games}")
    env.num_games = len(env.game_files)
    return env.init_env(batch_size=batch_size)
