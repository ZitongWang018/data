from __future__ import annotations

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


def make_alfworld_env(config: dict[str, Any], *, batch_size: int = 1):
    from alfworld.agents.environment import get_environment

    env_cls = get_environment(config["env"]["type"])
    env = env_cls(config, train_eval=config.get("split", "eval_out_of_distribution"))
    return env.init_env(batch_size=batch_size)

