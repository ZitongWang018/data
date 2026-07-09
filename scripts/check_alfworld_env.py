from __future__ import annotations

import argparse

from agentic_ttt.alfworld_env import build_alfworld_config, make_alfworld_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/data_cache/alfworld")
    parser.add_argument("--split", default="eval_out_of_distribution")
    args = parser.parse_args()

    config = build_alfworld_config(data_root=args.data_root, split=args.split, num_eval_games=1)
    env = make_alfworld_env(config, batch_size=1)
    obs, infos = env.reset()
    print("obs:", obs[0].splitlines()[0][:300])
    print("admissible_count:", len(infos["admissible_commands"][0]))
    print("first_actions:", infos["admissible_commands"][0][:10])
    env.close()


if __name__ == "__main__":
    main()

