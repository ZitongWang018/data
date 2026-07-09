from __future__ import annotations

from agentic_ttt.alfworld_env import build_alfworld_config, make_alfworld_env


def main() -> None:
    config = build_alfworld_config(num_eval_games=1, max_steps=10)
    env = make_alfworld_env(config, batch_size=1)
    obs, infos = env.reset()
    print("obs_prefix:", obs[0][:300].replace("\n", " | "))
    print("info_keys:", sorted(infos.keys()))
    print("gamefile:", infos["extra.gamefile"][0])
    print("admissible_count:", len(infos["admissible_commands"][0]))
    print("admissible_sample:", infos["admissible_commands"][0][:30])
    obs, scores, dones, infos = env.step(["look"])
    print("after_look:", {"score": scores[0], "done": bool(dones[0]), "won": bool(infos["won"][0])})
    env.close()


if __name__ == "__main__":
    main()
