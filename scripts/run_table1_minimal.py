from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/table1_qwen35_4b.json")
    parser.add_argument("--method", choices=["react", "self_attt", "self_adaptive"], default="react")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="/root/autodl-tmp/logs/table1_minimal_result.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = {
        "status": "stub",
        "note": "This runner reserves the table-one CLI contract; ALFWorld rollout integration is next.",
        "model": config["model"],
        "benchmark": config["benchmark"],
        "method": args.method,
        "episodes": args.episodes,
    }
    output = Path(args.output)
    if not str(output).startswith("/root/autodl-tmp"):
        raise RuntimeError(f"Refusing to write outside /root/autodl-tmp: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

