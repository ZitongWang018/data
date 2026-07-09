from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/table1_qwen35_4b.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(f"model={config['model']}")
    print(f"benchmark={config['benchmark']}")
    for name, value in config["paper_targets"].items():
        print(f"{name}: {value:.1f}")


if __name__ == "__main__":
    main()

