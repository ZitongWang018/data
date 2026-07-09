from __future__ import annotations

import argparse

from agentic_ttt.table1 import compare_results, format_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/table1_qwen35_4b.json")
    parser.add_argument("--results")
    args = parser.parse_args()

    print(format_comparison(compare_results(args.config, args.results)))


if __name__ == "__main__":
    main()

