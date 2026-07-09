from __future__ import annotations

import argparse

from agentic_ttt.smoke import format_smoke_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "attt"], default="attt")
    args = parser.parse_args()

    if args.mode == "baseline":
        print("baseline=no-ttt")
        print(format_smoke_report())
    else:
        print("method=attt")
        print(format_smoke_report())


if __name__ == "__main__":
    main()

