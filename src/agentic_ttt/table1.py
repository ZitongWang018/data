from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResultRow:
    name: str
    paper_target: float
    reproduced: float | None

    @property
    def delta(self) -> float | None:
        if self.reproduced is None:
            return None
        return self.reproduced - self.paper_target


def load_targets(config_path: str | Path) -> dict[str, float]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in config["paper_targets"].items()}


def compare_results(config_path: str | Path, results_path: str | Path | None = None) -> list[ResultRow]:
    targets = load_targets(config_path)
    reproduced: dict[str, float] = {}
    if results_path is not None and Path(results_path).exists():
        raw = json.loads(Path(results_path).read_text(encoding="utf-8"))
        reproduced = {str(k): float(v) for k, v in raw.items()}
    return [
        ResultRow(name=name, paper_target=target, reproduced=reproduced.get(name))
        for name, target in targets.items()
    ]


def format_comparison(rows: list[ResultRow]) -> str:
    lines = ["method,paper_target,reproduced,delta"]
    for row in rows:
        reproduced = "" if row.reproduced is None else f"{row.reproduced:.2f}"
        delta = "" if row.delta is None else f"{row.delta:+.2f}"
        lines.append(f"{row.name},{row.paper_target:.2f},{reproduced},{delta}")
    return "\n".join(lines)

