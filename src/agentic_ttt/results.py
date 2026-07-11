from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_completed_results(
    path: Path,
    *,
    resume: bool,
    expected: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not resume or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in (expected or {}).items():
        if payload.get(key) != value:
            raise ValueError(f"Cannot resume {path}: {key}={payload.get(key)!r}, expected {value!r}")
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Invalid results list in {path}")
    return results


def runtime_metadata() -> dict[str, str]:
    packages = ["torch", "transformers", "peft", "alfworld", "flash-linear-attention", "causal-conv1d"]
    metadata = {}
    for package in packages:
        try:
            metadata[package] = version(package)
        except PackageNotFoundError:
            metadata[package] = "not-installed"
    return metadata
