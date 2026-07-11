from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from agentic_ttt.results import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--targets", default="configs/table1_qwen35_4b.json")
    parser.add_argument("--output-json", default="/root/autodl-tmp/logs/qwen35_4b_table1_summary.json")
    parser.add_argument("--output-md", default="/root/autodl-tmp/logs/qwen35_4b_table1_summary.md")
    args = parser.parse_args()

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    for output in (output_json, output_md):
        if not str(output).startswith("/root/autodl-tmp"):
            raise RuntimeError(f"Refusing to write outside /root/autodl-tmp: {output}")

    targets = json.loads(Path(args.targets).read_text(encoding="utf-8")).get("paper_targets", {})
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs if Path(path).exists()]
    summaries = [summarize_payload(payload, targets) for payload in payloads]
    paired = paired_comparisons(payloads)
    summary = {"methods": summaries, "paired": paired}
    atomic_write_json(output_json, summary)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(summaries, paired), encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"), flush=True)


def summarize_payload(payload: dict[str, Any], targets: dict[str, float]) -> dict[str, Any]:
    rows = payload.get("results", [])
    wins = sum(bool(row.get("won")) for row in rows)
    episodes = len(rows)
    rate = 100.0 * wins / episodes if episodes else 0.0
    probability = wins / episodes if episodes else 0.0
    standard_error = 100.0 * math.sqrt(probability * (1.0 - probability) / episodes) if episodes else 0.0
    total_steps = sum(int(row.get("steps", 0)) for row in rows)
    invalid_actions = sum(int(row.get("invalid_actions", 0)) for row in rows)
    method = payload.get("method", "unknown")
    target = targets.get(method)
    return {
        "method": method,
        "complete": bool(payload.get("complete", False)),
        "wins": wins,
        "episodes": episodes,
        "success_rate": rate,
        "standard_error": standard_error,
        "paper_target": target,
        "target_delta": None if target is None else rate - target,
        "invalid_action_rate": 100.0 * invalid_actions / total_steps if total_steps else 0.0,
        "mean_steps": total_steps / episodes if episodes else 0.0,
        "task_types": summarize_task_types(rows),
    }


def summarize_task_types(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task_type = Path(row.get("gamefile", "unknown")).parent.parent.name.split("-")[0]
        groups.setdefault(task_type, []).append(row)
    return {
        task_type: {
            "wins": sum(bool(row.get("won")) for row in task_rows),
            "episodes": len(task_rows),
            "success_rate": 100.0 * sum(bool(row.get("won")) for row in task_rows) / len(task_rows),
        }
        for task_type, task_rows in sorted(groups.items())
    }


def paired_comparisons(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not payloads:
        return []
    baseline = next((payload for payload in payloads if payload.get("method") == "react"), payloads[0])
    baseline_rows = {row.get("gamefile"): bool(row.get("won")) for row in baseline.get("results", [])}
    comparisons = []
    for payload in payloads:
        if payload is baseline:
            continue
        method_rows = {row.get("gamefile"): bool(row.get("won")) for row in payload.get("results", [])}
        common = sorted(set(baseline_rows) & set(method_rows))
        rescues = sum(not baseline_rows[key] and method_rows[key] for key in common)
        regressions = sum(baseline_rows[key] and not method_rows[key] for key in common)
        comparisons.append(
            {
                "reference": baseline.get("method", "unknown"),
                "method": payload.get("method", "unknown"),
                "paired_episodes": len(common),
                "rescues": rescues,
                "regressions": regressions,
                "net_paired_delta": 100.0 * (rescues - regressions) / len(common) if common else 0.0,
            }
        )
    return comparisons


def render_markdown(summaries: list[dict[str, Any]], paired: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen3.5-4B ALFWorld Results",
        "",
        "| Method | Wins / N | Success (%) | SE | Paper target | Delta | Invalid actions (%) | Complete |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        target = "-" if row["paper_target"] is None else f"{row['paper_target']:.1f}"
        delta = "-" if row["target_delta"] is None else f"{row['target_delta']:+.1f}"
        lines.append(
            f"| {row['method']} | {row['wins']} / {row['episodes']} | {row['success_rate']:.2f} | "
            f"{row['standard_error']:.2f} | {target} | {delta} | {row['invalid_action_rate']:.2f} | "
            f"{'yes' if row['complete'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Paired Against ReAct",
            "",
            "| Method | Paired N | Rescues | Regressions | Net paired delta (pp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['method']} | {row['paired_episodes']} | {row['rescues']} | "
            f"{row['regressions']} | {row['net_paired_delta']:+.2f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
