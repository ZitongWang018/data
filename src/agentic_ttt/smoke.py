from __future__ import annotations

from dataclasses import dataclass

from .repetition import compute_adaptive_token_weights, compute_token_weights, max_jaccard_repetition


@dataclass
class EpisodeUpdate:
    step: int
    text: str
    jaccard: float
    repeated_positions: list[int]
    weights: list[float]
    adaptive_weights: list[float]


def run_smoke_episode() -> list[EpisodeUpdate]:
    history: list[str] = []
    updates: list[EpisodeUpdate] = []
    texts = [
        "look around kitchen and open cabinet",
        "look around kitchen and open cabinet",
        "take apple from cabinet and examine apple",
        "take apple from cabinet and examine apple",
    ]
    for step, text in enumerate(texts, start=1):
        rep = max_jaccard_repetition(text, history)
        weighting = compute_token_weights(text, history, ngram_size=3, min_weight=0.05)
        adaptive = compute_adaptive_token_weights(
            text,
            history,
            ngram_size=3,
            min_weight=0.05,
            high_repetition_min_weight=0.01,
            high_repetition_threshold=0.8,
        )
        updates.append(
            EpisodeUpdate(
                step=step,
                text=text,
                jaccard=rep,
                repeated_positions=weighting.repeated_positions,
                weights=weighting.weights,
                adaptive_weights=adaptive.weights,
            )
        )
        history.append(text)
    return updates


def format_smoke_report() -> str:
    lines = []
    for item in run_smoke_episode():
        lines.append(
            f"step={item.step} jaccard={item.jaccard:.2f} repeated={item.repeated_positions} "
            f"weights={item.weights} adaptive_weights={item.adaptive_weights}"
        )
    return "\n".join(lines)
