from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


def tokenize_words(text: str) -> list[str]:
    return [token for token in text.strip().split() if token]


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def max_jaccard_repetition(current_text: str, history_texts: Iterable[str]) -> float:
    current = set(tokenize_words(current_text))
    if not current:
        return 0.0
    best = 0.0
    for text in history_texts:
        prev = set(tokenize_words(text))
        if not prev:
            continue
        inter = len(current & prev)
        union = len(current | prev)
        if union:
            best = max(best, inter / union)
    return best


@dataclass
class TokenWeighting:
    weights: list[float]
    repeated_positions: list[int]


def compute_token_weights(
    current_text: str,
    history_texts: Iterable[str],
    *,
    ngram_size: int = 3,
    min_weight: float = 0.05,
) -> TokenWeighting:
    tokens = tokenize_words(current_text)
    if not tokens:
        return TokenWeighting([], [])

    history_tokens = []
    for text in history_texts:
        history_tokens.extend(tokenize_words(text))

    history_counter = Counter(ngrams(history_tokens, ngram_size))
    current_ngrams = ngrams(tokens, ngram_size)

    repeated_positions: set[int] = set()
    exposures = [0 for _ in tokens]
    for start, gram in enumerate(current_ngrams):
        exposure = history_counter[gram]
        if exposure > 0:
            for pos in range(start, start + ngram_size):
                repeated_positions.add(pos)
                exposures[pos] = max(exposures[pos], exposure)

    weights: list[float] = []
    for pos, _token in enumerate(tokens):
        if pos in repeated_positions:
            weight = max(min_weight, 1.0 / (1.0 + exposures[pos]))
        else:
            weight = 1.0
        weights.append(weight)

    return TokenWeighting(weights=weights, repeated_positions=sorted(repeated_positions))


def compute_adaptive_token_weights(
    current_text: str,
    history_texts: Iterable[str],
    *,
    ngram_size: int = 3,
    min_weight: float = 0.05,
    high_repetition_min_weight: float = 0.01,
    high_repetition_threshold: float = 0.8,
) -> TokenWeighting:
    repetition = max_jaccard_repetition(current_text, history_texts)
    floor = high_repetition_min_weight if repetition >= high_repetition_threshold else min_weight
    return compute_token_weights(
        current_text,
        history_texts,
        ngram_size=ngram_size,
        min_weight=floor,
    )
