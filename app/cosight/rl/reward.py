"""Compatibility wrapper for the canonical Co-Sight RL reward evaluator.

Older scripts import ``app.cosight.rl.reward``. The maintained implementation
now lives in ``cosight_rl.rewards.reward_evaluator`` so audits, PPO, and legacy
helpers all score with the same logic.
"""

from __future__ import annotations

from cosight_rl.rewards.reward_evaluator import (  # noqa: F401
    FAILURE_CAPS,
    REWARD_WEIGHTS,
    RewardResult,
    broken_url_like_count,
    classify_benchmark_reason,
    compute_score,
    evaluate_report,
    extract_urls,
    score_solution,
)
from cosight_rl.rewards.repetition import (  # noqa: F401
    compute_repetition_metrics,
    compute_soft_repetition_penalty,
    normalize_text_for_repetition,
    split_paragraphs,
    split_sentences,
)

__all__ = [
    "FAILURE_CAPS",
    "REWARD_WEIGHTS",
    "RewardResult",
    "broken_url_like_count",
    "classify_benchmark_reason",
    "compute_repetition_metrics",
    "compute_score",
    "compute_soft_repetition_penalty",
    "evaluate_report",
    "extract_urls",
    "normalize_text_for_repetition",
    "score_solution",
    "split_paragraphs",
    "split_sentences",
]
