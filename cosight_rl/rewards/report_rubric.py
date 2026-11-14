"""Deterministic report-criterion coverage for Section 4.4.2 / Appendix A.5.

Coverage and citation-local co-occurrence are structural proxies, not a factual
entailment judge. Rubric weights are used only for the diagnostic R_rubric.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Mapping, Optional


DIMENSIONS = ("GEN", "EVI", "VEF", "MM", "PRES")


def parse_structured_rubric(value: Any) -> Optional[Dict[str, Any]]:
    """Reject an incomplete five-dimension rubric as a whole.

    Legacy free text deliberately falls back to the base report heuristics.
    Prepared training rows require two criteria per dimension; the scorer also
    handles old/partial rows with empty criterion lists (their C_d is zero).
    """

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if not isinstance(value, Mapping):
        return None
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, Mapping) or any(
        not isinstance(dimensions.get(key), Mapping) for key in DIMENSIONS
    ):
        return None
    result = dict(value)
    result["dimensions"] = {}
    for key in DIMENSIONS:
        dimension = dict(dimensions[key])
        criteria = dimension.get("criteria")
        normalized = []
        for index, item in enumerate(criteria if isinstance(criteria, list) else []):
            if not isinstance(item, Mapping):
                continue
            criterion = dict(item)
            criterion["id"] = str(criterion.get("id") or f"{key}-{index + 1}")
            criterion["evaluation"] = str(criterion.get("evaluation") or "content")
            normalized.append(criterion)
        dimension["criteria"] = normalized
        result["dimensions"][key] = dimension
    return result


def _tokens(text: str) -> set[str]:
    lower = text.lower()
    tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}|\d+(?:\.\d+)?%?", lower))
    for run in re.findall(r"[\u4e00-\u9fff]+", lower):
        tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def criterion_coverage(report: str, criterion: str) -> float:
    """Token coverage with the paper's 0.55 threshold and numerical factor."""

    required = _tokens(criterion)
    if not required:
        return 0.0
    observed = _tokens(report)
    coverage = min(1.0, len(required & observed) / (0.55 * len(required)))
    numbers = {token for token in required if re.fullmatch(r"\d+(?:\.\d+)?%?", token)}
    if numbers:
        coverage *= 0.55 + 0.45 * len(numbers & observed) / len(numbers)
    return coverage


def _resolved_cited_sentences(text: str, resolved_ids: set[str]) -> str:
    # Reference entries are mappings, not cited claims. Keep punctuation-adjacent
    # citations with the preceding sentence ("claim.[1]" and "claim [1].").
    body = "\n".join(
        line for line in text.splitlines()
        if not re.match(r"^\s*(?:[-*+]\s*)?\[\d+\]\s*(?::|\s).*https?://", line)
    )
    sentences = re.split(r"(?<=[.!?。！？])\s*(?!\[\d+\])|\n+", body)
    return "\n".join(
        sentence for sentence in sentences
        if resolved_ids.intersection(re.findall(r"\[(\d+)\]", sentence))
    )


def score_structured_rubric(
    text: str, rubric: Mapping[str, Any], citation_details: Mapping[str, Any]
) -> Dict[str, Any]:
    cited = _resolved_cited_sentences(text, set(citation_details.get("resolved_citation_ids") or []))
    dimensions: Dict[str, float] = {}
    criteria: Dict[str, list[Dict[str, Any]]] = {}
    weights: Dict[str, float] = {}
    scoring = rubric.get("scoring")
    declared_weights = scoring.get("dimension_weights") if isinstance(scoring, Mapping) else {}
    for key in DIMENSIONS:
        dimension = rubric["dimensions"][key]
        entries = []
        for item in dimension["criteria"]:
            statement = str(item.get("criterion") or "")
            score = criterion_coverage(text, statement)
            if item["evaluation"] == "claim_with_citation":
                score = 0.20 * score + 0.80 * criterion_coverage(cited, statement)
            entries.append({"id": item["id"], "evaluation": item["evaluation"], "score": score})
        criteria[key] = entries
        dimensions[key] = sum(item["score"] for item in entries) / len(entries) if entries else 0.0
        fallback = declared_weights.get(key, 1.0) if isinstance(declared_weights, Mapping) else 1.0
        try:
            weight = float(dimension.get("weight", fallback))
        except (TypeError, ValueError):
            weight = 1.0
        weights[key] = max(0.0, weight) if math.isfinite(weight) else 0.0
    total_weight = sum(weights.values())
    aggregate = sum(weights[key] * dimensions[key] for key in DIMENSIONS) / total_weight if total_weight else 0.0
    return {"dimension_scores": dimensions, "criteria": criteria, "R_rubric": aggregate}
