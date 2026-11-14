"""Step-only reward used by the end-to-end trajectory GRPO ablation.

This evaluator intentionally ignores GEN/EVI/VEF/MM/PRES report dimensions.
It scores whether a policy-created Plan-and-Act trajectory completed the
teacher-defined steps, respected dependencies, produced grounded step
summaries, and submitted a final report. The shared trajectory reward router
supplies the trusted tool state.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional


STEP_SCHEMA_VERSION = "cosight.step_rubric.v1"
_EN_STOPWORDS = {
    "about", "after", "also", "and", "are", "from", "have", "into", "must",
    "that", "the", "their", "then", "this", "using", "with", "生成", "完成",
    "分析", "报告", "步骤", "最终", "进行", "基于", "包括", "相关",
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if isinstance(value, float) and math.isnan(value):
        return low
    return max(low, min(high, float(value)))


def parse_step_rubric(value: Any) -> Optional[Dict[str, Any]]:
    candidate = value
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    if not isinstance(candidate, Mapping):
        return None
    if candidate.get("schema_version") != STEP_SCHEMA_VERSION:
        return None
    steps = candidate.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    return dict(candidate)


def _tokens(text: Any) -> set[str]:
    lowered = str(text or "").lower()
    tokens = {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{2,}|\d+(?:\.\d+)?%?", lowered)
        if token not in _EN_STOPWORDS
    }
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
        for index in range(len(sequence) - 1):
            bigram = sequence[index : index + 2]
            if bigram not in _EN_STOPWORDS:
                tokens.add(bigram)
    return tokens


def criterion_coverage(actual: Any, expected: Any) -> float:
    expected_tokens = _tokens(expected)
    if not expected_tokens:
        return 0.0
    actual_tokens = _tokens(actual)
    overlap = len(expected_tokens & actual_tokens) / len(expected_tokens)
    # Teacher mark_step notes are often much more verbose than a valid
    # alternative result.  Reaching 45% coverage is therefore treated as full.
    score = clamp(overlap / 0.45)
    expected_numbers = {token for token in expected_tokens if re.fullmatch(r"\d+(?:\.\d+)?%?", token)}
    if expected_numbers:
        number_coverage = len(expected_numbers & actual_tokens) / len(expected_numbers)
        score *= 0.65 + 0.35 * number_coverage
    return clamp(score)


def _greedy_plan_alignment(generated_steps: List[str], teacher_steps: List[Mapping[str, Any]]) -> float:
    if not generated_steps or not teacher_steps:
        return 0.0
    unused = set(range(len(generated_steps)))
    scores: List[float] = []
    for teacher in teacher_steps:
        objective = teacher.get("objective") or ""
        candidates = [(criterion_coverage(generated_steps[index], objective), index) for index in unused]
        if not candidates:
            scores.append(0.0)
            continue
        score, matched = max(candidates)
        unused.remove(matched)
        scores.append(score)
    count_ratio = min(len(generated_steps), len(teacher_steps)) / max(len(generated_steps), len(teacher_steps))
    return clamp((sum(scores) / len(scores)) * (0.70 + 0.30 * count_ratio))


def _normalize_saved_steps(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = state.get("completed_steps") or []
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    result: List[Dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, Mapping):
            result.append(dict(item))
    return result


def _teacher_step_score(teacher: Mapping[str, Any], completed: List[Mapping[str, Any]]) -> Dict[str, Any]:
    best: Dict[str, Any] = {"score": 0.0, "matched_step_id": None}
    criteria = [item for item in teacher.get("criteria") or [] if isinstance(item, Mapping)]
    teacher_tools = set(str(item) for item in teacher.get("teacher_tool_names") or [])
    expects_external_evidence = bool(
        teacher_tools
        & {
            "serper_search", "image_search", "fetch_website_content",
            "fetch_website_content_with_images", "extract_document_content",
        }
    )
    expects_artifact = bool(teacher.get("expected_artifacts"))

    for item in completed:
        text = f"{item.get('objective', '')}\n{item.get('summary', '')}"
        criterion_scores = [criterion_coverage(text, criterion.get("criterion")) for criterion in criteria]
        semantic = sum(criterion_scores) / max(1, len(criterion_scores))
        evidence_ids = item.get("evidence_ids") or []
        evidence_score = 1.0 if evidence_ids or not expects_external_evidence else 0.0
        artifact_score = 1.0 if item.get("artifact") or not expects_artifact else 0.0
        dependency_score = 1.0 if item.get("dependency_valid", False) else 0.0
        score = clamp(0.70 * semantic + 0.12 * evidence_score + 0.08 * artifact_score + 0.10 * dependency_score)
        if score > best["score"]:
            best = {
                "score": score,
                "matched_step_id": item.get("step_id"),
                "semantic": semantic,
                "evidence": evidence_score,
                "artifact": artifact_score,
                "dependency": dependency_score,
            }
    return best


def _match_teacher_steps(
    teacher_steps: List[Mapping[str, Any]],
    completed: List[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Greedily enforce one-to-one teacher-step/workspace-step matching.

    Without unique matching, one generic saved summary can receive credit for
    several teacher steps.  The pair scores are computed first and the best
    remaining non-conflicting pair is selected at each iteration.
    """

    pair_scores: List[tuple[float, int, int, Dict[str, Any]]] = []
    for teacher_index, teacher in enumerate(teacher_steps):
        for completed_index, item in enumerate(completed):
            detail = _teacher_step_score(teacher, [item])
            pair_scores.append((float(detail["score"]), teacher_index, completed_index, detail))
    pair_scores.sort(key=lambda value: value[0], reverse=True)

    matched_teachers: set[int] = set()
    matched_completed: set[int] = set()
    result: Dict[str, Dict[str, Any]] = {
        str(teacher.get("id")): {"score": 0.0, "matched_step_id": None}
        for teacher in teacher_steps
    }
    for score, teacher_index, completed_index, detail in pair_scores:
        if score <= 0 or teacher_index in matched_teachers or completed_index in matched_completed:
            continue
        teacher = teacher_steps[teacher_index]
        result[str(teacher.get("id"))] = detail
        matched_teachers.add(teacher_index)
        matched_completed.add(completed_index)
    return result


def score_step_trajectory(
    step_rubric: Any,
    trajectory_state: Optional[Mapping[str, Any]] = None,
    final_report: str = "",
) -> Dict[str, Any]:
    rubric = parse_step_rubric(step_rubric)
    if rubric is None:
        return {
            "score": 0.0,
            "final_reward": 0.0,
            "error": "missing_or_invalid_step_rubric",
            "reward_mode": "step_only_trajectory",
        }

    state = dict(trajectory_state or {})
    teacher_steps = [dict(item) for item in rubric["steps"] if isinstance(item, Mapping)]
    generated_plan = state.get("plan") or {}
    generated_steps = [str(item) for item in generated_plan.get("steps") or []]
    plan_alignment = _greedy_plan_alignment(generated_steps, teacher_steps)
    completed = _normalize_saved_steps(state)

    non_final = [item for item in teacher_steps if not item.get("is_final_report")]
    step_breakdown = _match_teacher_steps(non_final, completed)
    step_scores = [float(step_breakdown[str(teacher.get("id"))]["score"]) for teacher in non_final]

    submitted = bool(state.get("submitted_report"))
    report = str(state.get("submitted_report") or final_report or "")
    # Submission is the final teacher step. Content quality is deliberately
    # not scored here; GEN/EVI/VEF/MM/PRES belong only to Dimension-only RL.
    report_score = 1.0 if submitted and report.strip() else 0.0
    step_scores.append(report_score)

    mean_step_score = sum(step_scores) / max(1, len(step_scores))
    completion_ratio = min(1.0, len(completed) / max(1, len(non_final)))
    invalid_dependency_attempts = int(state.get("invalid_dependency_attempts") or 0)
    invalid_penalty = min(0.20, invalid_dependency_attempts * 0.04)
    raw = clamp(0.15 * plan_alignment + 0.75 * mean_step_score + 0.10 * completion_ratio - invalid_penalty)

    cap = 1.0
    cap_reasons: List[str] = []
    if not generated_steps:
        cap, cap_reasons = min(cap, 0.40), [*cap_reasons, "missing_plan"]
    if not submitted:
        cap, cap_reasons = min(cap, 0.60), [*cap_reasons, "missing_submitted_report"]
    if len(completed) < max(1, len(non_final) // 2):
        cap, cap_reasons = min(cap, 0.55), [*cap_reasons, "most_steps_incomplete"]
    final = clamp(min(raw, cap))
    return {
        "score": final,
        "final_reward": final,
        "raw_reward": raw,
        "reward_cap": cap,
        "cap_reasons": cap_reasons,
        "reward_mode": "step_only_trajectory",
        "plan_alignment": round(plan_alignment, 6),
        "mean_step_score": round(mean_step_score, 6),
        "completion_ratio": round(completion_ratio, 6),
        "submitted_report": submitted,
        "invalid_dependency_attempts": invalid_dependency_attempts,
        "step_scores": step_breakdown,
        "final_report_step_score": round(report_score, 6),
    }


__all__ = [
    "STEP_SCHEMA_VERSION",
    "criterion_coverage",
    "parse_step_rubric",
    "score_step_trajectory",
]
