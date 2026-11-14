"""Strict dimension-only reward for the report-GRPO ablation.

Unlike the default Co-Sight reward, this module does not blend the structured
rubric with the five generic heuristic components.  The scalar reward is the
weighted GEN/EVI/VEF/MM/PRES rubric score, optionally capped only by gating
rules declared inside that rubric.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from cosight_rl.rewards.reward_evaluator import evaluate_report


_GATE_REASON_BY_PREFIX = {
    "EVI": "fabricated_source_data",
    "MM": "wrong_visual_identity",
    "PRES": "repetition_collapse",
}


def _rubric_from_inputs(ground_truth: Any, extra_info: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    value: Any = None
    if isinstance(ground_truth, Mapping):
        value = ground_truth.get("rubric")
    if not value and isinstance(extra_info, Mapping):
        value = extra_info.get("rubric")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != {"GEN", "EVI", "VEF", "MM", "PRES"}:
        return None
    return dict(value)


def evaluate_dimension_only_report(
    report_text: str,
    ground_truth: Any,
    extra_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rubric = _rubric_from_inputs(ground_truth, extra_info)
    if rubric is None:
        return {
            "score": 0.0,
            "final_reward": 0.0,
            "raw_reward": 0.0,
            "reward_cap": 0.0,
            "reward_mode": "dimension_only_report",
            "error": "missing_or_invalid_dimension_rubric",
        }

    base = evaluate_report(
        report_text,
        task_prompt=str((ground_truth or {}).get("task_prompt") or "")
        if isinstance(ground_truth, Mapping)
        else "",
        extra_info=extra_info,
        ground_truth=ground_truth,
    )
    raw = float(base.get("R_rubric", 0.0))
    detected = set(base.get("hard_gate_reasons") or [])
    cap = 1.0
    applied_gates = []
    for gate in rubric.get("gating_rules") or []:
        if not isinstance(gate, Mapping):
            continue
        gate_id = str(gate.get("id") or "")
        prefix = gate_id.split("-", 1)[0]
        reason = _GATE_REASON_BY_PREFIX.get(prefix)
        if reason and reason in detected:
            try:
                gate_cap = float(gate.get("overall_cap", 1.0))
            except (TypeError, ValueError):
                gate_cap = 1.0
            cap = min(cap, max(0.0, min(1.0, gate_cap)))
            applied_gates.append({"id": gate_id, "reason": reason, "cap": gate_cap})
    final = max(0.0, min(1.0, min(raw, cap)))
    structured = (base.get("details") or {}).get("structured_rubric") or {}
    return {
        "score": final,
        "final_reward": final,
        "raw_reward": raw,
        "reward_cap": cap,
        "reward_mode": "dimension_only_report",
        "rubric_schema_version": rubric.get("schema_version"),
        "rubric_dimension_scores": base.get("rubric_dimension_scores") or {},
        "rubric_criterion_scores": structured.get("criteria") or {},
        "applied_rubric_gates": applied_gates,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Mapping[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    result = evaluate_dimension_only_report(solution_str, ground_truth, extra_info)
    result["data_source"] = data_source
    return result


__all__ = ["compute_score", "evaluate_dimension_only_report"]
