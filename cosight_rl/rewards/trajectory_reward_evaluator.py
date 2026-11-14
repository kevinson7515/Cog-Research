"""Reward routing for Co-Sight end-to-end trajectory GRPO.

Both trajectory ablations use the same Plan-and-Act rollout environment. The
rubric schema is the only switch:

* ``cosight.multidim_rubric.v2`` scores the submitted report with the same
  decomposed, rubric-calibrated reward as the current report-only method;
* ``cosight.step_rubric.v1`` scores plan alignment and dependency-valid step
  completion without report-dimension rewards.

The VERL fallback intentionally returns zero. A trajectory reward is valid only
when the dedicated agent loop can read trusted tool state and the report
submitted through ``cosight_step``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from cosight_rl.rewards.reward_evaluator import evaluate_report
from cosight_rl.rewards.step_reward_evaluator import STEP_SCHEMA_VERSION, parse_step_rubric, score_step_trajectory


DIMENSION_SCHEMA_VERSION = "cosight.multidim_rubric.v2"


def _rubric_from_inputs(
    ground_truth: Any,
    extra_info: Optional[Mapping[str, Any]],
) -> Any:
    if isinstance(ground_truth, Mapping) and ground_truth.get("rubric"):
        return ground_truth.get("rubric")
    if isinstance(extra_info, Mapping):
        return extra_info.get("rubric")
    return None


def rubric_schema(value: Any) -> str:
    candidate = value
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError:
            return ""
    if not isinstance(candidate, Mapping):
        return ""
    return str(candidate.get("schema_version") or "")


def _empty_reward(mode: str, error: str) -> Dict[str, Any]:
    return {
        "score": 0.0,
        "final_reward": 0.0,
        "raw_reward": 0.0,
        "reward_cap": 0.0,
        "reward_mode": mode,
        "submitted_report": False,
        "error": error,
    }


def score_dimension_trajectory(
    ground_truth: Any,
    trajectory_state: Optional[Mapping[str, Any]] = None,
    extra_info: Optional[Mapping[str, Any]] = None,
    *,
    enable_source_provenance: Optional[bool] = None,
) -> Dict[str, Any]:
    """Score only the final report submitted by a full Plan-and-Act rollout.

    This calls the normal Co-Sight report reward unchanged. Consequently,
    Condition B and the report-only Condition A share the exact same reward;
    planning and tool-use tokens receive the terminal report reward through
    GRPO, but do not receive an additional process bonus.
    """

    rubric = _rubric_from_inputs(ground_truth, extra_info)
    if rubric_schema(rubric) != DIMENSION_SCHEMA_VERSION:
        return _empty_reward("dimension_trajectory", "missing_or_invalid_dimension_rubric")

    state = dict(trajectory_state or {})
    report = str(state.get("submitted_report") or "").strip()
    if not report:
        return _empty_reward("dimension_trajectory", "missing_submitted_report")

    task_prompt = ""
    if isinstance(ground_truth, Mapping):
        task_prompt = str(ground_truth.get("task_prompt") or "")
    scoring_extra = dict(extra_info or {})
    if "tool_errors" in state:
        # Trusted per-rollout state takes precedence over static row metadata.
        scoring_extra["tool_errors"] = state["tool_errors"]
        scoring_extra.pop("invalid_tool_calls", None)
    result = evaluate_report(
        report,
        task_prompt=task_prompt,
        extra_info=scoring_extra,
        ground_truth=ground_truth,
        enable_source_provenance=enable_source_provenance,
    )
    result.update(
        {
            "score": float(result["final_reward"]),
            "reward_mode": "dimension_trajectory",
            "submitted_report": True,
            "trajectory_diagnostics": {
                "plan_steps": len((state.get("plan") or {}).get("steps") or []),
                "completed_steps": len(state.get("completed_steps") or []),
                "searches": len(state.get("searches") or []),
                "invalid_dependency_attempts": int(state.get("invalid_dependency_attempts") or 0),
            },
        }
    )
    return result


def score_trajectory(
    ground_truth: Any,
    trajectory_state: Optional[Mapping[str, Any]] = None,
    extra_info: Optional[Mapping[str, Any]] = None,
    *,
    enable_source_provenance: Optional[bool] = None,
) -> Dict[str, Any]:
    rubric = _rubric_from_inputs(ground_truth, extra_info)
    schema = rubric_schema(rubric)
    if isinstance(ground_truth, Mapping) and "step_rubric" in ground_truth:
        step_rubric = ground_truth["step_rubric"]
        if schema != DIMENSION_SCHEMA_VERSION or parse_step_rubric(step_rubric) is None:
            raise ValueError("Joint trajectory reward requires valid report and step rubrics")
        report = score_dimension_trajectory(
            ground_truth, trajectory_state, extra_info,
            enable_source_provenance=enable_source_provenance,
        )
        step = score_step_trajectory(step_rubric, trajectory_state)
        joint = 0.5 * float(report["score"]) + 0.5 * float(step["score"])
        return {"score": joint, "final_reward": joint, "reward_mode": "joint_trajectory",
                "report_reward": report, "step_reward": step, "submitted_report": report["submitted_report"]}
    if schema == DIMENSION_SCHEMA_VERSION:
        return score_dimension_trajectory(
            ground_truth, trajectory_state, extra_info, enable_source_provenance=enable_source_provenance,
        )
    if schema == STEP_SCHEMA_VERSION:
        return score_step_trajectory(
            rubric,
            trajectory_state=trajectory_state,
            final_report=str((trajectory_state or {}).get("submitted_report") or ""),
        )
    return _empty_reward("unknown_trajectory", f"unsupported_rubric_schema:{schema or 'missing'}")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Mapping[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Fail closed if VERL bypasses the Co-Sight Plan-and-Act agent loop."""

    rubric = _rubric_from_inputs(ground_truth, extra_info)
    schema = rubric_schema(rubric)
    mode = "step_trajectory" if schema == STEP_SCHEMA_VERSION else "dimension_trajectory"
    result = _empty_reward(mode, "trajectory_reward_requires_cosight_plan_act_agent_loop")
    result["data_source"] = data_source
    return result


__all__ = [
    "DIMENSION_SCHEMA_VERSION",
    "compute_score",
    "rubric_schema",
    "score_dimension_trajectory",
    "score_trajectory",
]
