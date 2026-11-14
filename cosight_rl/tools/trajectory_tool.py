"""Stateful, deterministic Plan-and-Act tool for trajectory GRPO.

The tool exposes a controlled offline research environment.  Evidence from the
same teacher trace/workspace corpus used by report GRPO is hidden behind a
search operation instead of being inserted into the prompt.  This keeps group
rollouts reproducible while requiring the policy to plan, retrieve, maintain a
workspace, respect dependencies, and synthesize a report.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}|\d+(?:\.\d+)?%?", text))
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


_VISUAL_EVIDENCE_RE = re.compile(
    r"ask_question_about_image|image[_ -]?(?:analysis|inspection|description|extraction)|"
    r"visual[_ -]?(?:analysis|inspection|evidence|grounding)|ocr|chart[_ -]?analysis|"
    r"figure[_ -]?(?:analysis|inspection)",
    re.IGNORECASE,
)


def _is_visual_evidence(item: Mapping[str, Any]) -> bool:
    if item.get("evidence_role") == "frozen_visual_observation":
        return True
    label = f"{item.get('title') or ''}\n{item.get('source_ref') or ''}"
    return bool(_VISUAL_EVIDENCE_RE.search(label))


class CoSightTrajectoryTool(BaseTool):
    """One persistent workspace tool instance per end-to-end rollout."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id, response = await super().create(instance_id=instance_id)
        create_kwargs = kwargs.get("create_kwargs") if isinstance(kwargs.get("create_kwargs"), Mapping) else kwargs
        self._instance_dict[instance_id] = {
            "task_id": str(create_kwargs.get("task_id") or ""),
            "rubric": create_kwargs.get("rubric"),
            "local_evidence": [
                dict(item) for item in _as_list(create_kwargs.get("local_evidence")) if isinstance(item, Mapping)
            ],
            "plan": {},
            "completed_steps": [],
            "searches": [],
            "submitted_report": "",
            "last_report_candidate": "",
            "report_submission_mode": "",
            "invalid_dependency_attempts": 0,
            "events": [],
        }
        return instance_id, response

    def snapshot(self, instance_id: str) -> Dict[str, Any]:
        return deepcopy(self._instance_dict.get(instance_id) or {})

    def is_submitted(self, instance_id: str) -> bool:
        return bool((self._instance_dict.get(instance_id) or {}).get("submitted_report"))

    def promote_report_candidate(
        self,
        instance_id: str,
        report: str = "",
        mode: str = "terminal_fallback",
    ) -> bool:
        """Accept a policy-generated terminal report without adding reward.

        Dimension-E2E uses this only after the policy terminates.  It makes the
        terminal report action robust to an early/last-turn submit or a direct
        Markdown final answer while preserving the exact report reward.
        """

        state = self._state(instance_id)
        if not (state.get("plan") or {}).get("steps"):
            return False
        if not state.get("completed_steps"):
            return False
        candidate = str(report or state.get("last_report_candidate") or "").strip()
        if len(candidate) < 200:
            return False
        state["submitted_report"] = candidate
        state["report_submission_mode"] = mode
        state["events"].append({"operation": "promote_report_candidate", "mode": mode})
        return True

    async def cleanup(self, instance_id: str) -> None:
        self._instance_dict.pop(instance_id, None)

    def _state(self, instance_id: str) -> Dict[str, Any]:
        if instance_id not in self._instance_dict:
            raise KeyError(f"Unknown Co-Sight rollout instance: {instance_id}")
        return self._instance_dict[instance_id]

    @staticmethod
    def _normalize_plan(content: Any, parameters: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        parsed = _parse_json_object(content)
        if not parsed and isinstance(parameters, Mapping):
            parsed = {
                "title": parameters.get("title"),
                "steps": parameters.get("steps"),
                "dependencies": parameters.get("dependencies"),
            }
        steps = [str(item).strip() for item in _as_list(parsed.get("steps")) if str(item).strip()]
        raw_dependencies = parsed.get("dependencies") or {}
        dependencies: Dict[int, List[int]] = {index: [] for index in range(len(steps))}
        if isinstance(raw_dependencies, Mapping):
            for key, values in raw_dependencies.items():
                try:
                    step_index = int(key)
                except (TypeError, ValueError):
                    continue
                if step_index < 0 or step_index >= len(steps):
                    continue
                deps: List[int] = []
                for value in _as_list(values):
                    try:
                        dep = int(value)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= dep < len(steps) and dep != step_index and dep not in deps:
                        deps.append(dep)
                dependencies[step_index] = sorted(deps)
        return {"title": str(parsed.get("title") or ""), "steps": steps, "dependencies": dependencies}

    @staticmethod
    def _completed_ids(state: Mapping[str, Any]) -> set[int]:
        result: set[int] = set()
        for item in state.get("completed_steps") or []:
            try:
                result.add(int(item.get("step_id")))
            except (AttributeError, TypeError, ValueError):
                continue
        return result

    def _search(self, state: Dict[str, Any], query: str, top_k: int) -> str:
        query_tokens = _tokens(query)
        ranked = []
        for index, item in enumerate(state.get("local_evidence") or [], 1):
            haystack = f"{item.get('title', '')}\n{item.get('text', '')}\n{' '.join(str(x) for x in _as_list(item.get('urls')))}"
            item_tokens = _tokens(haystack)
            overlap = len(query_tokens & item_tokens) / max(1, len(query_tokens))
            ranked.append((overlap, index, item))
        ranked.sort(key=lambda value: (value[0], len(str(value[2].get("text") or ""))), reverse=True)
        selected = ranked[: max(1, min(int(top_k or 3), 5))]
        state["searches"].append({"query": query, "returned_ids": [index for _, index, _ in selected]})
        blocks = []
        for score, evidence_id, item in selected:
            urls = "; ".join(str(url) for url in _as_list(item.get("urls")) if url)
            text = str(item.get("text") or "")[:1600]
            blocks.append(
                f"[E{evidence_id}] {item.get('title') or 'Local evidence'}\n"
                f"URLs: {urls or '(none)'}\n{text}\nmatch={score:.3f}"
            )
        return "\n\n".join(blocks) if blocks else "No grounded local evidence matched this query."

    def _inspect_visual(self, state: Dict[str, Any], top_k: int) -> str:
        indexed = list(enumerate(state.get("local_evidence") or [], 1))
        selected = [(index, item) for index, item in indexed if _is_visual_evidence(item)]
        selected = selected[: max(1, min(int(top_k or 3), 5))]
        returned_ids = [index for index, _ in selected]
        state["searches"].append(
            {"query": "__frozen_visual_observation__", "returned_ids": returned_ids}
        )
        blocks = []
        for evidence_id, item in selected:
            text = str(item.get("text") or "")[:2400]
            blocks.append(
                f"[E{evidence_id}] Frozen visual observation: "
                f"{item.get('title') or 'image inspection'}\n{text}"
            )
        return (
            "\n\n".join(blocks)
            if blocks
            else "No frozen visual observation is available for this task. State the visual uncertainty explicitly."
        )

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        state = self._state(instance_id)
        operation = str(parameters.get("operation") or "").strip().lower()
        state["events"].append({"operation": operation, "parameters": deepcopy(parameters)})

        if operation == "create_plan":
            plan = self._normalize_plan(parameters.get("content"), parameters)
            if len(plan["steps"]) < 2 or len(plan["steps"]) > 12:
                return ToolResponse(text="ERROR: plan must contain 2-12 non-empty steps."), 0.0, {"valid": False}
            if not re.search(r"report|报告|总结|synth", plan["steps"][-1], re.IGNORECASE):
                return ToolResponse(text="ERROR: the last plan step must synthesize the final report."), 0.0, {"valid": False}
            state["plan"] = plan
            rendered = [f"S{index}: {step} deps={plan['dependencies'].get(index, [])}" for index, step in enumerate(plan["steps"])]
            return ToolResponse(text="PLAN_ACCEPTED\n" + "\n".join(rendered)), 0.0, {"valid": True}

        if operation == "search_evidence":
            query = str(parameters.get("query") or parameters.get("content") or "").strip()
            if not query:
                return ToolResponse(text="ERROR: search_evidence requires a query."), 0.0, {"valid": False}
            return ToolResponse(text=self._search(state, query, int(parameters.get("top_k") or 3))), 0.0, {"valid": True}

        if operation == "inspect_visual":
            text = self._inspect_visual(state, int(parameters.get("top_k") or 3))
            return ToolResponse(text=text), 0.0, {"valid": True}

        if operation == "read_workspace":
            if not state["completed_steps"]:
                text = "Workspace is empty."
            else:
                text = "\n\n".join(
                    f"S{item['step_id']} {item.get('objective', '')}\n{item.get('summary', '')}"
                    for item in state["completed_steps"]
                )
            return ToolResponse(text=text[:6000]), 0.0, {"valid": True}

        if operation == "save_step":
            plan = state.get("plan") or {}
            if not plan.get("steps"):
                return ToolResponse(text="ERROR: call create_plan before save_step."), 0.0, {"valid": False}
            try:
                step_id = int(parameters.get("step_id"))
            except (TypeError, ValueError):
                return ToolResponse(text="ERROR: save_step requires an integer step_id."), 0.0, {"valid": False}
            if step_id < 0 or step_id >= len(plan["steps"]) - 1:
                return ToolResponse(text="ERROR: step_id must refer to a non-final plan step."), 0.0, {"valid": False}
            completed_ids = self._completed_ids(state)
            dependencies = list(plan.get("dependencies", {}).get(step_id) or [])
            dependency_valid = all(dep in completed_ids for dep in dependencies)
            if not dependency_valid:
                state["invalid_dependency_attempts"] += 1
                missing = [dep for dep in dependencies if dep not in completed_ids]
                return ToolResponse(text=f"BLOCKED: complete dependencies {missing} before S{step_id}."), 0.0, {"valid": False}
            if step_id in completed_ids:
                return ToolResponse(text=f"ERROR: S{step_id} was already saved."), 0.0, {"valid": False}
            summary = str(parameters.get("content") or "").strip()
            if len(summary) < 40:
                return ToolResponse(text="ERROR: save_step requires a substantive grounded summary."), 0.0, {"valid": False}
            evidence_ids = [str(item).strip().upper() for item in _as_list(parameters.get("evidence_ids")) if str(item).strip()]
            seen_evidence_ids = {
                f"E{evidence_id}"
                for search in state.get("searches") or []
                for evidence_id in search.get("returned_ids") or []
            }
            invalid_evidence_ids = [item for item in evidence_ids if item not in seen_evidence_ids]
            if invalid_evidence_ids:
                return ToolResponse(
                    text=f"ERROR: evidence ids were not returned by search_evidence: {invalid_evidence_ids}."
                ), 0.0, {"valid": False}
            item = {
                "step_id": step_id,
                "objective": plan["steps"][step_id],
                "summary": summary,
                "evidence_ids": evidence_ids,
                "artifact": str(parameters.get("artifact") or ""),
                "dependency_valid": True,
            }
            state["completed_steps"].append(item)
            return ToolResponse(text=f"STEP_SAVED S{step_id}. Workspace now contains {len(state['completed_steps'])} completed steps."), 0.0, {"valid": True}

        if operation == "submit_report":
            plan_steps = list((state.get("plan") or {}).get("steps") or [])
            if not plan_steps:
                return ToolResponse(text="ERROR: call create_plan before submit_report."), 0.0, {"valid": False}
            report = str(parameters.get("content") or "").strip()
            if len(report) >= 200:
                # Preserve a valid policy-generated report even when the call
                # is rejected only because some bookkeeping steps are missing.
                # The Dimension-E2E loop may promote it at episode termination.
                state["last_report_candidate"] = report
            completed_ids = self._completed_ids(state)
            missing_ids = [step_id for step_id in range(len(plan_steps) - 1) if step_id not in completed_ids]
            if missing_ids:
                return ToolResponse(
                    text=f"BLOCKED: complete non-final steps {missing_ids} before submit_report."
                ), 0.0, {"valid": False, "missing_step_ids": missing_ids}
            if len(report) < 200:
                return ToolResponse(text="ERROR: submit_report requires a complete Markdown report."), 0.0, {"valid": False}
            state["submitted_report"] = report
            state["report_submission_mode"] = "explicit_complete"
            return ToolResponse(text="REPORT_SUBMITTED. incomplete_nonfinal_steps=0"), 0.0, {"valid": True}

        if operation == "status":
            return ToolResponse(
                text=json.dumps(
                    {
                        "has_plan": bool((state.get("plan") or {}).get("steps")),
                        "completed_step_ids": sorted(self._completed_ids(state)),
                        "search_count": len(state.get("searches") or []),
                        "report_submitted": bool(state.get("submitted_report")),
                    },
                    ensure_ascii=False,
                )
            ), 0.0, {"valid": True}

        return ToolResponse(
            text="ERROR: operation must be one of create_plan, inspect_visual, search_evidence, "
            "read_workspace, save_step, status, submit_report."
        ), 0.0, {"valid": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        # The custom agent loop owns the complete rollout lifecycle and calls
        # cleanup only after the step-only reward has consumed the state.
        return None


__all__ = ["CoSightTrajectoryTool"]
