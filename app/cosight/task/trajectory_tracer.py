

import copy
import json
import os
import time
from threading import Lock
from typing import Any, Dict, List, Optional

from app.common.logger_util import logger


_tracers: Dict[str, "TrajectoryTracer"] = {}
_registry_lock = Lock()


def create_trajectory_tracer(plan_id: str, trace_dir: str) -> "TrajectoryTracer":
    tracer = TrajectoryTracer(plan_id=plan_id, trace_dir=trace_dir)
    with _registry_lock:
        _tracers[plan_id] = tracer
    return tracer


def get_trajectory_tracer(plan_id: Optional[str]) -> Optional["TrajectoryTracer"]:
    if not plan_id:
        return None
    with _registry_lock:
        return _tracers.get(plan_id)


def remove_trajectory_tracer(plan_id: Optional[str]) -> None:
    if not plan_id:
        return
    with _registry_lock:
        _tracers.pop(plan_id, None)


class TrajectoryTracer:
    """Collect planner/actor conversations and dump them as jsonl records."""

    def __init__(self, plan_id: str, trace_dir: str):
        self.plan_id = plan_id
        self.trace_dir = trace_dir
        self._lock = Lock()
        self._messages = {
            "planner": [],
            "actor": [],
        }
        self._planner_complete = False
        self._dumped = False
        os.makedirs(self.trace_dir, exist_ok=True)

    @property
    def planner_trace_path(self) -> str:
        return os.path.join(self.trace_dir, "planner_trace.jsonl")

    @property
    def actor_trace_path(self) -> str:
        return os.path.join(self.trace_dir, "actor_trace.jsonl")

    def record_messages(self, trace_type: str, messages: List[Dict[str, Any]], step_index: Optional[int] = None) -> None:
        if trace_type not in self._messages:
            return
        normalized = []
        for message in messages:
            if trace_type == "planner" and self._planner_complete:
                break
            normalized_message = self._normalize_message(trace_type, message, step_index)
            normalized.append(normalized_message)
            if trace_type == "planner" and self._is_plan_created_message(normalized_message):
                self._planner_complete = True
                break
        with self._lock:
            self._messages[trace_type].extend(normalized)

    def dump(self) -> None:
        with self._lock:
            if self._dumped:
                return
            self._dumped = True
            planner_messages = copy.deepcopy(self._messages["planner"])
            actor_messages = copy.deepcopy(self._messages["actor"])

        if planner_messages:
            self._append_jsonl_record(self.planner_trace_path, {
                "plan_id": self.plan_id,
                "trace_type": "planner",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "messages": planner_messages,
            })

        for step_index, step_messages in self._actor_messages_by_step(actor_messages).items():
            if step_messages:
                self._append_jsonl_record(self.actor_trace_path, {
                    "plan_id": self.plan_id,
                    "trace_type": "actor",
                    "step": step_index,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": step_messages,
                })

    def _append_jsonl_record(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        logger.info(f"Wrote trajectory trace: {path}")

    def _normalize_message(self, trace_type: str, message: Dict[str, Any], step_index: Optional[int]) -> Dict[str, Any]:
        msg = self._json_safe(message)
        normalized_role = msg.get("role", "")

        normalized = {
            "role": normalized_role,
            "content": msg.get("content", ""),
        }
        if trace_type == "actor" and step_index is not None:
            normalized["step"] = step_index

        if normalized_role == "assistant":
            for key in ("tool_calls", "reasoning_content"):
                if key in msg:
                    normalized[key] = msg[key]
        elif normalized_role == "tool":
            for key in ("name", "tool_call_id"):
                if key in msg:
                    normalized[key] = msg[key]
        return normalized

    def _actor_messages_by_step(self, messages: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for message in messages:
            step_index = message.get("step")
            if step_index is None:
                continue
            message_copy = dict(message)
            message_copy.pop("step", None)
            grouped.setdefault(step_index, []).append(message_copy)
        return dict(sorted(grouped.items(), key=lambda item: item[0]))

    def _is_plan_created_message(self, message: Dict[str, Any]) -> bool:
        return (
            message.get("role") == "user"
            and "Plan created successfully" in str(message.get("content", ""))
        )

    def _json_safe(self, value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False, default=str)
            return copy.deepcopy(value)
        except Exception:
            return str(value)
