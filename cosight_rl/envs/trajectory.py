"""Unified trajectory schema for Co-Sight multi-role rollouts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Transition:
    """One flattened role/action/tool transition for JADE-style buffers."""

    task_id: str
    role: str
    observation: str
    action: str
    action_text: str
    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None
    tool_result: Optional[Any] = None
    workspace_delta: List[str] = field(default_factory=list)
    response_mask: List[int] = field(default_factory=list)
    action_logprob: Optional[float] = None
    local_reward: float = 0.0
    final_reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    """A full Co-Sight task trajectory with flattened transitions."""

    task_id: str
    transitions: List[Transition] = field(default_factory=list)
    final_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "final_reward": self.final_reward,
            "metadata": self.metadata,
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except Exception:
        return str(value)


def _tool_calls_from_message(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = message.get("tool_calls") or []
    normalized: List[Dict[str, Any]] = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            function = {}
        args = function.get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except Exception:
            pass
        normalized.append(
            {
                "id": call.get("id") if isinstance(call, dict) else None,
                "name": function.get("name") or call.get("name"),
                "arguments": args,
            }
        )
    return normalized


def flatten_trace_messages(task_id: str, role: str, messages: List[Dict[str, Any]], final_reward: float = 0.0) -> List[Transition]:
    """Flatten OpenAI-style messages into policy/tool transitions."""

    transitions: List[Transition] = []
    last_observation = ""
    pending_by_id: Dict[str, Transition] = {}

    for message in messages:
        msg_role = message.get("role", "")
        content = str(message.get("content") or "")
        if msg_role in {"system", "user", "tool"}:
            last_observation = content

        if msg_role == "assistant":
            tool_calls = _tool_calls_from_message(message)
            if tool_calls:
                for call in tool_calls:
                    transition = Transition(
                        task_id=task_id,
                        role=role,
                        observation=last_observation,
                        action="tool_call",
                        action_text=content,
                        tool_name=call.get("name"),
                        tool_args=_safe_json(call.get("arguments")),
                        final_reward=final_reward,
                        metadata={"tool_call_id": call.get("id")},
                    )
                    transitions.append(transition)
                    if call.get("id"):
                        pending_by_id[str(call["id"])] = transition
            else:
                transitions.append(
                    Transition(
                        task_id=task_id,
                        role=role,
                        observation=last_observation,
                        action="respond",
                        action_text=content,
                        final_reward=final_reward,
                    )
                )
            last_observation = content
        elif msg_role == "tool":
            tool_id = str(message.get("tool_call_id") or "")
            if tool_id and tool_id in pending_by_id:
                pending_by_id[tool_id].tool_result = content
                pending_by_id[tool_id].metadata["tool_message_name"] = message.get("name")

    if transitions:
        transitions[-1].done = True
        transitions[-1].final_reward = final_reward
    return transitions


def load_trace_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def trajectory_from_trace_files(task_id: str, trace_paths: Iterable[Path], final_reward: float = 0.0) -> Trajectory:
    trajectory = Trajectory(task_id=task_id, final_reward=final_reward)
    for path in trace_paths:
        role = "Planner" if "planner" in path.name.lower() else "Actor"
        for record in load_trace_jsonl(path):
            messages = record.get("messages") or []
            step = record.get("step")
            for transition in flatten_trace_messages(task_id, role, messages, final_reward=final_reward):
                if step is not None:
                    transition.metadata["step"] = step
                transition.metadata["trace_path"] = str(path)
                trajectory.add(transition)
    if trajectory.transitions:
        trajectory.transitions[-1].done = True
    return trajectory


def write_trajectories_jsonl(trajectories: Iterable[Trajectory], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            f.write(json.dumps(trajectory.to_dict(), ensure_ascii=False, default=str) + "\n")


def transition_to_unified_experience(transition: Transition) -> Dict[str, Any]:
    """Convert one transition to the JADE-style flattened buffer schema."""

    return {
        "task_id": transition.task_id,
        "role": transition.role,
        "observation": transition.observation,
        "action": transition.action,
        "action_text": transition.action_text,
        "tool_name": transition.tool_name,
        "tool_args": _safe_json(transition.tool_args),
        "tool_result": _safe_json(transition.tool_result),
        "workspace_delta": transition.workspace_delta,
        "response_mask": transition.response_mask,
        "action_logprob": transition.action_logprob,
        "local_reward": transition.local_reward,
        "final_reward": transition.final_reward,
        "done": transition.done,
        "metadata": transition.metadata,
    }


def trajectory_to_unified_buffer(trajectory: Trajectory) -> List[Dict[str, Any]]:
    """Flatten a trajectory into transition rows for audit/debug storage."""

    rows = [transition_to_unified_experience(transition) for transition in trajectory.transitions]
    if rows:
        rows[-1]["done"] = True
        rows[-1]["final_reward"] = trajectory.final_reward
    return rows


def write_unified_buffer_jsonl(trajectories: Iterable[Trajectory], path: Path) -> None:
    """Write flattened transition rows as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            for row in trajectory_to_unified_buffer(trajectory):
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


__all__ = [
    "Trajectory",
    "Transition",
    "flatten_trace_messages",
    "load_trace_jsonl",
    "trajectory_from_trace_files",
    "trajectory_to_unified_buffer",
    "transition_to_unified_experience",
    "write_trajectories_jsonl",
    "write_unified_buffer_jsonl",
]
