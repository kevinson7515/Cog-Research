"""Co-Sight environment wrapper for reward audits and PPO dry runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from cosight_rl.rewards.reward_evaluator import evaluate_report

from .role_prompts import ROLE_PROMPTS
from .trajectory import Trajectory, Transition, trajectory_from_trace_files


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass
class CosightTask:
    task_id: str
    prompt: str
    image_paths: List[str] = field(default_factory=list)
    ground_truth: Any = None
    expected_format: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CosightEnvConfig:
    workspace_root: Path = Path("work_space/cosight_rl_rollouts")
    output_format: str = "markdown"
    trace_dir_name: str = "traces"
    dry_run: bool = True


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def _resolve_images(raw_images: Any, image_base_dir: Path) -> List[str]:
    if raw_images is None:
        return []
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    images: List[str] = []
    for item in raw_images:
        if isinstance(item, dict):
            item = item.get("path") or item.get("url") or item.get("image") or item.get("image_url")
        if not item:
            continue
        value = str(item).replace("file://", "")
        if value.startswith(("http://", "https://")):
            images.append(value)
            continue
        path = Path(value)
        if not path.is_absolute():
            path = image_base_dir / path
        images.append(str(path))
    return images


def load_jsonl_tasks(path: str | Path, image_base_dir: str | Path = "data/images") -> List[CosightTask]:
    """Load Co-Sight tasks from quiz/benchmark JSONL."""

    image_base = Path(image_base_dir)
    tasks: List[CosightTask] = []
    for idx, record in enumerate(_iter_jsonl(Path(path))):
        task_id = str(record.get("id", record.get("qid", idx)))
        prompt = str(record.get("body") or record.get("query") or record.get("question") or record.get("prompt") or "")
        raw_images = record.get("image_url") or record.get("images") or record.get("image_paths")
        tasks.append(
            CosightTask(
                task_id=task_id,
                prompt=prompt,
                image_paths=_resolve_images(raw_images, image_base),
                ground_truth=record.get("ground_truth") or record.get("answer") or record.get("rubric"),
                expected_format=record.get("expected_format", ""),
                metadata={k: v for k, v in record.items() if k not in {"body", "query", "question", "prompt"}},
            )
        )
    return tasks


class CoSightRLEnv:
    """Minimal adapter around Co-Sight task execution and existing workspaces."""

    def __init__(self, config: Optional[CosightEnvConfig] = None):
        self.config = config or CosightEnvConfig()
        self.config.workspace_root.mkdir(parents=True, exist_ok=True)

    def task_workspace(self, task: CosightTask) -> Path:
        return self.config.workspace_root / f"task_{task.task_id}"

    def prepare_workspace(self, task: CosightTask) -> Path:
        workspace = self.task_workspace(task)
        workspace.mkdir(parents=True, exist_ok=True)
        for image in task.image_paths:
            path = Path(image)
            if path.exists() and path.suffix.lower() in IMAGE_SUFFIXES:
                dest = workspace / path.name
                if not dest.exists():
                    shutil.copy2(path, dest)
        return workspace

    def build_initial_observation(self, task: CosightTask) -> str:
        image_lines = "\n".join(f"- {Path(image).name}: {image}" for image in task.image_paths)
        return (
            f"Task id: {task.task_id}\n"
            f"User task:\n{task.prompt}\n\n"
            f"Images:\n{image_lines or '(none)'}\n\n"
            "Shared role prompts:\n"
            + "\n".join(f"[{role}] {prompt}" for role, prompt in ROLE_PROMPTS.items())
        )

    def rollout_existing_workspace(self, task_dir: str | Path, task_prompt: str = "") -> Trajectory:
        """Build a trajectory and reward from an existing Co-Sight task folder."""

        task_path = Path(task_dir)
        report = self._find_final_report(task_path)
        report_text = report.read_text(encoding="utf-8", errors="ignore") if report else ""
        images = sorted(path.name for path in task_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        reward = evaluate_report(report_text, task_prompt=task_prompt, extra_info={"expected_images": images})
        trace_paths = list(task_path.glob("**/*trace*.jsonl"))
        trajectory = trajectory_from_trace_files(task_path.name, trace_paths, final_reward=reward["final_reward"])
        if not trajectory.transitions:
            trajectory.add(
                Transition(
                    task_id=task_path.name,
                    role="ReportExecutor",
                    observation=task_prompt,
                    action="final_report",
                    action_text=report_text,
                    final_reward=reward["final_reward"],
                    done=True,
                    metadata={"report_path": str(report) if report else "", "reward": reward},
                )
            )
        trajectory.final_reward = reward["final_reward"]
        trajectory.metadata.update({"reward": reward, "report_path": str(report) if report else ""})
        return trajectory

    def dry_run_task(self, task: CosightTask) -> Trajectory:
        """Create a no-GPU smoke trajectory for config/reward verification."""

        workspace = self.prepare_workspace(task)
        observation = self.build_initial_observation(task)
        placeholder = (
            "# Dry run report\n\n"
            "This placeholder verifies task loading, image path handling, trajectory schema, and reward wiring."
        )
        reward = evaluate_report(
            placeholder,
            task_prompt=task.prompt,
            extra_info={
                "expected_images": [Path(path).name for path in task.image_paths],
                "tool_names": [],
                "task_id": task.task_id,
            },
            ground_truth=task.ground_truth,
        )
        transition = Transition(
            task_id=task.task_id,
            role="ReportExecutor",
            observation=observation,
            action="dry_run",
            action_text=placeholder,
            workspace_delta=[str(workspace)],
            final_reward=reward["final_reward"],
            done=True,
            metadata={"reward": reward},
        )
        return Trajectory(task_id=task.task_id, transitions=[transition], final_reward=reward["final_reward"], metadata={"dry_run": True})

    def execute_task(self, task: CosightTask) -> Trajectory:
        """Run the real Co-Sight stack, then convert traces/report to a trajectory.

        This requires configured model endpoints in ``.env``. PPO dry-runs should
        call ``dry_run_task``; production data generation can call this method.
        """

        workspace = self.prepare_workspace(task)
        try:
            from CoSight import CoSight
            from llm import llm_for_act, llm_for_plan, llm_for_tool, llm_for_vision
        except Exception as exc:
            raise RuntimeError("Could not import CoSight runtime; check project dependencies and .env.") from exc

        cosight = CoSight(
            llm_for_plan,
            llm_for_act,
            llm_for_tool,
            llm_for_vision,
            work_space_path=str(workspace),
            message_uuid=f"rl_{task.task_id}",
            trace_dir=str(workspace / self.config.trace_dir_name),
        )
        cosight.execute(task.prompt, attached_files=task.image_paths, output_format=self.config.output_format)
        return self.rollout_existing_workspace(workspace, task_prompt=task.prompt)

    def _find_final_report(self, task_dir: Path) -> Optional[Path]:
        candidates: List[tuple[float, Path]] = []
        for path in task_dir.glob("*.md"):
            text_len = len(path.read_text(encoding="utf-8", errors="ignore"))
            score = text_len / 1000.0
            lower = path.name.lower()
            if "report" in lower or "报告" in path.name:
                score += 10
            if "image_analysis" in lower or "evidence" in lower or "证据" in path.name:
                score -= 5
            candidates.append((score, path))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]


__all__ = ["CoSightRLEnv", "CosightEnvConfig", "CosightTask", "load_jsonl_tasks"]
