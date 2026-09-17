
"""Text-only DeepResearch-Bench II runner for Co-Sight.

This entry point intentionally leaves ``CogResearch.py`` untouched. It reuses the
same CogResearch runtime, but reads ``data/tasks_and_rubrics.jsonl`` records where
the task text lives in ``prompt`` and no image attachments are supplied.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
import time
from typing import Any, Dict, Iterable, Tuple

from CogResearch import CogResearch
from app.common.logger_util import logger
from llm import llm_for_act, llm_for_plan, llm_for_tool, llm_for_vision


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid integer for {name}: {raw}; using {default}")
        return default


def _safe_task_name(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip() or fallback
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE).strip("._")
    return text or fallback


def _iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_num, json.loads(line)


def _format_blocked_sources(record: Dict[str, Any]) -> str:
    blocked = record.get("blocked")
    if not isinstance(blocked, dict):
        return ""
    title = blocked.get("title")
    urls = blocked.get("urls") or []
    authors = blocked.get("authors") or []
    parts = []
    if title:
        parts.append(f"Blocked source title: {title}")
    if authors:
        parts.append("Blocked source authors: " + ", ".join(str(item) for item in authors))
    if urls:
        parts.append("Blocked URLs:\n" + "\n".join(f"- {url}" for url in urls))
    if not parts:
        return ""
    return (
        "\n\nImportant source restriction: during research, do not open, use, "
        "or cite the following blocked source. If you encounter it accidentally, "
        "ignore its content.\n"
        + "\n".join(parts)
    )


def _build_question(record: Dict[str, Any]) -> str:
    prompt = str(record.get("prompt") or record.get("body") or record.get("task") or "").strip()
    if not prompt:
        raise ValueError("Record has no prompt/body/task field")

    include_description = _env_bool("DEEPRESEARCH_INCLUDE_DESCRIPTION", False)
    description = str(record.get("description") or "").strip()
    if include_description and description:
        prompt = f"[{description}]\n{prompt}"

    # Some records already include a highest-priority blocked-source instruction
    # in prompt. Append the structured rule only when those URLs are absent.
    blocked = record.get("blocked") if isinstance(record.get("blocked"), dict) else {}
    blocked_urls = [str(url) for url in (blocked.get("urls") or [])]
    if blocked_urls and not any(url in prompt for url in blocked_urls):
        prompt = prompt + _format_blocked_sources(record)

    if _env_bool("DEEPRESEARCH_INCLUDE_RUBRIC_IN_PROMPT", False):
        content = record.get("content") if isinstance(record.get("content"), dict) else {}
        rubric = record.get("rubric") or content.get("rubric")
        if rubric:
            prompt += "\n\nEvaluation rubric for self-checking only:\n"
            prompt += json.dumps(rubric, ensure_ascii=False, indent=2)

    suffix = (
        "This is a text-only deep research task with no local image attachments. "
        "Use open web pages, documents, datasets, and accessible URLs for evidence, "
        "cross-checking, and the final markdown report. Preserve the task language. "
        "If the user has not specified any particular requirements for the final output, "
        "you usually need to generate a final markdown report in response."
    )
    return f"{prompt}\n\n{suffix}"


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    task_file_path = os.getenv(
        "DEEPRESEARCH_TASKS_PATH",
        os.getenv(
            "QUIZ_FILE_PATH",
            os.path.join(base_dir, "data", "tasks_and_rubrics.jsonl"),
        ),
    )
    trace_dir = os.getenv("COSIGHT_TRACE_DIR", os.path.join(base_dir, "trace_deepresearch"))
    workspace_root = os.getenv(
        "DEEPRESEARCH_WORKSPACE_ROOT",
        os.path.join(base_dir, "work_space", f"deepresearch_bench_{timestamp}"),
    )
    output_file_path = os.getenv(
        "DEEPRESEARCH_RESULTS_PATH",
        os.path.join(base_dir, f"deepresearch_bench_results_{timestamp}.jsonl"),
    )

    start_index = _env_int("DEEPRESEARCH_START_INDEX", 1)
    max_tasks = _env_int("DEEPRESEARCH_MAX_TASKS", 0)

    if not os.path.exists(task_file_path):
        logger.error(f"Task file not found: {task_file_path}")
        return

    os.makedirs(workspace_root, exist_ok=True)
    os.makedirs(os.path.dirname(output_file_path) or base_dir, exist_ok=True)

    logger.info(f"Starting DeepResearch-Bench II text-only run: {task_file_path}")
    logger.info(f"Workspace root: {workspace_root}")
    logger.info(f"Results JSONL: {output_file_path}")
    logger.info(f"Start index: {start_index}; max tasks: {max_tasks or 'all'}")

    processed = 0
    for line_num, task_data in _iter_jsonl(task_file_path):
        idx = task_data.get("idx") or line_num
        try:
            numeric_idx = int(idx)
        except Exception:
            numeric_idx = line_num
        if numeric_idx < start_index:
            continue
        if max_tasks and processed >= max_tasks:
            break

        raw_id = task_data.get("id") or idx or f"line_{line_num}"
        task_name = _safe_task_name(idx, f"line_{line_num}")
        id_name = _safe_task_name(raw_id, task_name)
        task_work_space = os.path.join(workspace_root, f"task_{task_name}")
        os.makedirs(task_work_space, exist_ok=True)

        try:
            question = _build_question(task_data)
            logger.info(f"========== Start DeepResearch task: {raw_id} (idx={idx}) ==========")

            cosight = CogResearch(
                plan_llm=llm_for_plan,
                act_llm=llm_for_act,
                tool_llm=llm_for_tool,
                vision_llm=llm_for_vision,
                work_space_path=task_work_space,
                message_uuid=f"deepresearch_{task_name}_{id_name}_{int(time.time())}",
                trace_dir=trace_dir,
            )

            result = cosight.execute(
                question=question,
                attached_files=[],
                output_format="markdown",
            )

            output_data = task_data.copy()
            output_data["cosight_result"] = result
            output_data["workspace"] = task_work_space
            with open(output_file_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

            logger.info(f"DeepResearch task finished: {raw_id}")
            processed += 1
        except Exception as exc:
            logger.error(f"DeepResearch task failed: {raw_id}; {exc}", exc_info=True)
            output_data = task_data.copy()
            output_data["cosight_error"] = str(exc)
            output_data["workspace"] = task_work_space
            with open(output_file_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            processed += 1

    logger.info("========== DeepResearch-Bench II text-only run finished ==========")


if __name__ == "__main__":
    main()
