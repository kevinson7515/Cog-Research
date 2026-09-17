# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
"""MiroEval runner for Co-Sight.

This entry point intentionally leaves ``CoSight.py`` untouched. It reuses the
same CoSight runtime, but reads MiroEval text-only and multimodal records where
the task text lives in ``rewritten_query``. Multimodal records may include local
image or document attachments under the ``files`` field.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

from CoSight import CoSight
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


def _env_value(*names: str) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return None


def _safe_task_name(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip() or fallback
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE).strip("._")
    return text or fallback


def _split_path_list(raw: str) -> List[str]:
    return [part.strip() for part in re.split(r"[;,]", raw) if part.strip()]


def _resolve_path(path: str, *base_dirs: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)

    for base_dir in base_dirs:
        candidate = os.path.abspath(os.path.join(base_dir, expanded))
        if os.path.exists(candidate):
            return candidate

    return os.path.abspath(os.path.join(base_dirs[0], expanded))


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def _infer_source_name(path: str, fallback: str) -> str:
    name = os.path.basename(path).lower()
    if "multimodal" in name:
        return "multimodal"
    if "text" in name:
        return "text"
    return fallback


def _resolve_task_sources(base_dir: str) -> List[Tuple[str, str]]:
    explicit_paths = _env_value("MIROEVAL_TASKS_PATH", "DEEPRESEARCH_TASKS_PATH", "QUIZ_FILE_PATH")
    if explicit_paths:
        sources = []
        for index, raw_path in enumerate(_split_path_list(explicit_paths), 1):
            path = _resolve_path(raw_path, base_dir)
            sources.append((_infer_source_name(path, f"custom_{index}"), path))
        return sources

    split = (_env_value("MIROEVAL_SPLIT") or "all").lower().replace("_", "-")
    text_path = _resolve_path(
        _env_value("MIROEVAL_TEXT_TASKS_PATH") or os.path.join("data", "mirobench_text.json"),
        base_dir,
    )
    multimodal_path = _resolve_path(
        _env_value("MIROEVAL_MULTIMODAL_TASKS_PATH") or os.path.join("data", "mirobench_multimodal.json"),
        base_dir,
    )

    if split in {"all", "both", "full"}:
        return [("text", text_path), ("multimodal", multimodal_path)]
    if split in {"text", "text-only", "textonly"}:
        return [("text", text_path)]
    if split in {"multimodal", "mm", "multi-modal"}:
        return [("multimodal", multimodal_path)]

    raise ValueError(
        "Invalid MIROEVAL_SPLIT. Use one of: all, text, multimodal; "
        "or set MIROEVAL_TASKS_PATH for a custom file."
    )


def _iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_num, json.loads(line)


def _iter_records(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Yield records from MiroEval JSON arrays, JSON objects, or JSONL files."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(4096).lstrip()[:1]
        f.seek(0)
        if first_char in {"[", "{"}:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                payload = None

            if isinstance(payload, list):
                for item_num, item in enumerate(payload, 1):
                    if isinstance(item, dict):
                        yield item_num, item
                    else:
                        raise ValueError(f"Record {item_num} is not a JSON object")
                return

            if isinstance(payload, dict):
                for key in ("data", "tasks", "items"):
                    records = payload.get(key)
                    if isinstance(records, list):
                        for item_num, item in enumerate(records, 1):
                            if isinstance(item, dict):
                                yield item_num, item
                            else:
                                raise ValueError(f"Record {item_num} in {key} is not a JSON object")
                        return
                yield 1, payload
                return

    yield from _iter_jsonl(path)


def _attachment_base_dir(base_dir: str) -> str:
    raw = _env_value("MIROEVAL_ATTACHMENT_BASE_DIR", "MIROEVAL_DATA_DIR")
    if raw:
        return _resolve_path(raw, base_dir)
    return os.path.join(base_dir, "data")


def _build_attached_files(
    record: Dict[str, Any],
    *,
    base_dir: str,
    task_file_path: str,
    attachment_base_dir: str,
) -> Tuple[List[str], List[str]]:
    raw_files = record.get("files") or record.get("attachments") or []
    if isinstance(raw_files, dict):
        raw_files = [raw_files]
    if not isinstance(raw_files, list):
        raise ValueError("Record files/attachments field must be a list or object")

    attached_files: List[str] = []
    missing_files: List[str] = []
    task_file_dir = os.path.dirname(task_file_path)
    strict_attachments = _env_bool("MIROEVAL_STRICT_ATTACHMENTS", True)

    for index, item in enumerate(raw_files, 1):
        if isinstance(item, str):
            raw_path = item
        elif isinstance(item, dict):
            raw_path = (
                item.get("dir")
                or item.get("path")
                or item.get("file")
                or item.get("filename")
                or item.get("url")
            )
        else:
            raise ValueError(f"Attachment {index} is not a string or object")

        raw_path = str(raw_path or "").strip()
        if not raw_path:
            raise ValueError(f"Attachment {index} has no dir/path/file/filename/url")

        if _looks_like_url(raw_path):
            logger.warning(f"URL attachment is not supported by CoSight attached_files: {raw_path}")
            missing_files.append(raw_path)
            continue

        resolved_path = _resolve_path(raw_path, attachment_base_dir, task_file_dir, base_dir)
        if os.path.exists(resolved_path):
            attached_files.append(resolved_path)
        else:
            missing_files.append(resolved_path)

    if missing_files and strict_attachments:
        raise FileNotFoundError("Missing MiroEval attachment(s): " + ", ".join(missing_files))

    for path in missing_files:
        logger.warning(f"Missing MiroEval attachment: {path}")

    return attached_files, missing_files


def _attachment_summary(record: Dict[str, Any], attached_files: Sequence[str]) -> str:
    if not attached_files:
        return ""

    raw_files = record.get("files") or record.get("attachments") or []
    if isinstance(raw_files, dict):
        raw_files = [raw_files]
    file_types = [
        str(item.get("type") or os.path.splitext(str(item.get("filename") or ""))[1].lstrip(".") or "file")
        for item in raw_files
        if isinstance(item, dict)
    ]
    counts = Counter(file_types or ["file"] * len(attached_files))
    return ", ".join(f"{count} {file_type}" for file_type, count in sorted(counts.items()))


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


def _build_question(record: Dict[str, Any], attached_files: Sequence[str]) -> str:
    prompt = str(
        record.get("rewritten_query")
        or record.get("prompt")
        or record.get("query")
        or record.get("body")
        or record.get("task")
        or ""
    ).strip()
    if not prompt:
        raise ValueError("Record has no rewritten_query/prompt/query/body/task field")

    include_description = _env_bool(
        "MIROEVAL_INCLUDE_DESCRIPTION",
        _env_bool("DEEPRESEARCH_INCLUDE_DESCRIPTION", False),
    )
    description = str(record.get("description") or "").strip()
    if include_description and description:
        prompt = f"[{description}]\n{prompt}"

    # Some records already include a highest-priority blocked-source instruction
    # in prompt. Append the structured rule only when those URLs are absent.
    blocked = record.get("blocked") if isinstance(record.get("blocked"), dict) else {}
    blocked_urls = [str(url) for url in (blocked.get("urls") or [])]
    if blocked_urls and not any(url in prompt for url in blocked_urls):
        prompt = prompt + _format_blocked_sources(record)

    if _env_bool(
        "MIROEVAL_INCLUDE_RUBRIC_IN_PROMPT",
        _env_bool("DEEPRESEARCH_INCLUDE_RUBRIC_IN_PROMPT", False),
    ):
        content = record.get("content") if isinstance(record.get("content"), dict) else {}
        rubric = record.get("rubric") or content.get("rubric")
        if rubric:
            prompt += "\n\nEvaluation rubric for self-checking only:\n"
            prompt += json.dumps(rubric, ensure_ascii=False, indent=2)

    if attached_files:
        summary = _attachment_summary(record, attached_files)
        suffix = (
            "This is a MiroEval multimodal research task with local attachments"
            f"{f' ({summary})' if summary else ''}. "
            "Inspect and use the attached images/documents when they are relevant; "
            "use open web pages, documents, datasets, and accessible URLs for additional "
            "evidence and cross-checking. Preserve the task language. If the user has "
            "not specified any particular requirements for the final output, you usually "
            "need to generate a final markdown report in response."
        )
    else:
        suffix = (
            "This is a text-only MiroEval research task with no local attachments. "
            "Use open web pages, documents, datasets, and accessible URLs for evidence, "
            "cross-checking, and the final markdown report. Preserve the task language. "
            "If the user has not specified any particular requirements for the final output, "
            "you usually need to generate a final markdown report in response."
        )
    return f"{prompt}\n\n{suffix}"


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    task_sources = _resolve_task_sources(base_dir)
    attachment_base_dir = _attachment_base_dir(base_dir)
    trace_dir = os.getenv(
        "MIROEVAL_TRACE_DIR",
        os.getenv("COSIGHT_TRACE_DIR", os.path.join(base_dir, "trace_miroeval")),
    )
    workspace_root = os.getenv(
        "MIROEVAL_WORKSPACE_ROOT",
        os.getenv(
            "DEEPRESEARCH_WORKSPACE_ROOT",
            os.path.join(base_dir, "work_space", f"miroeval_{timestamp}"),
        ),
    )
    output_file_path = os.getenv(
        "MIROEVAL_RESULTS_PATH",
        os.getenv(
            "DEEPRESEARCH_RESULTS_PATH",
            os.path.join(base_dir, f"miroeval_results_{timestamp}.jsonl"),
        ),
    )

    start_index = _env_int("MIROEVAL_START_INDEX", _env_int("DEEPRESEARCH_START_INDEX", 1))
    max_tasks = _env_int("MIROEVAL_MAX_TASKS", _env_int("DEEPRESEARCH_MAX_TASKS", 0))

    missing_task_files = [path for _, path in task_sources if not os.path.exists(path)]
    if missing_task_files:
        logger.error(f"Task file(s) not found: {missing_task_files}")
        return

    os.makedirs(workspace_root, exist_ok=True)
    os.makedirs(os.path.dirname(output_file_path) or base_dir, exist_ok=True)

    logger.info("Starting MiroEval run")
    logger.info(f"Task sources: {task_sources}")
    logger.info(f"Attachment base dir: {attachment_base_dir}")
    logger.info(f"Workspace root: {workspace_root}")
    logger.info(f"Results JSONL: {output_file_path}")
    logger.info(f"Start index: {start_index}; max tasks: {max_tasks or 'all'}")

    processed = 0
    use_split_subdirs = len(task_sources) > 1
    stop_requested = False

    for source_name, task_file_path in task_sources:
        logger.info(f"Loading MiroEval {source_name} source: {task_file_path}")
        source_workspace_root = (
            os.path.join(workspace_root, source_name) if use_split_subdirs else workspace_root
        )

        for line_num, task_data in _iter_records(task_file_path):
            idx = task_data.get("idx") or task_data.get("id") or line_num
            try:
                numeric_idx = int(idx)
            except Exception:
                numeric_idx = line_num
            if numeric_idx < start_index:
                continue
            if max_tasks and processed >= max_tasks:
                stop_requested = True
                break

            raw_id = task_data.get("id") or idx or f"line_{line_num}"
            task_name = _safe_task_name(idx, f"line_{line_num}")
            id_name = _safe_task_name(raw_id, task_name)
            task_work_space = os.path.join(source_workspace_root, f"task_{task_name}")
            os.makedirs(task_work_space, exist_ok=True)

            attached_files: List[str] = []
            missing_attachments: List[str] = []
            try:
                attached_files, missing_attachments = _build_attached_files(
                    task_data,
                    base_dir=base_dir,
                    task_file_path=task_file_path,
                    attachment_base_dir=attachment_base_dir,
                )
                question = _build_question(task_data, attached_files)
                logger.info(
                    f"========== Start MiroEval {source_name} task: {raw_id} "
                    f"(idx={idx}; attachments={len(attached_files)}) =========="
                )

                cosight = CoSight(
                    plan_llm=llm_for_plan,
                    act_llm=llm_for_act,
                    tool_llm=llm_for_tool,
                    vision_llm=llm_for_vision,
                    work_space_path=task_work_space,
                    message_uuid=f"miroeval_{source_name}_{task_name}_{id_name}_{int(time.time())}",
                    trace_dir=trace_dir,
                )

                result = cosight.execute(
                    question=question,
                    attached_files=attached_files,
                    output_format="markdown",
                )

                output_data = task_data.copy()
                output_data["miroeval_split"] = source_name
                output_data["cosight_result"] = result
                output_data["cosight_report_path"] = cosight.plan.get_final_report_path()
                output_data["workspace"] = task_work_space
                output_data["resolved_attached_files"] = attached_files
                if missing_attachments:
                    output_data["missing_attachments"] = missing_attachments
                with open(output_file_path, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

                logger.info(f"MiroEval {source_name} task finished: {raw_id}")
                processed += 1
            except Exception as exc:
                logger.error(f"MiroEval {source_name} task failed: {raw_id}; {exc}", exc_info=True)
                output_data = task_data.copy()
                output_data["miroeval_split"] = source_name
                output_data["cosight_error"] = str(exc)
                output_data["workspace"] = task_work_space
                output_data["resolved_attached_files"] = attached_files
                if missing_attachments:
                    output_data["missing_attachments"] = missing_attachments
                with open(output_file_path, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                processed += 1

        if stop_requested:
            break

    logger.info("========== MiroEval run finished ==========")


if __name__ == "__main__":
    main()
