#!/usr/bin/env python3
"""Build an offline Co-Sight evidence corpus from saved traces/workspaces.

The output is keyed by task id so ``prepare_cosight_rl_data.py`` can attach a
small evidence pack to each RL training row. Workspace markdown is the primary
source. Trace extraction is conservative and only keeps short tool-observation
snippets, not full final answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Mapping, Optional


DEFAULT_WORKSPACE_DIR = "work_space/work_space_20260612_134333"
DEFAULT_TRACE_DIR = "trace"
DEFAULT_OUTPUT_DIR = "data/rl_evidence_corpus"
URL_RE = re.compile(r"https?://[^\s)\]}>\"'\u3002\uff0c\uff1b\uff1a\uff01\uff1f]+")
TASK_ID_RE = re.compile(r"(?:^|[/\\])task_(\d+)(?:[/\\]|$)|\btask_(\d+)\b")
TIMESTAMP_RE = re.compile(r"20\d{6}[_-]\d{6}")

EVIDENCE_NAME_HINTS = (
    "检索",
    "来源",
    "验证",
    "资料",
    "证据",
    "图像分析",
    "应用场景",
    "URL",
    "url",
    "source",
    "evidence",
    "search",
)
FINAL_REPORT_HINTS = ("最终报告", "final_report", "final-report", "final_answer", "final-answer")
TRACE_EVIDENCE_TOOLS = {
    "ask_question_about_image",
    "file_read",
    "serper_search",
    "tavily_search",
    "search",
    "image_search",
    "fetch_website_content",
    "fetch_website_content_with_images",
    "extract_document_content",
    "execute_code",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace_dir", default=os.getenv("COSIGHT_WORKSPACE_DIR", DEFAULT_WORKSPACE_DIR))
    parser.add_argument("--trace_dir", default=os.getenv("COSIGHT_TRACE_DIR", DEFAULT_TRACE_DIR))
    parser.add_argument("--output_dir", default=os.getenv("COSIGHT_EVIDENCE_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_tasks", type=int, default=int(os.getenv("COSIGHT_EVIDENCE_MAX_TASKS", "0")))
    parser.add_argument("--max_docs_per_task", type=int, default=int(os.getenv("COSIGHT_MAX_EVIDENCE_DOCS_PER_TASK", "10")))
    parser.add_argument("--max_chars_per_doc", type=int, default=int(os.getenv("COSIGHT_MAX_EVIDENCE_DOC_CHARS", "2400")))
    parser.add_argument("--max_trace_items_per_task", type=int, default=int(os.getenv("COSIGHT_MAX_TRACE_EVIDENCE_PER_TASK", "4")))
    parser.add_argument("--include_trace", action="store_true", default=_env_bool("COSIGHT_INCLUDE_TRACE_EVIDENCE", True))
    parser.add_argument("--no_include_trace", dest="include_trace", action="store_false")
    parser.add_argument("--include_final_reports", action="store_true", default=_env_bool("COSIGHT_INCLUDE_FINAL_REPORT_EVIDENCE", False))
    return parser.parse_args()


def extract_urls(text: str) -> List[str]:
    return sorted(set(url.rstrip(".,;:") for url in URL_RE.findall(text or "")))


def task_id_from_text(text: str) -> Optional[str]:
    match = TASK_ID_RE.search(text or "")
    if not match:
        return None
    return match.group(1) or match.group(2)


def task_sort_key(path: Path) -> tuple[int, str]:
    task_id = task_id_from_text(str(path))
    return (int(task_id), path.name) if task_id is not None else (10**9, path.name)


def read_text(path: Path, max_bytes: int = 1_000_000) -> str:
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text or "")
    text = re.sub(r"<\|[^|]+?\|>", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 24].rstrip() + "\n[... truncated ...]"
    return text


def is_probable_final_report(path: Path) -> bool:
    stem = path.stem
    lower = stem.lower()
    if any(hint in stem or hint in lower for hint in FINAL_REPORT_HINTS):
        return True
    if "最终" in stem and "报告" in stem:
        return True
    # Co-Sight's generated final reports are usually timestamped, while
    # intermediate evidence notes are not.  Exclude timestamped reports even
    # when their filename is a task-specific English title without "report".
    return bool(TIMESTAMP_RE.search(stem))


def workspace_doc_score(path: Path, text: str) -> int:
    name = path.name
    lower = name.lower()
    score = 0
    if extract_urls(text):
        score += 5
    score += sum(2 for hint in EVIDENCE_NAME_HINTS if hint in name or hint in lower)
    if is_probable_final_report(path):
        score -= 20
    if len(text) > 500:
        score += 1
    return score


def item_id(source_type: str, task_id: str, source_ref: str, text: str) -> str:
    raw = f"{source_type}\n{task_id}\n{source_ref}\n{text[:500]}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def make_item(
    task_id: str,
    source_type: str,
    source_ref: str,
    title: str,
    text: str,
    max_chars: int,
) -> Dict[str, Any]:
    compact = compact_text(text, max_chars)
    urls = extract_urls(compact)
    return {
        "id": item_id(source_type, task_id, source_ref, compact),
        "task_id": task_id,
        "source_type": source_type,
        "source_ref": source_ref,
        "title": title,
        "urls": urls,
        "url_count": len(urls),
        "text": compact,
        "char_count": len(compact),
    }


def iter_workspace_items(
    workspace_dir: Path,
    max_tasks: int,
    max_docs_per_task: int,
    max_chars_per_doc: int,
    include_final_reports: bool,
) -> Iterator[Dict[str, Any]]:
    if not workspace_dir.exists():
        return

    task_dirs = [path for path in workspace_dir.iterdir() if path.is_dir() and task_id_from_text(str(path))]
    task_dirs.sort(key=task_sort_key)
    if max_tasks and max_tasks > 0:
        task_dirs = task_dirs[:max_tasks]

    for task_dir in task_dirs:
        task_id = task_id_from_text(str(task_dir))
        if task_id is None:
            continue
        candidates: List[tuple[int, Path, str]] = []
        for path in task_dir.rglob("*.md"):
            if not include_final_reports and is_probable_final_report(path):
                continue
            text = read_text(path)
            if len(text.strip()) < 80:
                continue
            candidates.append((workspace_doc_score(path, text), path, text))
        candidates.sort(key=lambda item: (-item[0], len(item[2]), item[1].name))
        for _, path, text in candidates[:max_docs_per_task]:
            try:
                source_ref = str(path.relative_to(workspace_dir.parent))
            except ValueError:
                source_ref = str(path)
            yield make_item(task_id, "workspace_md", source_ref, path.stem, text, max_chars_per_doc)


def message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def iter_trace_items(
    trace_dir: Path,
    allowed_task_ids: Optional[set[str]],
    max_trace_items_per_task: int,
    max_chars_per_doc: int,
) -> Iterator[Dict[str, Any]]:
    if not trace_dir.exists() or max_trace_items_per_task <= 0:
        return

    counts: DefaultDict[str, int] = defaultdict(int)
    for path in sorted(trace_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                task_id = task_id_from_text(line)
                if task_id is None:
                    continue
                if allowed_task_ids is not None and task_id not in allowed_task_ids:
                    continue
                if counts[task_id] >= max_trace_items_per_task:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for msg in record.get("messages") or []:
                    if not isinstance(msg, Mapping):
                        continue
                    role = str(msg.get("role") or "")
                    tool_name = str(msg.get("name") or "")
                    if role != "tool" or tool_name not in TRACE_EVIDENCE_TOOLS:
                        continue
                    text = compact_text(message_text(msg.get("content")), max_chars_per_doc)
                    if len(text) < 80:
                        continue
                    if not extract_urls(text) and tool_name not in {"ask_question_about_image", "file_read", "extract_document_content"}:
                        continue
                    source_ref = f"{path.name}:{line_no}:{tool_name}"
                    yield make_item(task_id, "trace_tool", source_ref, tool_name, text, max_chars_per_doc)
                    counts[task_id] += 1
                    if counts[task_id] >= max_trace_items_per_task:
                        break


def dedupe_items(items: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for item in items:
        task_id = str(item.get("task_id"))
        signature = (task_id, str(item.get("id")))
        if signature in seen:
            continue
        seen.add(signature)
        grouped[task_id].append(item)
    for task_items in grouped.values():
        task_items.sort(
            key=lambda item: (
                0 if item.get("url_count") else 1,
                0 if item.get("source_type") == "trace_tool" else 1,
                -int(item.get("url_count") or 0),
                int(item.get("char_count") or 0),
            )
        )
    return dict(sorted(grouped.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else 10**9))


def write_outputs(grouped: Mapping[str, List[Dict[str, Any]]], output_dir: Path, manifest: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_path = output_dir / "evidence_corpus.jsonl"
    task_path = output_dir / "task_evidence.json"
    manifest_path = output_dir / "manifest.json"

    item_count = 0
    with flat_path.open("w", encoding="utf-8") as f:
        for items in grouped.values():
            for item in items:
                item_count += 1
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with task_path.open("w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    manifest.update(
        {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "task_count": len(grouped),
            "item_count": item_count,
            "flat_path": str(flat_path),
            "task_evidence_path": str(task_path),
        }
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    workspace_dir = Path(args.workspace_dir)
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)

    workspace_items = list(
        iter_workspace_items(
            workspace_dir=workspace_dir,
            max_tasks=args.max_tasks,
            max_docs_per_task=args.max_docs_per_task,
            max_chars_per_doc=args.max_chars_per_doc,
            include_final_reports=args.include_final_reports,
        )
    )
    allowed_task_ids = {str(item["task_id"]) for item in workspace_items} or None
    trace_items = (
        list(
            iter_trace_items(
                trace_dir=trace_dir,
                allowed_task_ids=allowed_task_ids,
                max_trace_items_per_task=args.max_trace_items_per_task,
                max_chars_per_doc=args.max_chars_per_doc,
            )
        )
        if args.include_trace
        else []
    )
    grouped = dedupe_items([*workspace_items, *trace_items])
    write_outputs(
        grouped,
        output_dir,
        {
            "workspace_dir": str(workspace_dir),
            "trace_dir": str(trace_dir),
            "max_tasks": args.max_tasks,
            "max_docs_per_task": args.max_docs_per_task,
            "max_chars_per_doc": args.max_chars_per_doc,
            "max_trace_items_per_task": args.max_trace_items_per_task,
            "include_trace": args.include_trace,
            "include_final_reports": args.include_final_reports,
            "workspace_item_count": len(workspace_items),
            "trace_item_count": len(trace_items),
        },
    )


if __name__ == "__main__":
    main()
