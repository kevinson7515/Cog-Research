"""Offline reward audit for Co-Sight workspaces.

Example:
  python -m cosight_rl.rewards.audit_rewards \
    --workspace ./work_space/work_space_20260621_075554 \
    --logs ./logs \
    --output ./outputs/reward_audit_20260621.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .reward_evaluator import evaluate_report


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
KNOWN_TOOL_NAMES = {
    "ask_question_about_image",
    "ask_question_about_video",
    "audio_recognition",
    "serper_search",
    "tavily_search",
    "search",
    "image_search",
    "fetch_website_content",
    "fetch_website_content_with_images",
    "fetch_website_images_only",
    "file_saver",
    "file_read",
    "file_find_in_content",
    "extract_document_content",
    "execute_code",
    "generate_markdown_report",
    "mark_step",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Workspace root containing task_* directories.")
    parser.add_argument("--logs", default="", help="Optional log directory for tool-call metadata fallback.")
    parser.add_argument("--output", required=True, help="JSONL output path.")
    parser.add_argument("--summary-output", default="", help="Summary JSON path. Defaults to output + .summary.json.")
    parser.add_argument("--metadata-jsonl", default="", help="Optional quiz_results/benchmark JSONL with id/body/image_url.")
    parser.add_argument("--min-report-chars", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-all-md", action="store_true", help="Score every markdown file instead of one final report per task.")
    return parser.parse_args()


def _task_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"task_(\d+)", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def iter_task_dirs(workspace: Path) -> Iterable[Path]:
    yield from sorted((path for path in workspace.glob("task_*") if path.is_dir()), key=_task_sort_key)


def _candidate_score(path: Path, text_len: int) -> float:
    name = path.name.lower()
    score = min(50.0, text_len / 2000.0)
    if any(token in name for token in ("report", "\u62a5\u544a", "analysis", "\u7814\u7a76", "\u5206\u6790")):
        score += 8.0
    if re.search(r"\d{8}_\d{6}", name):
        score += 4.0
    if any(token in name for token in ("image_analysis", "evidence", "sources", "\u68c0\u7d22", "\u8bc1\u636e", "\u7ebf\u7d22", "step")):
        score -= 6.0
    return score


def find_markdown_reports(task_dir: Path, min_chars: int, include_all: bool) -> List[Path]:
    candidates: List[tuple[float, Path]] = []
    for path in task_dir.glob("*.md"):
        try:
            text_len = len(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if text_len < min_chars:
            continue
        candidates.append((_candidate_score(path, text_len), path))
    candidates.sort(key=lambda item: (item[0], item[1].stat().st_mtime), reverse=True)
    if include_all:
        return [path for _, path in candidates]
    return [candidates[0][1]] if candidates else []


def _default_metadata_candidates(workspace: Path) -> List[Path]:
    root = workspace.resolve().parents[1] if len(workspace.resolve().parents) >= 2 else workspace.resolve().parent
    return sorted(root.glob("quiz_results_*.jsonl"), reverse=True)


def load_task_metadata(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path or not path.exists():
        return {}
    metadata: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_id = record.get("id", record.get("qid", line_no - 1))
            task_id = f"task_{raw_id}"
            metadata[task_id] = record
    return metadata


def _image_names_from_task(task_dir: Path) -> List[str]:
    return sorted(path.name for path in task_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _image_names_from_metadata(record: Mapping[str, Any]) -> List[str]:
    raw_images = record.get("image_url") or record.get("images") or record.get("image_paths") or []
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    names: List[str] = []
    for item in raw_images:
        if isinstance(item, Mapping):
            item = item.get("path") or item.get("url") or item.get("image") or item.get("image_url")
        if item:
            names.append(Path(str(item).replace("file://", "")).name)
    return sorted(set(names))


def _extract_trace_tool_names(task_dir: Path) -> List[str]:
    names: List[str] = []
    for trace_path in task_dir.glob("**/*trace*.jsonl"):
        if trace_path.stat().st_size > 20_000_000:
            continue
        try:
            with trace_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    for match in re.finditer(r'"name"\s*:\s*"([^"]+)"', line):
                        names.append(match.group(1))
                    for tool_name in KNOWN_TOOL_NAMES:
                        if tool_name in line:
                            names.append(tool_name)
        except OSError:
            continue
    return names


def _extract_log_tool_names(logs_dir: Optional[Path], task_id: str) -> List[str]:
    """Best-effort tool-name extraction from global logs."""

    if not logs_dir or not logs_dir.exists():
        return []
    match = re.search(r"task_(\d+)", task_id)
    markers = {task_id}
    if match:
        raw_id = match.group(1)
        markers.update({f"task {raw_id}", f"task_id={raw_id}", f'"task_id": {raw_id}', f"task_{raw_id}"})

    names: List[str] = []
    for path in logs_dir.glob("*"):
        if not path.is_file() or path.suffix.lower() not in {".log", ".out", ".txt", ".jsonl"}:
            continue
        try:
            if path.stat().st_size > 80_000_000:
                continue
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not any(marker in line for marker in markers):
                        continue
                    for tool_name in KNOWN_TOOL_NAMES:
                        if tool_name in line:
                            names.append(tool_name)
        except OSError:
            continue
    return names


def build_extra_info(
    task_dir: Path,
    metadata: Mapping[str, Dict[str, Any]],
    logs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    task_meta = metadata.get(task_dir.name, {})
    images = _image_names_from_metadata(task_meta) or _image_names_from_task(task_dir)
    trace_tool_names = _extract_trace_tool_names(task_dir)
    log_tool_names = _extract_log_tool_names(logs_dir, task_dir.name)
    return {
        "task_id": task_dir.name,
        "task_prompt": task_meta.get("body") or task_meta.get("question") or "",
        "expected_images": images,
        "expected_format": task_meta.get("expected_format", ""),
        "language": task_meta.get("language"),
        "tags": task_meta.get("tags", []),
        "difficulty": task_meta.get("difficulty"),
        "tool_names": trace_tool_names or log_tool_names,
    }


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * pct))))
    return values[idx]


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [float(row["final_reward"]) for row in rows]
    dup_ratios = [float(row["repetition_metrics"].get("duplicate_paragraph_ratio", 0.0)) for row in rows]
    severe = [row for row in rows if row["repetition_metrics"].get("severe_repetition")]
    tool_counts = [int(row.get("details", {}).get("tool_call_count", 0)) for row in rows]
    estimated_tokens = [int(row.get("details", {}).get("estimated_tokens", 0)) for row in rows]
    hard_gate_counter: Counter[str] = Counter()
    for row in rows:
        hard_gate_counter.update(row.get("hard_gate_reasons") or [])

    top_repeats = sorted(
        rows,
        key=lambda row: (
            int(row["repetition_metrics"].get("max_repeated_paragraph_count", 0)),
            int(row["repetition_metrics"].get("max_repeated_sentence_count", 0)),
        ),
        reverse=True,
    )[:20]
    worst = sorted(rows, key=lambda row: float(row["final_reward"]))[:20]
    hard_failure_rates = {
        reason: hard_gate_counter.get(reason, 0) / max(1, len(rows))
        for reason in [
            "wrong_visual_identity",
            "fabricated_source_data",
            "repetition_collapse",
            "format_instruction_fail",
            "core_quant_error",
            "empty_or_crashed",
        ]
    }
    return {
        "records": len(rows),
        "average_reward": mean(rewards) if rewards else 0.0,
        "average_tool_calls": mean(tool_counts) if tool_counts else 0.0,
        "average_estimated_report_tokens": mean(estimated_tokens) if estimated_tokens else 0.0,
        "severe_repetition_count": len(severe),
        "severe_repetition_rate": len(severe) / max(1, len(rows)),
        "duplicate_paragraph_ratio_distribution": {
            "min": min(dup_ratios) if dup_ratios else 0.0,
            "mean": mean(dup_ratios) if dup_ratios else 0.0,
            "p50": _percentile(dup_ratios, 0.50),
            "p90": _percentile(dup_ratios, 0.90),
            "max": max(dup_ratios) if dup_ratios else 0.0,
        },
        "hard_gate_reason_counts": dict(hard_gate_counter.most_common()),
        "hard_failure_rates": hard_failure_rates,
        "max_repeated_paragraph_count_top20": [
            {
                "task_id": row["task_id"],
                "report_path": row["report_path"],
                "max_repeated_paragraph_count": row["repetition_metrics"].get("max_repeated_paragraph_count"),
                "max_repeated_sentence_count": row["repetition_metrics"].get("max_repeated_sentence_count"),
                "duplicate_paragraph_ratio": row["repetition_metrics"].get("duplicate_paragraph_ratio"),
                "duplicate_sentence_ratio": row["repetition_metrics"].get("duplicate_sentence_ratio"),
                "final_reward": row["final_reward"],
            }
            for row in top_repeats
        ],
        "worst_samples_top20": [
            {
                "task_id": row["task_id"],
                "report_path": row["report_path"],
                "final_reward": row["final_reward"],
                "hard_gate_reasons": row.get("hard_gate_reasons", []),
                "short_diagnosis": row.get("short_diagnosis", ""),
            }
            for row in worst
        ],
    }


def audit_workspace(
    workspace: Path,
    output: Path,
    summary_output: Path,
    metadata: Mapping[str, Dict[str, Any]],
    min_report_chars: int,
    logs_dir: Optional[Path] = None,
    limit: int = 0,
    include_all_md: bool = False,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    task_dirs = list(iter_task_dirs(workspace))
    if limit > 0:
        task_dirs = task_dirs[:limit]

    with output.open("w", encoding="utf-8") as f:
        for task_dir in task_dirs:
            extra_info = build_extra_info(task_dir, metadata, logs_dir)
            reports = find_markdown_reports(task_dir, min_report_chars, include_all_md)
            if not reports:
                row = evaluate_report("", task_prompt=extra_info.get("task_prompt", ""), extra_info=extra_info)
                row.update({"task_id": task_dir.name, "report_path": ""})
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            for report_path in reports:
                text = report_path.read_text(encoding="utf-8", errors="ignore")
                row = evaluate_report(text, task_prompt=extra_info.get("task_prompt", ""), extra_info=extra_info)
                row.update(
                    {
                        "task_id": task_dir.name,
                        "report_path": str(report_path),
                        "report_name": report_path.name,
                    }
                )
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(rows)
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    output = Path(args.output)
    summary_output = Path(args.summary_output) if args.summary_output else output.with_suffix(output.suffix + ".summary.json")

    metadata_path: Optional[Path] = Path(args.metadata_jsonl) if args.metadata_jsonl else None
    if metadata_path is None:
        candidates = _default_metadata_candidates(workspace)
        metadata_path = candidates[0] if candidates else None
    metadata = load_task_metadata(metadata_path)

    summary = audit_workspace(
        workspace=workspace,
        output=output,
        summary_output=summary_output,
        metadata=metadata,
        min_report_chars=args.min_report_chars,
        logs_dir=Path(args.logs) if args.logs else None,
        limit=args.limit,
        include_all_md=args.include_all_md,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
