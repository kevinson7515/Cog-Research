#!/usr/bin/env python3
"""Build focused vision SFT data from Co-Sight strong-model traces/workspaces.

The output format is the multimodal chat JSONL expected by
scripts/train_vision_sft.py:

  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "/abs/path/to/image.png"},
          {"type": "text", "text": "..."}
        ]
      },
      {"role": "assistant", "content": "..."}
    ],
    "images": ["/abs/path/to/image.png"],
    "meta": {...}
  }

It intentionally extracts only image-understanding turns/files so a follow-up
run can train the visual tower without touching the workflow/tool-policy model.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TRACE_FILES = ("actor_trace.jsonl",)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_dir", default="trace", help="Directory containing actor_trace.jsonl.")
    parser.add_argument(
        "--workspace_dir",
        default="work_space/work_space_20260612_134333",
        help="Strong-model workspace containing task_* directories.",
    )
    parser.add_argument("--quiz_file", default="data/quiz_train.jsonl", help="Training quiz JSONL used to recover task prompts.")
    parser.add_argument("--score_file", default="co-sight-vl-8b-wf-sft.jsonl", help="Optional eval score JSONL for failure taxonomy stats only.")
    parser.add_argument("--output_dir", default="data/vision_sft", help="Directory for generated vision SFT files.")
    parser.add_argument("--output_file", default="", help="Optional explicit output JSONL path.")
    parser.add_argument("--image_base_dir", default="data/images", help="Directory containing original benchmark images.")
    parser.add_argument("--image_search_dir", action="append", default=[], help="Extra directories to index by image basename.")
    parser.add_argument("--path_map", action="append", default=[], help="Rewrite absolute paths as FROM=TO. Can be repeated.")
    parser.add_argument("--include_trace_tool_results", type=parse_bool, default=True, help="Extract ask_question_about_image tool result turns.")
    parser.add_argument("--include_workspace_md", type=parse_bool, default=True, help="Extract image-analysis markdown files from task workspaces.")
    parser.add_argument("--allow_final_reports", type=parse_bool, default=False, help="Allow final/report markdown files as vision targets.")
    parser.add_argument("--allow_multi_image_workspace_samples", type=parse_bool, default=True, help="Use all task images when an md file is not tied to one image.")
    parser.add_argument("--max_md_files_per_task", type=int, default=4, help="Cap markdown-derived samples per task; 0 means all.")
    parser.add_argument("--min_md_score", type=int, default=5, help="Minimum heuristic score for workspace markdown vision samples.")
    parser.add_argument("--max_prompt_chars", type=int, default=5000)
    parser.add_argument("--max_response_chars", type=int, default=12000)
    parser.add_argument("--min_response_chars", type=int, default=80)
    parser.add_argument("--max_samples", type=int, default=0, help="Cap final samples after dedupe/shuffle; 0 means all.")
    parser.add_argument(
        "--hard_visual_repeat",
        type=int,
        default=2,
        help="Total copies for samples whose task_id matches visual-failure tags in score_file. 1 disables oversampling.",
    )
    parser.add_argument(
        "--hard_visual_tags",
        default="wrong_visual_identity,visual_hallucination,chart_or_numeric_reading,missing_required_visual_detail",
        help="Comma-separated score-file failure tags that should be oversampled.",
    )
    parser.add_argument("--dedupe", type=parse_bool, default=True)
    parser.add_argument("--shuffle", type=parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_text(text: str, limit: int, label: str) -> Tuple[str, bool]:
    if not limit or len(text) <= limit:
        return text, False
    head = max(1, int(limit * 0.65))
    tail = max(1, limit - head)
    omitted = len(text) - head - tail
    marker = f"\n\n[... {omitted} chars omitted from {label} ...]\n\n"
    return text[:head].rstrip() + marker + text[-tail:].lstrip(), True


def parse_path_maps(path_maps: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for item in path_maps or []:
        if "=" not in item:
            continue
        source, target = item.split("=", 1)
        if source:
            parsed.append((source.rstrip("/\\"), target.rstrip("/\\")))
    return parsed


def normalized_parts(path: str) -> List[str]:
    return [part for part in path.replace("\\", "/").split("/") if part]


def tail_from_marker(path: str, marker: str) -> str:
    parts = normalized_parts(path)
    marker_parts = normalized_parts(marker)
    if not marker_parts:
        return ""
    for idx in range(len(parts)):
        if parts[idx : idx + len(marker_parts)] == marker_parts:
            return "/".join(parts[idx:])
    return ""


def build_image_index(search_dirs: Sequence[Path]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    seen_roots: set[Path] = set()
    for root in search_dirs:
        try:
            root = root.resolve()
        except Exception:
            root = root.absolute()
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                index[path.name].append(path.resolve())
    return index


def resolve_path(raw_path: str, project_root: Path, path_maps: Sequence[Tuple[str, str]], image_index: Dict[str, List[Path]]) -> Path | None:
    if not raw_path:
        return None

    raw_path = os.path.expandvars(os.path.expanduser(raw_path.strip()))
    candidates: List[Path] = []

    candidates.append(Path(raw_path))
    for source, target in path_maps:
        if raw_path.startswith(source):
            candidates.append(Path(target + raw_path[len(source) :]))

    for marker in ("work_space", "data/images", "data\\images"):
        tail = tail_from_marker(raw_path, marker)
        if tail:
            candidates.append(project_root / Path(*normalized_parts(tail)))

    raw = Path(raw_path)
    if not raw.is_absolute():
        candidates.append(project_root / raw)

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue

    matches = image_index.get(Path(raw_path).name, [])
    if matches:
        return matches[0]
    return None


def parse_tool_arguments(raw_args: Any) -> Any:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return raw_args
    return raw_args if raw_args is not None else {}


def find_tool_result(messages: List[Dict[str, Any]], start_index: int, tool_name: str, tool_call_id: str | None) -> Dict[str, Any] | None:
    for message in messages[start_index + 1 :]:
        if message.get("role") == "assistant":
            break
        if message.get("role") != "tool":
            continue
        if message.get("name") != tool_name:
            continue
        if tool_call_id and message.get("tool_call_id") and message.get("tool_call_id") != tool_call_id:
            continue
        return message
    return None


def task_id_from_text(text: str) -> int | None:
    for pattern in (r"task_(\d+)", r"\b[QGD](\d+)I\d+\b", r"\b[QGD](\d+)\b"):
        match = re.search(pattern, text.replace("\\", "/"))
        if match:
            return int(match.group(1))
    return None


def load_quiz_index(path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_id: Dict[int, Dict[str, Any]] = {}
    by_image: Dict[str, Dict[str, Any]] = {}
    for record in iter_jsonl(path) or []:
        raw_id = record.get("id")
        try:
            item_id = int(raw_id)
        except Exception:
            continue
        by_id[item_id] = record
        for image_name in record.get("image_url") or []:
            by_image[Path(str(image_name)).name] = record
    return by_id, by_image


def quiz_task_text(record: Dict[str, Any] | None) -> str:
    if not record:
        return ""
    parts = []
    if record.get("caption"):
        parts.append(str(record["caption"]).strip())
    if record.get("body"):
        parts.append(str(record["body"]).strip())
    return "\n\n".join(part for part in parts if part)


def build_user_content(image_paths: Sequence[Path], prompt: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for path in image_paths:
        content.append({"type": "image", "image": os.fspath(path)})
    content.append({"type": "text", "text": prompt})
    return content


def make_sample(
    image_paths: Sequence[Path],
    prompt: str,
    answer: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    image_strings = [os.fspath(path) for path in image_paths]
    return {
        "messages": [
            {"role": "user", "content": build_user_content(image_paths, prompt)},
            {"role": "assistant", "content": answer},
        ],
        "images": image_strings,
        "meta": {
            **meta,
            "image_paths": image_strings,
            "assistant_chars": len(answer),
        },
    }


def trace_prompt(task_prompt: str, max_chars: int) -> str:
    prompt = task_prompt.strip()
    prompt, _ = compact_text(prompt, max_chars, "vision prompt")
    return prompt


def build_trace_samples(
    trace_dir: Path,
    project_root: Path,
    path_maps: Sequence[Tuple[str, str]],
    image_index: Dict[str, List[Path]],
    args: argparse.Namespace,
    stats: Counter[str],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for trace_file in TRACE_FILES:
        for record in iter_jsonl(trace_dir / trace_file) or []:
            messages = record.get("messages")
            if not isinstance(messages, list):
                continue
            for message_index, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    name = function.get("name") or tool_call.get("name") or ""
                    if name != "ask_question_about_image":
                        continue
                    stats["trace_vision_candidates"] += 1
                    arguments = parse_tool_arguments(function.get("arguments", tool_call.get("arguments", {})))
                    if not isinstance(arguments, dict):
                        stats["trace_dropped_bad_arguments"] += 1
                        continue
                    image_path = resolve_path(
                        str(arguments.get("image_path_url") or ""),
                        project_root,
                        path_maps,
                        image_index,
                    )
                    prompt = trace_prompt(str(arguments.get("task_prompt") or ""), args.max_prompt_chars)
                    result_message = find_tool_result(messages, message_index, name, tool_call.get("id"))
                    answer = str((result_message or {}).get("content") or "").strip()
                    answer, changed = compact_text(answer, args.max_response_chars, "vision answer")
                    if changed:
                        stats["trace_compressed_answers"] += 1
                    if not image_path or not prompt:
                        stats["trace_dropped_missing_image_or_prompt"] += 1
                        continue
                    if len(answer) < args.min_response_chars:
                        stats["trace_dropped_short_answer"] += 1
                        continue
                    task_id = task_id_from_text(os.fspath(image_path)) or task_id_from_text(str(record.get("plan_id") or ""))
                    samples.append(
                        make_sample(
                            [image_path],
                            prompt,
                            answer,
                            {
                                "sample_mode": "vision_trace_tool",
                                "source": "actor_trace",
                                "source_tool": "ask_question_about_image",
                                "trace_file": trace_file,
                                "plan_id": record.get("plan_id"),
                                "task_id": task_id,
                                "step": record.get("step"),
                                "created_at": record.get("created_at"),
                            },
                        )
                    )
                    stats["trace_kept"] += 1
    return samples


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def task_dirs(workspace_dir: Path) -> List[Path]:
    dirs = [path for path in workspace_dir.glob("task_*") if path.is_dir()]
    return sorted(dirs, key=lambda path: (task_id_from_text(path.name) is None, task_id_from_text(path.name) or 0, path.name))


def task_images(task_dir: Path, quiz_record: Dict[str, Any] | None, project_root: Path, path_maps: Sequence[Tuple[str, str]], image_index: Dict[str, List[Path]]) -> List[Path]:
    images = sorted(path.resolve() for path in task_dir.iterdir() if is_image_file(path))
    if images:
        return images

    resolved: List[Path] = []
    for image_name in (quiz_record or {}).get("image_url") or []:
        path = resolve_path(str(image_name), project_root, path_maps, image_index)
        if path:
            resolved.append(path)
    return resolved


def is_final_or_report(path: Path) -> bool:
    lower = path.name.lower()
    report_markers = ("final", "report", "markdown_report", "html_report")
    return any(marker in lower for marker in report_markers)


def md_keyword_score(path: Path, text_head: str, image_stems: Sequence[str]) -> int:
    lower_name = path.name.lower()
    lower_head = text_head.lower()
    score = 0
    if any(stem and stem in path.name for stem in image_stems):
        score += 8
    if any(stem and stem in text_head for stem in image_stems):
        score += 3
    for marker in ("image", "vision", "visual", "figure", "chart", "diagram", "ocr"):
        if marker in lower_name:
            score += 4
        if marker in lower_head[:1500]:
            score += 1
    for marker in ("\u56fe\u50cf", "\u56fe\u7247", "\u89c6\u89c9", "\u56fe\u8868", "\u56fe\u7247\u5206\u6790"):
        if marker in path.name:
            score += 5
        if marker in text_head[:1500]:
            score += 1
    for marker in ("search", "source", "evidence_map", "references"):
        if marker in lower_name:
            score -= 3
    if is_final_or_report(path):
        score -= 5
    return score


def selected_md_files(task_dir: Path, image_stems: Sequence[str], allow_final_reports: bool, max_files: int, min_score: int) -> List[Tuple[Path, str, int]]:
    candidates: List[Tuple[Path, str, int]] = []
    for path in task_dir.glob("*.md"):
        if is_final_or_report(path) and not allow_final_reports:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if not text:
            continue
        score = md_keyword_score(path, text[:3000], image_stems)
        if score < min_score:
            continue
        candidates.append((path, text, score))
    candidates.sort(key=lambda item: (-item[2], item[0].name))
    if max_files and max_files > 0:
        candidates = candidates[:max_files]
    return candidates


def images_mentioned_by_md(path: Path, text: str, images: Sequence[Path], allow_multi: bool) -> List[Path]:
    name_mentions = [image for image in images if image.stem in path.name or image.name in path.name]
    if name_mentions:
        return name_mentions

    mentioned = []
    haystack = text[:4000]
    for image in images:
        if image.stem in haystack or image.name in haystack:
            mentioned.append(image)
    if mentioned:
        return mentioned
    if len(images) == 1:
        return [images[0]]
    return list(images) if allow_multi else []


def workspace_prompt(quiz_record: Dict[str, Any] | None, image_paths: Sequence[Path], max_chars: int) -> str:
    task_text = quiz_task_text(quiz_record)
    task_text, _ = compact_text(task_text, max_chars, "original task")
    image_names = ", ".join(path.name for path in image_paths)
    grounding = (
        "Analyze the attached visual evidence before doing any external research. "
        "Ground the answer in visible details: subject identity, labels/OCR, chart axes, legends, numbers, "
        "spatial relations, visual style, and uncertainties. State clearly when a detail is not visible."
    )
    if task_text:
        return f"{grounding}\n\nImage file(s): {image_names}\n\nOriginal Co-Sight task:\n{task_text}"
    return f"{grounding}\n\nImage file(s): {image_names}"


def build_workspace_samples(
    workspace_dir: Path,
    quiz_by_id: Dict[int, Dict[str, Any]],
    project_root: Path,
    path_maps: Sequence[Tuple[str, str]],
    image_index: Dict[str, List[Path]],
    args: argparse.Namespace,
    stats: Counter[str],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if not workspace_dir.exists():
        stats["workspace_missing"] += 1
        return samples

    for task_dir in task_dirs(workspace_dir):
        task_id = task_id_from_text(task_dir.name)
        quiz_record = quiz_by_id.get(task_id) if task_id is not None else None
        images = task_images(task_dir, quiz_record, project_root, path_maps, image_index)
        if not images:
            stats["workspace_dropped_no_images"] += 1
            continue

        image_stems = [image.stem for image in images]
        md_files = selected_md_files(task_dir, image_stems, args.allow_final_reports, args.max_md_files_per_task, args.min_md_score)
        if not md_files:
            stats["workspace_dropped_no_md"] += 1
            continue

        for md_path, text, score in md_files:
            selected_images = images_mentioned_by_md(md_path, text, images, args.allow_multi_image_workspace_samples)
            if not selected_images:
                stats["workspace_dropped_no_image_match"] += 1
                continue
            answer, changed = compact_text(text, args.max_response_chars, "workspace vision answer")
            if changed:
                stats["workspace_compressed_answers"] += 1
            if len(answer) < args.min_response_chars:
                stats["workspace_dropped_short_answer"] += 1
                continue
            prompt = workspace_prompt(quiz_record, selected_images, args.max_prompt_chars)
            samples.append(
                make_sample(
                    selected_images,
                    prompt,
                    answer,
                    {
                        "sample_mode": "vision_workspace_md",
                        "source": "strong_workspace_md",
                        "task_id": task_id,
                        "source_md": os.fspath(md_path.resolve()),
                        "md_score": score,
                    },
                )
            )
            stats["workspace_kept"] += 1
    return samples


def dedupe_samples(samples: Sequence[Dict[str, Any]], stats: Counter[str]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    output: List[Dict[str, Any]] = []
    for sample in samples:
        images = tuple(sample.get("images") or [])
        messages = sample.get("messages") or []
        prompt = ""
        answer = ""
        if messages:
            prompt = json.dumps(messages[0].get("content", ""), ensure_ascii=False, sort_keys=True)
            answer = str(messages[-1].get("content", ""))
        key = json.dumps([images, prompt, answer], ensure_ascii=False)
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        output.append(sample)
    return output


def score_failure_tags(reason: str) -> List[str]:
    lower = reason.lower()
    tags = []
    if any(token in lower for token in ("wrong visual", "misinterpret", "incorrectly identif", "false premise")):
        tags.append("wrong_visual_identity")
    if any(token in lower for token in ("hallucinat", "fabricat", "not supported", "false presence")):
        tags.append("visual_hallucination")
    if any(token in lower for token in ("data", "chart", "numerical", "number", "axis", "legend")):
        tags.append("chart_or_numeric_reading")
    if any(token in lower for token in ("omission", "fails to provide", "missing", "incomplete")):
        tags.append("missing_required_visual_detail")
    if "json parse error" in lower:
        tags.append("json_parse_error")
    return tags


def load_score_failure_tasks(score_file: Path, allowed_tags: Sequence[str], stats: Counter[str]) -> Dict[int, List[str]]:
    allowed = set(allowed_tags)
    failures: Dict[int, List[str]] = {}
    if not score_file.exists():
        stats["score_file_missing"] += 1
        return failures

    for record in iter_jsonl(score_file) or []:
        reason = str((record.get("accuracy_detail") or {}).get("reason") or "")
        matched_tags = sorted(set(score_failure_tags(reason)) & allowed)
        if not matched_tags:
            continue
        raw_id = record.get("qid") or record.get("id") or record.get("task_id") or record.get("report_path") or ""
        task_id = task_id_from_text(str(raw_id))
        if task_id is None:
            stats["score_visual_failure_missing_task_id"] += 1
            continue
        failures[task_id] = matched_tags
        stats["score_visual_failure_tasks"] += 1
        for tag in matched_tags:
            stats[f"score_visual_failure_tag:{tag}"] += 1
    return failures


def add_score_file_stats(score_file: Path, stats: Counter[str]) -> None:
    if not score_file.exists():
        stats["score_file_missing"] += 1
        return
    for record in iter_jsonl(score_file) or []:
        reason = str((record.get("accuracy_detail") or {}).get("reason") or "")
        for tag in score_failure_tags(reason):
            stats[f"score_tag:{tag}"] += 1


def sample_task_id(sample: Dict[str, Any]) -> int | None:
    raw_id = (sample.get("meta") or {}).get("task_id")
    try:
        return int(raw_id)
    except Exception:
        return None


def repeat_hard_visual_samples(
    samples: Sequence[Dict[str, Any]],
    failure_tasks: Dict[int, List[str]],
    repeat: int,
    stats: Counter[str],
) -> List[Dict[str, Any]]:
    repeat = max(1, repeat)
    if repeat <= 1 or not failure_tasks:
        return list(samples)

    output: List[Dict[str, Any]] = []
    for sample in samples:
        task_id = sample_task_id(sample)
        tags = failure_tasks.get(task_id) if task_id is not None else None
        copies = repeat if tags else 1
        if tags:
            stats["hard_visual_repeat_matched_samples"] += 1
            stats[f"hard_visual_repeat_factor:{repeat}"] += 1
        for copy_index in range(copies):
            copied = sample if copy_index == 0 else copy.deepcopy(sample)
            if tags:
                copied.setdefault("meta", {})
                copied["meta"]["hard_visual_failure"] = True
                copied["meta"]["hard_visual_failure_tags"] = tags
                copied["meta"]["hard_visual_repeat_index"] = copy_index
            output.append(copied)

    stats["hard_visual_repeat_added_samples"] = len(output) - len(samples)
    return output


def render_preview(sample: Dict[str, Any]) -> str:
    chunks = []
    for message in sample.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "image":
                    parts.append(f"<image>{item.get('image')}</image>")
                elif item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            content = "\n".join(parts)
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(chunks)[:2000]


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    trace_dir = Path(args.trace_dir)
    workspace_dir = Path(args.workspace_dir)
    output_dir = Path(args.output_dir)
    output_file = Path(args.output_file) if args.output_file else output_dir / "vision_sft_messages.jsonl"
    stats_path = output_dir / "stats.json"
    preview_path = output_dir / "vision_text_preview.jsonl"

    search_dirs = [Path(args.image_base_dir), workspace_dir]
    search_dirs.extend(Path(path) for path in args.image_search_dir or [])
    path_maps = parse_path_maps(args.path_map)
    image_index = build_image_index(search_dirs)

    stats: Counter[str] = Counter()
    stats["indexed_image_basenames"] = len(image_index)
    stats["indexed_image_paths"] = sum(len(paths) for paths in image_index.values())

    quiz_by_id, _quiz_by_image = load_quiz_index(Path(args.quiz_file))
    stats["quiz_records"] = len(quiz_by_id)
    add_score_file_stats(Path(args.score_file), stats)
    hard_visual_tags = [item.strip() for item in args.hard_visual_tags.split(",") if item.strip()]
    hard_visual_failure_tasks = load_score_failure_tasks(Path(args.score_file), hard_visual_tags, stats)
    stats["hard_visual_allowed_tags"] = len(hard_visual_tags)
    stats["hard_visual_failure_task_ids"] = len(hard_visual_failure_tasks)

    samples: List[Dict[str, Any]] = []
    if args.include_trace_tool_results:
        samples.extend(build_trace_samples(trace_dir, project_root, path_maps, image_index, args, stats))
    if args.include_workspace_md:
        samples.extend(build_workspace_samples(workspace_dir, quiz_by_id, project_root, path_maps, image_index, args, stats))

    stats["samples_before_dedupe"] = len(samples)
    if args.dedupe:
        samples = dedupe_samples(samples, stats)
    samples = repeat_hard_visual_samples(samples, hard_visual_failure_tasks, args.hard_visual_repeat, stats)
    if args.shuffle:
        random.Random(args.seed).shuffle(samples)
    if args.max_samples and args.max_samples > 0 and len(samples) > args.max_samples:
        stats["dropped_max_samples"] = len(samples) - args.max_samples
        samples = samples[: args.max_samples]

    stats["samples"] = len(samples)
    for sample in samples:
        stats[f"sample_mode:{sample.get('meta', {}).get('sample_mode')}"] += 1

    write_jsonl(output_file, samples)
    write_jsonl(preview_path, ({"text": render_preview(sample), "meta": sample.get("meta", {})} for sample in samples[:200]))
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(dict(stats), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_file": os.fspath(output_file),
                "preview_file": os.fspath(preview_path),
                "stats_file": os.fspath(stats_path),
                **dict(stats),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
