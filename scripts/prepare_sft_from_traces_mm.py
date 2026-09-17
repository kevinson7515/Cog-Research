#!/usr/bin/env python3
"""Convert Co-Sight planner/actor traces into SFT jsonl files.

The default output trains short state-action samples instead of full long
trajectories:
  context messages -> next assistant tool call

This matches the runtime decision the actor/planner must make, keeps samples
well below the 16k training limit, and avoids teaching long pre-tool prose.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


VALID_ROLES = {"system", "user", "assistant", "tool"}
IMAGE_BASENAME_CACHE: Dict[Tuple[str, str], List[str]] = {}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_dir", default="trace", help="Directory containing planner_trace.jsonl and actor_trace.jsonl.")
    parser.add_argument("--output_dir", default="data/sft_traces", help="Directory for generated SFT jsonl files.")
    parser.add_argument("--include", default="planner,actor", help="Comma-separated trace types to include: planner,actor.")
    parser.add_argument("--sample_mode", choices=["turn", "full", "both"], default="turn", help="Build short action samples, full conversations, or both.")
    parser.add_argument("--context_messages", type=int, default=8, help="Recent non-system messages kept before each target assistant action.")
    parser.add_argument("--max_tool_result_chars", type=int, default=4000, help="Compress long tool/user result messages in context; 0 disables.")
    parser.add_argument("--max_arg_chars", type=int, default=6000, help="Compress very long tool-call string arguments; 0 disables.")
    parser.add_argument("--strip_tool_preamble", type=parse_bool, default=True, help="Remove assistant prose before structured tool_calls.")
    parser.add_argument("--tool_only", type=parse_bool, default=True, help="In turn mode, keep only assistant turns that call tools.")
    parser.add_argument("--include_workflow_sft", type=parse_bool, default=True, help="Add weighted workflow-transition samples for saving intermediate files and final reporting.")
    parser.add_argument("--workflow_context_messages", type=int, default=12, help="Recent non-system messages kept before workflow targets.")
    parser.add_argument("--workflow_max_samples", type=int, default=6000, help="Cap workflow-transition samples before weighting; 0 disables.")
    parser.add_argument("--workflow_file_saver_weight", type=int, default=3, help="Training repeat weight for workflow file_saver targets.")
    parser.add_argument("--workflow_report_weight", type=int, default=2, help="Training repeat weight for workflow generate_markdown_report targets.")
    parser.add_argument("--workflow_mark_step_weight", type=int, default=2, help="Training repeat weight for workflow mark_step targets after save/report.")
    parser.add_argument("--include_vision_sft", type=parse_bool, default=True, help="Also build image+prompt -> vision result samples from ask_question_about_image tool traces.")
    parser.add_argument("--vision_max_samples", type=int, default=384, help="Cap direct image+prompt -> answer samples; 0 disables.")
    parser.add_argument("--vision_sample_ratio", type=float, default=0.02, help="Cap direct vision samples to this fraction of non-vision samples; 0 disables.")
    parser.add_argument("--vision_max_response_chars", type=int, default=8000, help="Compress long vision tool results; 0 disables.")
    parser.add_argument("--vision_min_response_chars", type=int, default=20, help="Drop vision samples with shorter tool results.")
    parser.add_argument("--vision_path_map", action="append", default=[], help="Optional FROM=TO path rewrite for image paths. Can be repeated.")
    parser.add_argument("--vision_image_search_dir", action="append", default=[], help="Fallback directories to search by image basename when traced path is unavailable.")
    parser.add_argument("--vision_fallback_to_md", type=parse_bool, default=True, help="For empty vision tool results, try a markdown file in the task directory.")
    parser.add_argument("--max_chars", type=int, default=0, help="Drop rendered samples whose text exceeds this many chars; 0 disables.")
    parser.add_argument("--min_assistant_chars", type=int, default=1, help="Drop samples with less assistant target text than this.")
    parser.add_argument("--dedupe", action="store_true", help="Drop exact duplicate rendered conversations.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output records.")
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


def load_trace_records(trace_dir: Path, include: set[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    sources = {
        "planner": trace_dir / "planner_trace.jsonl",
        "actor": trace_dir / "actor_trace.jsonl",
    }
    for trace_type, path in sources.items():
        if trace_type not in include:
            continue
        for record in iter_jsonl(path) or []:
            record.setdefault("trace_type", trace_type)
            records.append(record)
    return records


def parse_arguments(raw_args: Any) -> Any:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return raw_args
    return raw_args if raw_args is not None else {}


def dump_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)


def compact_text(text: str, limit: int, label: str = "content") -> Tuple[str, bool]:
    if not limit or len(text) <= limit:
        return text, False
    head = max(1, int(limit * 0.6))
    tail = max(1, limit - head)
    omitted = len(text) - head - tail
    marker = f"\n\n[... {omitted} chars omitted from {label} ...]\n\n"
    return text[:head].rstrip() + marker + text[-tail:].lstrip(), True


def normalize_content_for_text(content: Any) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"image", "image_url"}:
                image_value = item.get("image") or item.get("image_url") or ""
                if isinstance(image_value, dict):
                    image_value = image_value.get("url", "")
                parts.append(f"<image>{image_value}</image>")
            elif item_type == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "")


def shrink_arguments(arguments: Any, limit: int) -> Tuple[Any, bool]:
    if not limit:
        return arguments, False
    changed = False
    if isinstance(arguments, str):
        return compact_text(arguments, limit, "tool arguments")
    if isinstance(arguments, list):
        result = []
        for item in arguments:
            new_item, item_changed = shrink_arguments(item, limit)
            result.append(new_item)
            changed = changed or item_changed
        return result, changed
    if isinstance(arguments, dict):
        result = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                result[key], item_changed = compact_text(value, limit, f"argument {key}")
            else:
                result[key], item_changed = shrink_arguments(value, limit)
            changed = changed or item_changed
        return result, changed
    return arguments, False


def normalize_tool_call(tool_call: Dict[str, Any], args: argparse.Namespace, stats: Dict[str, int]) -> Dict[str, Any]:
    function = tool_call.get("function") or {}
    name = function.get("name") or tool_call.get("name") or ""
    arguments = parse_arguments(function.get("arguments", tool_call.get("arguments", {})))
    arguments, changed = shrink_arguments(arguments, args.max_arg_chars)
    if changed:
        stats["compressed_tool_arguments"] += 1
    normalized = {
        "id": tool_call.get("id") or f"call_{stats['tool_calls_total']}",
        "type": tool_call.get("type") or "function",
        "function": {
            "name": str(name),
            "arguments": dump_arguments(arguments),
        },
    }
    stats["tool_calls_total"] += 1
    stats[f"tool_call:{name}"] += 1
    return normalized


def render_tool_call(tool_call: Dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    name = function.get("name") or tool_call.get("name") or ""
    payload = {
        "name": str(name),
        "arguments": parse_arguments(function.get("arguments", {})),
    }
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def message_text(message: Dict[str, Any]) -> str:
    parts: List[str] = []
    content = message.get("content")
    if content:
        parts.append(normalize_content_for_text(content).strip())
    for tool_call in message.get("tool_calls") or []:
        parts.append(render_tool_call(tool_call))
    return "\n\n".join(part for part in parts if part).strip()


def normalize_message(message: Dict[str, Any], args: argparse.Namespace, stats: Dict[str, int]) -> Dict[str, Any]:
    role = message.get("role")
    if role not in VALID_ROLES:
        raise ValueError(f"Unsupported role in trace: {role!r}")

    normalized: Dict[str, Any] = {"role": role, "content": str(message.get("content") or "")}

    if role == "assistant":
        tool_calls = [
            normalize_tool_call(tool_call, args, stats)
            for tool_call in (message.get("tool_calls") or [])
            if isinstance(tool_call, dict)
        ]
        if tool_calls:
            normalized["tool_calls"] = tool_calls
            if args.strip_tool_preamble and normalized["content"]:
                stats["stripped_tool_preamble"] += 1
                stats["stripped_tool_preamble_chars"] += len(normalized["content"])
                normalized["content"] = ""
        elif message.get("reasoning_content"):
            normalized["content"] = "\n\n".join(
                part for part in [str(message.get("reasoning_content") or "").strip(), normalized["content"].strip()] if part
            )
    elif role == "tool":
        for key in ("name", "tool_call_id"):
            if key in message:
                normalized[key] = message[key]

    return normalized


def compress_context_message(message: Dict[str, Any], args: argparse.Namespace, stats: Dict[str, int]) -> Dict[str, Any]:
    copied = copy.deepcopy(message)
    content = copied.get("content")
    if isinstance(content, str) and args.max_tool_result_chars and copied.get("role") in {"tool", "user"}:
        copied["content"], changed = compact_text(content, args.max_tool_result_chars, f"{copied.get('role')} message")
        if changed:
            stats["compressed_context_messages"] += 1
    return copied


def normalize_record_messages(record: Dict[str, Any], args: argparse.Namespace, stats: Dict[str, int]) -> List[Dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    normalized_messages: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized = normalize_message(message, args, stats)
        if message_text(normalized) or normalized.get("role") == "assistant":
            normalized_messages.append(normalized)
    return normalized_messages


def assistant_chars(messages: List[Dict[str, Any]]) -> int:
    return sum(len(message_text(message)) for message in messages if message.get("role") == "assistant")


def parse_path_maps(path_maps: List[str]) -> List[Tuple[str, str]]:
    parsed = []
    for item in path_maps or []:
        if "=" not in item:
            continue
        source, target = item.split("=", 1)
        if source:
            parsed.append((source, target))
    return parsed


def normalized_path_parts(path: str) -> List[str]:
    return [part for part in path.replace("\\", "/").split("/") if part]


def useful_path_tail(raw_path: str) -> List[str]:
    parts = normalized_path_parts(raw_path)
    for marker in ("work_space", "data", "images"):
        if marker in parts:
            return parts[parts.index(marker):]
    return parts[-3:]


def parts_endwith(parts: List[str], suffix: List[str]) -> bool:
    if not suffix or len(parts) < len(suffix):
        return False
    return parts[-len(suffix):] == suffix


def resolve_image_path(raw_path: str, args: argparse.Namespace, stats: Dict[str, int]) -> str | None:
    if not raw_path:
        stats["vision_dropped_missing_image_path"] += 1
        return None

    candidates = [raw_path]
    for source, target in parse_path_maps(args.vision_path_map):
        if raw_path.startswith(source):
            candidates.append(target + raw_path[len(source):])

    basename = os.path.basename(raw_path)
    raw_tail = useful_path_tail(raw_path)
    basename_matches: List[str] = []
    for search_dir in args.vision_image_search_dir or []:
        if not search_dir:
            continue
        candidates.append(os.path.join(search_dir, basename))
        try:
            cache_key = (os.path.abspath(os.fspath(search_dir)), basename)
            if cache_key not in IMAGE_BASENAME_CACHE:
                IMAGE_BASENAME_CACHE[cache_key] = [os.fspath(path) for path in Path(search_dir).rglob(basename)]
            matches = [Path(path) for path in IMAGE_BASENAME_CACHE[cache_key]]
            suffix_matches = [
                os.fspath(match)
                for match in matches
                if parts_endwith(normalized_path_parts(os.fspath(match)), raw_tail)
            ]
            candidates.extend(suffix_matches[:3])
            basename_matches.extend(os.fspath(match) for match in matches[:3])
        except Exception:
            pass
    candidates.extend(basename_matches)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)

    stats["vision_dropped_image_not_found"] += 1
    return None


def find_tool_result(messages: List[Dict[str, Any]], start_index: int, tool_name: str, tool_call_id: str | None) -> Dict[str, Any] | None:
    for message in messages[start_index + 1:]:
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


def markdown_fallback_for_vision(image_path: str, limit: int) -> str:
    task_dir = Path(image_path).parent
    if not task_dir.exists():
        return ""
    image_stem = Path(image_path).stem
    preferred = []
    fallback = []
    for path in task_dir.glob("*.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        score_text = f"{path.name}\n{text[:2000]}"
        if image_stem in score_text or "图像" in score_text or "视觉" in score_text or "image" in score_text.lower():
            preferred.append(text)
        else:
            fallback.append(text)
    for text in preferred + fallback:
        text, _ = compact_text(text, limit, "markdown vision fallback")
        if text.strip():
            return text.strip()
    return ""


def build_vision_items(records: List[Dict[str, Any]], args: argparse.Namespace, stats: Dict[str, int]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for record in records:
        if record.get("trace_type") != "actor":
            continue
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

                stats["vision_candidate_tool_calls"] += 1
                arguments = parse_arguments(function.get("arguments", tool_call.get("arguments", {})))
                if not isinstance(arguments, dict):
                    stats["vision_dropped_bad_arguments"] += 1
                    continue
                image_path = resolve_image_path(str(arguments.get("image_path_url") or ""), args, stats)
                task_prompt = str(arguments.get("task_prompt") or "").strip()
                if not image_path or not task_prompt:
                    stats["vision_dropped_missing_prompt_or_image"] += 1
                    continue

                result_message = find_tool_result(messages, message_index, name, tool_call.get("id"))
                result = str((result_message or {}).get("content") or "").strip()
                if not result and args.vision_fallback_to_md:
                    result = markdown_fallback_for_vision(image_path, args.vision_max_response_chars)
                    if result:
                        stats["vision_used_markdown_fallback"] += 1

                if args.vision_max_response_chars:
                    result, changed = compact_text(result, args.vision_max_response_chars, "vision tool result")
                    if changed:
                        stats["vision_compressed_results"] += 1
                if len(result) < args.vision_min_response_chars:
                    stats["vision_dropped_short_result"] += 1
                    continue

                items.append({
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": task_prompt},
                            ],
                        },
                        {"role": "assistant", "content": result},
                    ],
                    "images": [image_path],
                    "meta": {
                        "plan_id": record.get("plan_id"),
                        "trace_type": record.get("trace_type"),
                        "step": record.get("step"),
                        "created_at": record.get("created_at"),
                        "sample_mode": "vision_tool",
                        "source_tool": "ask_question_about_image",
                        "image_path": image_path,
                        "assistant_chars": len(result),
                    },
                })
    return items


def tool_call_names_for_message(message: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name") or ""
        if name:
            names.append(str(name))
    return names


def recent_tool_names(messages: List[Dict[str, Any]], target_index: int, lookback: int) -> List[str]:
    start = max(0, target_index - lookback)
    names: List[str] = []
    for message in messages[start:target_index]:
        if message.get("role") == "assistant":
            names.extend(tool_call_names_for_message(message))
        elif message.get("role") == "tool" and message.get("name"):
            names.append(str(message.get("name")))
    return names


def workflow_mode_and_weight(tool_names: List[str], recent_names: List[str], args: argparse.Namespace) -> Tuple[str, int]:
    recent = set(recent_names)
    target = set(tool_names)
    research_tools = {
        "serper_search",
        "fetch_website_content",
        "fetch_website_content_with_images",
        "fetch_website_images_only",
        "image_search",
        "extract_document_content",
        "file_read",
        "execute_code",
    }

    if "file_saver" in target:
        if "ask_question_about_image" in recent:
            return "workflow_image_to_file_saver", max(1, args.workflow_file_saver_weight)
        if recent.intersection(research_tools):
            return "workflow_research_to_file_saver", max(1, args.workflow_file_saver_weight)
        return "workflow_file_saver", max(1, args.workflow_file_saver_weight)
    if "generate_markdown_report" in target:
        return "workflow_generate_report", max(1, args.workflow_report_weight)
    if "mark_step" in target and recent.intersection({"file_saver", "generate_markdown_report"}):
        return "workflow_mark_after_save_or_report", max(1, args.workflow_mark_step_weight)
    return "", 1


def trigger_tools_for_workflow(sample_mode: str) -> set[str]:
    if sample_mode == "workflow_image_to_file_saver":
        return {"ask_question_about_image"}
    if sample_mode == "workflow_research_to_file_saver":
        return {
            "serper_search",
            "fetch_website_content",
            "fetch_website_content_with_images",
            "fetch_website_images_only",
            "image_search",
            "extract_document_content",
            "file_read",
            "execute_code",
        }
    if sample_mode == "workflow_generate_report":
        return {"file_saver", "file_read"}
    if sample_mode == "workflow_mark_after_save_or_report":
        return {"file_saver", "generate_markdown_report"}
    return set()


def add_recent_tool_pair_indices(
    indices: set[int],
    messages: List[Dict[str, Any]],
    target_index: int,
    tool_names: set[str],
    max_pairs: int = 2,
) -> None:
    if not tool_names:
        return
    added_pairs = 0
    for idx in range(target_index - 1, -1, -1):
        message = messages[idx]
        if message.get("role") == "tool" and message.get("name") in tool_names:
            indices.add(idx)
            if idx > 0 and messages[idx - 1].get("role") == "assistant":
                indices.add(idx - 1)
            added_pairs += 1
        elif message.get("role") == "assistant" and set(tool_call_names_for_message(message)).intersection(tool_names):
            indices.add(idx)
            if idx + 1 < target_index and messages[idx + 1].get("role") == "tool":
                indices.add(idx + 1)
            added_pairs += 1
        if added_pairs >= max_pairs:
            break


def build_workflow_items(record: Dict[str, Any], messages: List[Dict[str, Any]], args: argparse.Namespace, stats: Dict[str, int]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for target_index, target in enumerate(messages):
        if target.get("role") != "assistant":
            continue
        tool_names = tool_call_names_for_message(target)
        if not tool_names:
            continue
        recent_names = recent_tool_names(messages, target_index, max(args.workflow_context_messages + 6, 12))
        sample_mode, sample_weight = workflow_mode_and_weight(tool_names, recent_names, args)
        if not sample_mode:
            continue

        context_indices = set(selected_context_indices(messages, target_index, args.workflow_context_messages))
        add_recent_tool_pair_indices(
            context_indices,
            messages,
            target_index,
            trigger_tools_for_workflow(sample_mode),
        )
        context = [
            compress_context_message(messages[idx], args, stats)
            for idx in sorted(context_indices)
        ]
        if not context:
            stats["workflow_dropped_empty_context"] += 1
            continue

        target_copy = copy.deepcopy(target)
        chars = assistant_chars([target_copy])
        if chars < args.min_assistant_chars:
            stats["workflow_dropped_short_assistant"] += 1
            continue

        items.append({
            "messages": context + [target_copy],
            "meta": {
                "plan_id": record.get("plan_id"),
                "trace_type": record.get("trace_type"),
                "step": record.get("step"),
                "created_at": record.get("created_at"),
                "sample_mode": sample_mode,
                "target_tool_names": tool_names,
                "recent_tool_names": recent_names[-8:],
                "assistant_chars": chars,
                "sample_weight": sample_weight,
            },
        })
        stats[f"workflow_candidate:{sample_mode}"] += 1
    return items


def limit_sampled_items(
    items: List[Dict[str, Any]],
    max_samples: int,
    sample_ratio: float,
    reference_count: int,
    seed: int,
    stats: Dict[str, int],
    prefix: str,
) -> List[Dict[str, Any]]:
    if not items:
        return items
    limit = len(items)
    if sample_ratio and sample_ratio > 0 and reference_count > 0:
        limit = min(limit, max(1, int(reference_count * sample_ratio)))
    if max_samples and max_samples > 0:
        limit = min(limit, max_samples)
    if limit >= len(items):
        return items
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    keep = set(indices[:limit])
    stats[f"{prefix}_sampled_from"] = len(items)
    stats[f"{prefix}_sampled_to"] = limit
    return [item for idx, item in enumerate(items) if idx in keep]


def render_plain_chat(messages: List[Dict[str, Any]]) -> str:
    chunks = []
    for message in messages:
        role = message["role"]
        content = message_text(message)
        if role == "tool" and message.get("name"):
            role = f"tool name={message['name']}"
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(chunks)


def selected_context_indices(messages: List[Dict[str, Any]], target_index: int, context_messages: int) -> List[int]:
    indices = set()
    for idx, message in enumerate(messages[:target_index]):
        if message.get("role") == "system":
            indices.add(idx)

    for idx, message in enumerate(messages[:target_index]):
        if message.get("role") == "user":
            indices.add(idx)
            break

    start = max(0, target_index - context_messages)
    for idx in range(start, target_index):
        indices.add(idx)
        if messages[idx].get("role") == "tool" and idx > 0:
            indices.add(idx - 1)

    return sorted(indices)


def build_full_item(record: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    chars = assistant_chars(messages)
    if chars <= 0:
        return None
    return {
        "messages": messages,
        "meta": {
            "plan_id": record.get("plan_id"),
            "trace_type": record.get("trace_type"),
            "step": record.get("step"),
            "created_at": record.get("created_at"),
            "sample_mode": "full",
            "assistant_chars": chars,
        },
    }


def build_turn_items(record: Dict[str, Any], messages: List[Dict[str, Any]], args: argparse.Namespace, stats: Dict[str, int]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for target_index, target in enumerate(messages):
        if target.get("role") != "assistant":
            continue
        target_tool_calls = target.get("tool_calls") or []
        if args.tool_only and not target_tool_calls:
            stats["dropped_non_tool_assistant_turn"] += 1
            continue

        context = [
            compress_context_message(messages[idx], args, stats)
            for idx in selected_context_indices(messages, target_index, args.context_messages)
        ]
        if not context:
            stats["dropped_empty_context"] += 1
            continue

        target_copy = copy.deepcopy(target)
        item_messages = context + [target_copy]
        chars = assistant_chars([target_copy])
        if chars < args.min_assistant_chars:
            stats["dropped_short_assistant"] += 1
            continue

        tool_names = [
            (tool_call.get("function") or {}).get("name", "")
            for tool_call in target_tool_calls
        ]
        items.append({
            "messages": item_messages,
            "meta": {
                "plan_id": record.get("plan_id"),
                "trace_type": record.get("trace_type"),
                "step": record.get("step"),
                "created_at": record.get("created_at"),
                "sample_mode": "turn",
                "target_tool_names": tool_names,
                "assistant_chars": chars,
            },
        })
    return items


def keep_item(item: Dict[str, Any], args: argparse.Namespace, stats: Dict[str, int], seen: set[str]) -> bool:
    if item["meta"]["assistant_chars"] < args.min_assistant_chars:
        stats["dropped_short_assistant"] += 1
        return False
    rendered = render_plain_chat(item["messages"])
    if args.max_chars and len(rendered) > args.max_chars:
        stats["dropped_too_long"] += 1
        return False
    if args.dedupe and rendered in seen:
        stats["dropped_duplicate"] += 1
        return False
    seen.add(rendered)
    item["text_preview"] = rendered[:1000]
    return True


def append_weighted_item(output: List[Dict[str, Any]], item: Dict[str, Any], stats: Dict[str, int]) -> int:
    """Append an item, expanding workflow sample weights into real JSONL rows."""
    sample_mode = str(item.get("meta", {}).get("sample_mode", ""))
    repeat = 1
    if sample_mode.startswith("workflow_"):
        try:
            repeat = max(1, int(item.get("meta", {}).get("sample_weight", 1)))
        except Exception:
            repeat = 1

    for repeat_index in range(repeat):
        if repeat_index == 0:
            output.append(item)
            continue
        copied = copy.deepcopy(item)
        copied.setdefault("meta", {})["repeat_index"] = repeat_index
        output.append(copied)

    stats[f"kept_{sample_mode}"] += repeat
    stats[f"kept_{sample_mode}_unique"] += 1
    stats[f"kept_trace:{item['meta'].get('trace_type')}"] += repeat
    if repeat > 1:
        stats["weighted_extra_rows"] += repeat - 1
    return repeat


def build_dataset(records: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    output: List[Dict[str, Any]] = []
    stats: Counter[str] = Counter()
    stats["input_records"] = len(records)
    seen: set[str] = set()

    for record in records:
        messages = normalize_record_messages(record, args, stats)
        if not messages:
            stats["dropped_empty"] += 1
            continue

        candidates: List[Dict[str, Any]] = []
        if args.include_workflow_sft:
            candidates.extend(build_workflow_items(record, messages, args, stats))
        if args.sample_mode in {"full", "both"}:
            full_item = build_full_item(record, messages)
            if full_item is not None:
                candidates.append(full_item)
        if args.sample_mode in {"turn", "both"}:
            candidates.extend(build_turn_items(record, messages, args, stats))

        for item in candidates:
            sample_mode = str(item.get("meta", {}).get("sample_mode", ""))
            if (
                sample_mode.startswith("workflow_")
                and args.workflow_max_samples
                and stats["kept_workflow_unique"] >= args.workflow_max_samples
            ):
                stats["workflow_dropped_max_samples"] += 1
                continue
            if keep_item(item, args, stats, seen):
                added = append_weighted_item(output, item, stats)
                if sample_mode.startswith("workflow_"):
                    stats["kept_workflow_unique"] += 1
                    stats["kept_workflow_total"] += added

    if args.include_vision_sft:
        vision_items = build_vision_items(records, args, stats)
        vision_items = limit_sampled_items(
            vision_items,
            args.vision_max_samples,
            args.vision_sample_ratio,
            len(output),
            args.seed + 17,
            stats,
            "vision",
        )
        for item in vision_items:
            if keep_item(item, args, stats, seen):
                output.append(item)
                stats["kept_vision_tool"] += 1
                stats["kept_vision_tool_unique"] += 1
                stats[f"kept_trace:{item['meta'].get('trace_type')}"] += 1

    if args.shuffle:
        random.Random(args.seed).shuffle(output)
    stats["kept"] = len(output)
    return output, dict(stats)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    include = {item.strip() for item in args.include.split(",") if item.strip()}

    raw_records = load_trace_records(trace_dir, include)
    records, stats = build_dataset(raw_records, args)

    messages_path = output_dir / "sft_messages.jsonl"
    vision_messages_path = output_dir / "vision_sft_messages.jsonl"
    workflow_messages_path = output_dir / "workflow_sft_messages.jsonl"
    text_path = output_dir / "sft_text.jsonl"
    stats_path = output_dir / "stats.json"

    write_jsonl(messages_path, records)
    write_jsonl(
        vision_messages_path,
        (item for item in records if item.get("meta", {}).get("sample_mode") == "vision_tool"),
    )
    write_jsonl(
        workflow_messages_path,
        (item for item in records if str(item.get("meta", {}).get("sample_mode", "")).startswith("workflow_")),
    )
    write_jsonl(
        text_path,
        ({"text": render_plain_chat(item["messages"]), "meta": item["meta"]} for item in records),
    )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "messages_path": os.fspath(messages_path),
        "vision_messages_path": os.fspath(vision_messages_path),
        "workflow_messages_path": os.fspath(workflow_messages_path),
        "text_path": os.fspath(text_path),
        "stats_path": os.fspath(stats_path),
        **stats,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
