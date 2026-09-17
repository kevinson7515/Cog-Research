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


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_dir", default="trace", help="Directory containing planner_trace.jsonl and actor_trace.jsonl.")
    parser.add_argument("--output_dir", default="data/sft_traces", help="Directory for generated SFT jsonl files.")
    parser.add_argument("--include", default="planner,actor", help="Comma-separated trace types to include: planner,actor.")
    parser.add_argument(
        "--sample_mode",
        choices=["turn", "full", "both", "workflow", "all"],
        default="turn",
        help="Build short action samples, full conversations, workflow-chain samples, or all.",
    )
    parser.add_argument("--context_messages", type=int, default=8, help="Recent non-system messages kept before each target assistant action.")
    parser.add_argument("--max_tool_result_chars", type=int, default=4000, help="Compress long tool/user result messages in context; 0 disables.")
    parser.add_argument("--max_arg_chars", type=int, default=6000, help="Compress very long tool-call string arguments; 0 disables.")
    parser.add_argument("--strip_tool_preamble", type=parse_bool, default=True, help="Remove assistant prose before structured tool_calls.")
    parser.add_argument("--tool_only", type=parse_bool, default=True, help="In turn mode, keep only assistant turns that call tools.")
    parser.add_argument("--add_workflow_samples", type=parse_bool, default=False, help="Add multi-turn samples that teach evidence -> file_saver/report -> mark_step order.")
    parser.add_argument("--workflow_context_messages", type=int, default=14, help="Recent messages kept before each workflow anchor.")
    parser.add_argument("--workflow_max_target_messages", type=int, default=24, help="Maximum messages to keep from a workflow anchor through its terminal marker.")
    parser.add_argument("--workflow_repeat", type=int, default=1, help="Repeat workflow samples for loss weighting; repeats bypass exact dedupe.")
    parser.add_argument(
        "--workflow_anchor_tools",
        default="file_saver,generate_markdown_report",
        help="Comma-separated tools that start workflow-chain targets.",
    )
    parser.add_argument(
        "--workflow_terminal_tools",
        default="mark_step",
        help="Comma-separated tools that close workflow-chain targets when found after an anchor.",
    )
    parser.add_argument(
        "--workflow_previous_tools",
        default=(
            "ask_question_about_image,ask_question_about_video,audio_recognition,"
            "serper_search,image_search,fetch_website_content,fetch_website_content_with_images,"
            "fetch_website_images_only,file_read,extract_document_content,execute_code,file_find_in_content"
        ),
        help="Tools to force into workflow context when they precede a workflow anchor.",
    )
    parser.add_argument("--max_chars", type=int, default=0, help="Drop rendered samples whose text exceeds this many chars; 0 disables.")
    parser.add_argument("--min_assistant_chars", type=int, default=1, help="Drop samples with less assistant target text than this.")
    parser.add_argument("--dedupe", action="store_true", help="Drop exact duplicate rendered conversations.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output records.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_csv(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


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


def tool_call_name(tool_call: Dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    return str(function.get("name") or tool_call.get("name") or "")


def assistant_tool_names(message: Dict[str, Any]) -> List[str]:
    if message.get("role") != "assistant":
        return []
    names = []
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict):
            name = tool_call_name(tool_call)
            if name:
                names.append(name)
    return names


def primary_tool_name(message: Dict[str, Any]) -> str:
    names = assistant_tool_names(message)
    return names[0] if names else ""


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
        parts.append(str(content).strip())
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


def assistant_chars_from(messages: List[Dict[str, Any]], start_index: int) -> int:
    return sum(
        len(message_text(message))
        for index, message in enumerate(messages)
        if index >= start_index and message.get("role") == "assistant"
    )


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
            "supervised_from_message_index": 0,
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

        supervised_from = len(context)
        tool_names = assistant_tool_names(target_copy)
        items.append({
            "messages": item_messages,
            "meta": {
                "plan_id": record.get("plan_id"),
                "trace_type": record.get("trace_type"),
                "step": record.get("step"),
                "created_at": record.get("created_at"),
                "sample_mode": "turn",
                "target_tool_names": tool_names,
                "supervised_from_message_index": supervised_from,
                "assistant_chars": chars,
            },
        })
    return items


def include_tool_result_after(messages: List[Dict[str, Any]], index: int, stop: int) -> int:
    stop = min(stop, len(messages))
    if index + 1 < len(messages) and index + 1 >= stop and messages[index + 1].get("role") == "tool":
        return index + 2
    return stop


def next_assistant_tool_index(messages: List[Dict[str, Any]], start_index: int, tool_names: set[str], max_messages: int) -> int | None:
    stop = min(len(messages), start_index + max(1, max_messages))
    for index in range(start_index + 1, stop):
        if primary_tool_name(messages[index]) in tool_names:
            return index
    return None


def previous_assistant_tool_index(messages: List[Dict[str, Any]], target_index: int, tool_names: set[str]) -> int | None:
    for index in range(target_index - 1, -1, -1):
        name = primary_tool_name(messages[index])
        if name and (not tool_names or name in tool_names):
            return index
    return None


def selected_workflow_context_indices(messages: List[Dict[str, Any]], target_index: int, args: argparse.Namespace) -> List[int]:
    indices = set(selected_context_indices(messages, target_index, args.workflow_context_messages))
    previous_tools = split_csv(args.workflow_previous_tools)
    previous_index = previous_assistant_tool_index(messages, target_index, previous_tools)
    if previous_index is not None:
        indices.add(previous_index)
        if previous_index + 1 < target_index and messages[previous_index + 1].get("role") == "tool":
            indices.add(previous_index + 1)
    return sorted(index for index in indices if index < target_index)


def compress_workflow_message(message: Dict[str, Any], args: argparse.Namespace, stats: Dict[str, int]) -> Dict[str, Any]:
    if message.get("role") in {"tool", "user"}:
        return compress_context_message(message, args, stats)
    return copy.deepcopy(message)


def build_workflow_items(record: Dict[str, Any], messages: List[Dict[str, Any]], args: argparse.Namespace, stats: Dict[str, int]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    anchor_tools = split_csv(args.workflow_anchor_tools)
    terminal_tools = split_csv(args.workflow_terminal_tools)
    repeat = max(1, args.workflow_repeat)

    for anchor_index, anchor in enumerate(messages):
        anchor_name = primary_tool_name(anchor)
        if anchor_name not in anchor_tools:
            continue

        terminal_index = next_assistant_tool_index(
            messages,
            anchor_index,
            terminal_tools,
            args.workflow_max_target_messages,
        )
        if terminal_index is not None:
            end_index = include_tool_result_after(
                messages,
                terminal_index,
                min(len(messages), terminal_index + 1),
            )
            terminal_name = primary_tool_name(messages[terminal_index])
        else:
            end_index = include_tool_result_after(
                messages,
                anchor_index,
                min(len(messages), anchor_index + 1),
            )
            terminal_name = ""

        end_index = min(end_index, anchor_index + max(1, args.workflow_max_target_messages))
        context_indices = selected_workflow_context_indices(messages, anchor_index, args)
        if not context_indices:
            stats["dropped_workflow_empty_context"] += 1
            continue

        context = [
            compress_context_message(messages[index], args, stats)
            for index in context_indices
        ]
        target_segment = [
            compress_workflow_message(message, args, stats)
            for message in messages[anchor_index:end_index]
        ]
        if not target_segment:
            stats["dropped_workflow_empty_target"] += 1
            continue

        supervised_from = len(context)
        item_messages = context + target_segment
        chars = assistant_chars_from(item_messages, supervised_from)
        if chars < args.min_assistant_chars:
            stats["dropped_short_assistant"] += 1
            continue

        target_tool_names: List[str] = []
        for message in target_segment:
            target_tool_names.extend(assistant_tool_names(message))
        previous_name = ""
        previous_index = previous_assistant_tool_index(messages, anchor_index, set())
        if previous_index is not None:
            previous_name = primary_tool_name(messages[previous_index])

        for repeat_index in range(repeat):
            items.append({
                "messages": copy.deepcopy(item_messages),
                "meta": {
                    "plan_id": record.get("plan_id"),
                    "trace_type": record.get("trace_type"),
                    "step": record.get("step"),
                    "created_at": record.get("created_at"),
                    "sample_mode": "workflow",
                    "workflow_anchor_tool": anchor_name,
                    "workflow_previous_tool": previous_name,
                    "workflow_terminal_tool": terminal_name,
                    "workflow_repeat_index": repeat_index,
                    "target_tool_names": target_tool_names,
                    "supervised_from_message_index": supervised_from,
                    "allow_duplicate": repeat_index > 0,
                    "assistant_chars": chars,
                },
            })
            stats[f"workflow_anchor:{anchor_name}"] += 1
            if terminal_name:
                stats[f"workflow_transition:{anchor_name}->{terminal_name}"] += 1
            if previous_name:
                stats[f"workflow_previous:{previous_name}->{anchor_name}"] += 1
            if recovery_mode:
                stats[f"workflow_recovery:{recovery_mode}"] += 1

    return items


def keep_item(item: Dict[str, Any], args: argparse.Namespace, stats: Dict[str, int], seen: set[str]) -> bool:
    if item["meta"]["assistant_chars"] < args.min_assistant_chars:
        stats["dropped_short_assistant"] += 1
        return False
    rendered = render_plain_chat(item["messages"])
    if args.max_chars and len(rendered) > args.max_chars:
        stats["dropped_too_long"] += 1
        return False
    allow_duplicate = bool(item["meta"].get("allow_duplicate"))
    if args.dedupe and not allow_duplicate and rendered in seen:
        stats["dropped_duplicate"] += 1
        return False
    if allow_duplicate:
        stats["kept_weighted_duplicate"] += 1
    else:
        seen.add(rendered)
    item["text_preview"] = rendered[:1000]
    return True


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
        if args.sample_mode in {"full", "both", "all"}:
            full_item = build_full_item(record, messages)
            if full_item is not None:
                candidates.append(full_item)
        if args.sample_mode in {"turn", "both", "all"}:
            candidates.extend(build_turn_items(record, messages, args, stats))
        if args.add_workflow_samples or args.sample_mode in {"workflow", "all"}:
            candidates.extend(build_workflow_items(record, messages, args, stats))

        for item in candidates:
            if keep_item(item, args, stats, seen):
                output.append(item)
                stats[f"kept_{item['meta']['sample_mode']}"] += 1
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
    text_path = output_dir / "sft_text.jsonl"
    stats_path = output_dir / "stats.json"

    write_jsonl(messages_path, records)
    write_jsonl(
        text_path,
        ({"text": render_plain_chat(item["messages"]), "meta": item["meta"]} for item in records),
    )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "messages_path": os.fspath(messages_path),
        "text_path": os.fspath(text_path),
        "stats_path": os.fspath(stats_path),
        **stats,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
