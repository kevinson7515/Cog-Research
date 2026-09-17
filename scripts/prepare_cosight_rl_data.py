#!/usr/bin/env python3
"""Prepare Co-Sight quiz/rubric JSONL as VERL/JADE RL parquet data."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

SYSTEM_PROMPT = """You are Co-Sight, a multimodal deep-research agent.
Use the provided images and the user task as grounded evidence. Do not invent
sources, dates, visual details, URLs, or numerical data. If evidence is missing,
state the uncertainty explicitly. Separate observations from external evidence,
and connect important claims to concrete cited sources. Use bracket citations
only when they resolve to a real source in the final References/Sources section;
do not leave orphan citations or add unsupported reference dumps. Avoid repeated
sections or repeated prose. Produce a coherent Markdown research report that
follows all explicit user constraints."""


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", default="data/quiz_train_rubric.jsonl")
    parser.add_argument("--val_jsonl", default="")
    parser.add_argument("--output_dir", default="data/rl_cosight")
    parser.add_argument("--image_base_dir", default="data/images")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Split train_jsonl if val_jsonl is empty.")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--evidence_corpus", default=os.getenv("COSIGHT_EVIDENCE_CORPUS", ""))
    parser.add_argument("--max_evidence_items", type=int, default=int(os.getenv("COSIGHT_MAX_EVIDENCE_ITEMS", "6")))
    parser.add_argument("--max_evidence_chars", type=int, default=int(os.getenv("COSIGHT_MAX_EVIDENCE_CHARS", "4000")))
    parser.add_argument(
        "--include_evidence_in_prompt",
        action="store_true",
        default=_env_bool("COSIGHT_INCLUDE_EVIDENCE_IN_PROMPT", False),
        help="Append the local evidence pack to the user prompt for offline-RAG style RL.",
    )
    parser.add_argument(
        "--min_evidence_coverage",
        type=float,
        default=float(os.getenv("COSIGHT_MIN_EVIDENCE_COVERAGE", "0")),
        help=(
            "Fail if the fraction of prepared train rows with local_evidence is below this value. "
            "Use 0 to allow evidence-free RL data."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl_copy", action="store_true", help="Also write JSONL copies next to parquet files.")
    return parser.parse_args()


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def image_items(record: Dict[str, Any], image_base_dir: Path) -> List[Dict[str, str]]:
    raw_images = record.get("image_url")
    if raw_images is None:
        raw_images = record.get("images")
    if isinstance(raw_images, str):
        raw_images = [raw_images]

    items: List[Dict[str, str]] = []
    for raw in raw_images or []:
        if isinstance(raw, dict):
            raw = raw.get("url") or raw.get("path") or raw.get("image_url")
        if not raw:
            continue

        value = str(raw)
        if value.startswith(("http://", "https://", "file://")):
            image_ref = value
        else:
            image_path = Path(value)
            if not image_path.is_absolute():
                image_path = image_base_dir / image_path
            image_ref = "file://" + str(image_path.resolve())
        items.append({"image": image_ref})
    return items


def report_suffix(language: str) -> str:
    if str(language or "").lower().startswith("en"):
        return "Finally, generate a complete Markdown report."
    return "Finally, generate a complete Markdown report in the same language as the task."


def task_text(record: Dict[str, Any]) -> str:
    return str(
        record.get("body")
        or record.get("query")
        or record.get("question")
        or record.get("prompt")
        or ""
    ).strip()


def visual_identity_anchors(rubric: str) -> List[str]:
    """Extract explicit image-to-identity assertions from the training rubric."""

    anchors: List[str] = []
    pattern = re.compile(
        r"(?:图(?:像|片)?\s*\d+(?:\s*[\-\u2013\u2014~\u81f3]\s*(?:图(?:像|片)?\s*)?\d+)?|"
        r"images?\s*\d+(?:\s*[\-\u2013\u2014~]\s*\d+)?|"
        r"figures?\s*\d+(?:\s*[\-\u2013\u2014~]\s*\d+)?)"
        r".{0,24}?"
        r"(?:对应|指向|描绘|显示|展示|识别为|depicts?|shows?|presents?|points?\s+to|"
        r"corresponds?\s+to|identif(?:y|ies)\s+as)"
        r".{2,220}?(?=[：:；;。\n]|$)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(rubric or ""):
        anchor = re.sub(r"\s+", " ", match.group(0)).strip()
        if anchor and anchor not in anchors:
            anchors.append(anchor)
    return anchors[:12]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _task_keys(record: Mapping[str, Any], idx: int) -> List[str]:
    keys: List[str] = []
    values: List[Any] = []
    for field in ("qid", "id"):
        if field in record and record.get(field) is not None:
            values.append(record.get(field))
    if not values:
        values.append(idx)

    for value in values:
        if value is None:
            continue
        text = str(value)
        keys.append(text)
        match = re.search(r"(\d+)$", text)
        if match:
            keys.append(match.group(1))
    return list(dict.fromkeys(keys))


def _normalize_evidence_item(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        urls = [str(item) for item in _as_list(raw.get("urls")) if item]
        return {
            "id": raw.get("id"),
            "source_type": raw.get("source_type"),
            "source_ref": raw.get("source_ref") or raw.get("path") or raw.get("url"),
            "title": raw.get("title") or raw.get("source_ref") or raw.get("path") or "local evidence",
            "urls": urls,
            "text": str(raw.get("text") or raw.get("content") or "").strip(),
        }
    return {"title": "local evidence", "urls": [], "text": str(raw).strip()}


def load_evidence_corpus(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path:
        return {}
    evidence_path = Path(path)
    if not evidence_path.exists():
        return {}

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    if evidence_path.suffix.lower() == ".jsonl":
        with evidence_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                task_id = str(item.get("task_id") or "")
                if not task_id:
                    continue
                grouped.setdefault(task_id, []).append(_normalize_evidence_item(item))
        return grouped

    with evidence_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, Mapping):
        for task_id, items in data.items():
            grouped[str(task_id)] = [_normalize_evidence_item(item) for item in _as_list(items)]
    return grouped


def evidence_for_record(
    record: Mapping[str, Any],
    idx: int,
    evidence_by_task: Mapping[str, List[Dict[str, Any]]],
    max_items: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    if not evidence_by_task or max_items == 0 or max_chars == 0:
        return []

    items: List[Dict[str, Any]] = []
    for key in _task_keys(record, idx):
        items = list(evidence_by_task.get(key) or [])
        if items:
            break
    if not items:
        return []

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "trace_url": [],
        "workspace_url": [],
        "trace_other": [],
        "workspace_other": [],
    }
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        is_trace = item.get("source_type") == "trace_tool"
        has_urls = bool(_as_list(item.get("urls")))
        key = (
            "trace_url" if is_trace and has_urls
            else "workspace_url" if has_urls
            else "trace_other" if is_trace
            else "workspace_other"
        )
        buckets[key].append(item)

    for bucket in buckets.values():
        bucket.sort(
            key=lambda item: (
                -len(set(str(url) for url in _as_list(item.get("urls")) if url)),
                len(str(item.get("text") or "")),
            )
        )

    ordered: List[Dict[str, Any]] = []
    bucket_order = ("trace_url", "workspace_url", "trace_other", "workspace_other")
    while any(buckets.values()):
        for key in bucket_order:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))

    limit = max_items if max_items > 0 else len(ordered)
    candidates = ordered[:limit]
    if not candidates:
        return []

    total_budget = max_chars if max_chars > 0 else sum(len(str(item.get("text") or "")) for item in candidates)
    per_item_budget = max(320, total_budget // len(candidates))
    selected: List[Dict[str, Any]] = []
    remaining = total_budget
    for index, item in enumerate(candidates):
        text = str(item.get("text") or "").strip()
        items_left = len(candidates) - index
        fair_share = max(1, remaining // max(1, items_left))
        allowance = min(len(text), max(per_item_budget, fair_share), remaining)
        if allowance <= 0:
            break
        clipped = text[:allowance].rstrip()
        remaining -= len(clipped)
        urls = list(dict.fromkeys(str(url) for url in _as_list(item.get("urls")) if url))[:5]
        selected.append(
            {
                "id": item.get("id"),
                "source_type": item.get("source_type"),
                "source_ref": item.get("source_ref"),
                "title": item.get("title"),
                "urls": urls,
                "text": clipped,
            }
        )
    return selected


def format_evidence_for_prompt(evidence_items: List[Dict[str, Any]]) -> str:
    if not evidence_items:
        return ""
    blocks: List[str] = []
    for idx, item in enumerate(evidence_items, 1):
        title = str(item.get("title") or f"Evidence {idx}")
        source_ref = str(item.get("source_ref") or "")
        urls = [str(url) for url in _as_list(item.get("urls")) if url]
        header = f"[{idx}] {title}"
        if source_ref:
            header += f"\nSource: {source_ref}"
        if urls:
            header += "\nURLs: " + "; ".join(urls[:5])
        blocks.append(f"{header}\n{str(item.get('text') or '').strip()}")
    return "\n\n".join(blocks).strip()


def build_prompt(
    record: Dict[str, Any],
    image_count: int,
    evidence_items: List[Dict[str, Any]] | None = None,
    include_evidence: bool = False,
) -> List[Dict[str, str]]:
    body = task_text(record)
    language = str(record.get("language") or "")
    image_tokens = "\n".join("<image>" for _ in range(image_count))
    evidence_text = format_evidence_for_prompt(evidence_items or []) if include_evidence else ""
    evidence_section = (
        "\n\nLocal evidence cache from previous tool traces/workspaces. Use it as source material; do not treat it as a final answer. "
        "When you use cached external evidence, cite the matching source id like [1] and include a final References/Sources section mapping every used id to its URL or Source. "
        "Do not cite claims that are not supported by the images, task, or cached evidence.\n"
        f"{evidence_text}"
        if evidence_text
        else ""
    )
    user_text = f"{image_tokens}\n{body}{evidence_section}\n\n{report_suffix(language)}".strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def convert_records(
    records: List[Dict[str, Any]],
    image_base_dir: Path,
    split: str,
    evidence_by_task: Mapping[str, List[Dict[str, Any]]] | None = None,
    max_evidence_items: int = 0,
    max_evidence_chars: int = 0,
    include_evidence_in_prompt: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        images = image_items(record, image_base_dir)
        image_names = [Path((item.get("image") or "").replace("file://", "")).name for item in images]
        qid = record.get("qid", record.get("id", idx))
        body = task_text(record)
        rubric = str(record.get("rubric") or "").strip()
        identity_anchors = visual_identity_anchors(rubric)
        local_evidence = evidence_for_record(
            record,
            idx,
            evidence_by_task or {},
            max_evidence_items,
            max_evidence_chars,
        )
        ground_truth = {
            "task_prompt": body,
            "language": record.get("language"),
            "tags": record.get("tags", []),
        }
        if rubric:
            ground_truth["rubric"] = rubric
        if identity_anchors:
            ground_truth["visual_identity_anchors"] = identity_anchors
        if local_evidence:
            ground_truth["local_evidence"] = local_evidence
        row = {
            "data_source": "cosight_deep_research",
            "prompt": build_prompt(record, len(images), local_evidence, include_evidence_in_prompt),
            "images": images,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "index": idx,
                "qid": qid,
                "split": split,
                "task_prompt": body,
                "rubric": rubric,
                "visual_identity_anchors": identity_anchors,
                "caption": record.get("caption"),
                "expected_images": image_names,
                "language": record.get("language"),
                "difficulty": record.get("difficulty"),
                "tags": record.get("tags", []),
                "local_evidence": local_evidence,
                "local_evidence_in_prompt": bool(include_evidence_in_prompt and local_evidence),
            },
        }
        rows.append(row)
    return rows


def limit_rows(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit and limit > 0:
        return rows[:limit]
    return rows


def write_rows(rows: List[Dict[str, Any]], parquet_path: Path, jsonl_copy: bool) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("prepare_cosight_rl_data.py requires pandas and pyarrow/fastparquet to write parquet.") from exc

    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    if jsonl_copy:
        jsonl_path = parquet_path.with_suffix(".jsonl")
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError(f"--val_ratio must be in [0, 1), got {args.val_ratio}.")
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    image_base_dir = Path(args.image_base_dir)
    evidence_by_task = load_evidence_corpus(args.evidence_corpus)

    train_records = list(iter_jsonl(args.train_jsonl))
    original_train_count = len(train_records)
    val_path = Path(args.val_jsonl) if args.val_jsonl else None
    val_records = list(iter_jsonl(val_path)) if val_path and val_path.exists() else []
    if not train_records:
        raise ValueError(f"No train records found in {args.train_jsonl}.")

    if not val_records and args.val_ratio > 0:
        random.shuffle(train_records)
        n_val = max(1, int(len(train_records) * args.val_ratio))
        val_records = train_records[:n_val]
        train_records = train_records[n_val:]

    train_rows = limit_rows(
        convert_records(
            train_records,
            image_base_dir,
            "train",
            evidence_by_task=evidence_by_task,
            max_evidence_items=args.max_evidence_items,
            max_evidence_chars=args.max_evidence_chars,
            include_evidence_in_prompt=args.include_evidence_in_prompt,
        ),
        args.max_train_samples,
    )
    val_rows = limit_rows(
        convert_records(
            val_records,
            image_base_dir,
            "val",
            evidence_by_task=evidence_by_task,
            max_evidence_items=args.max_evidence_items,
            max_evidence_chars=args.max_evidence_chars,
            include_evidence_in_prompt=args.include_evidence_in_prompt,
        ),
        args.max_val_samples,
    )

    write_rows(train_rows, output_dir / "train.parquet", args.jsonl_copy)
    write_rows(val_rows, output_dir / "val.parquet", args.jsonl_copy)

    evidence_train = sum(1 for row in train_rows if row["extra_info"].get("local_evidence"))
    evidence_val = sum(1 for row in val_rows if row["extra_info"].get("local_evidence"))
    evidence_train_coverage = evidence_train / max(1, len(train_rows))
    if args.min_evidence_coverage > 0 and evidence_train_coverage < args.min_evidence_coverage:
        raise ValueError(
            "Prepared RL data has insufficient local evidence coverage: "
            f"{evidence_train}/{len(train_rows)} ({evidence_train_coverage:.1%}) < "
            f"--min_evidence_coverage={args.min_evidence_coverage:.1%}. "
            "Rebuild the evidence corpus from the training workspace/trace, check task-id alignment, "
            "or set COSIGHT_MIN_EVIDENCE_COVERAGE=0 only for an intentional evidence-free run."
        )
    print(
        json.dumps(
            {
                "train": len(train_rows),
                "val": len(val_rows),
                "input_train_records": original_train_count,
                "output_dir": str(output_dir),
                "evidence_corpus": str(args.evidence_corpus) if evidence_by_task else "",
                "train_with_local_evidence": evidence_train,
                "val_with_local_evidence": evidence_val,
                "train_local_evidence_coverage": evidence_train_coverage,
                "val_local_evidence_coverage": evidence_val / max(1, len(val_rows)),
                "local_evidence_in_prompt": args.include_evidence_in_prompt,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
