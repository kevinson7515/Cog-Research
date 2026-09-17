#!/usr/bin/env python3
"""Mix a primary SFT JSONL with replay samples from a previous stage."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary_file", required=True, help="Main Stage-2 SFT JSONL.")
    parser.add_argument("--replay_file", required=True, help="Stage-1 replay SFT JSONL.")
    parser.add_argument("--output_file", required=True, help="Mixed output JSONL.")
    parser.add_argument(
        "--replay_ratio",
        type=float,
        default=0.35,
        help="Target replay fraction in the mixed file: replay / (primary + replay). Must be in [0, 1).",
    )
    parser.add_argument("--max_replay_samples", type=int, default=0, help="Cap replay samples after ratio calculation; 0 means no cap.")
    parser.add_argument(
        "--replay_sample_modes",
        default="",
        help="Optional comma-separated meta.sample_mode allowlist for replay records. Supports prefix wildcard such as workflow_*.",
    )
    parser.add_argument("--shuffle", type=parse_bool, default=True)
    parser.add_argument("--dedupe", type=parse_bool, default=False, help="Dedupe exact JSON records. Defaults false to preserve weighting repeats.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_modes(raw_modes: str) -> List[str]:
    return [item.strip() for item in raw_modes.split(",") if item.strip()]


def mode_matches(mode: str, patterns: List[str]) -> bool:
    if not patterns:
        return True
    for pattern in patterns:
        if pattern.endswith("*") and mode.startswith(pattern[:-1]):
            return True
        if mode == pattern:
            return True
    return False


def filter_replay(records: List[Dict[str, Any]], raw_modes: str) -> List[Dict[str, Any]]:
    patterns = parse_modes(raw_modes)
    if not patterns:
        return records
    filtered = []
    for record in records:
        mode = str((record.get("meta") or {}).get("sample_mode") or "")
        if mode_matches(mode, patterns):
            filtered.append(record)
    return filtered


def replay_count(primary_count: int, available_replay: int, ratio: float, max_replay_samples: int) -> int:
    if ratio < 0 or ratio >= 1:
        raise ValueError(f"--replay_ratio must be in [0, 1), got {ratio}.")
    if ratio == 0 or primary_count <= 0 or available_replay <= 0:
        return 0
    count = int(round(primary_count * ratio / (1 - ratio)))
    count = max(1, count)
    if max_replay_samples > 0:
        count = min(count, max_replay_samples)
    return min(count, available_replay)


def dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for record in records:
        key = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def main() -> None:
    args = parse_args()
    primary = load_jsonl(args.primary_file)
    replay_all = load_jsonl(args.replay_file)
    replay_pool = filter_replay(replay_all, args.replay_sample_modes)

    rng = random.Random(args.seed)
    selected_count = replay_count(len(primary), len(replay_pool), args.replay_ratio, args.max_replay_samples)
    replay_indices = list(range(len(replay_pool)))
    rng.shuffle(replay_indices)
    replay = [replay_pool[index] for index in replay_indices[:selected_count]]

    mixed = list(primary) + replay
    before_dedupe = len(mixed)
    if args.dedupe:
        mixed = dedupe_records(mixed)
    if args.shuffle:
        rng.shuffle(mixed)
    write_jsonl(args.output_file, mixed)

    stats = {
        "primary_file": str(Path(args.primary_file)),
        "replay_file": str(Path(args.replay_file)),
        "output_file": str(Path(args.output_file)),
        "primary_samples": len(primary),
        "replay_samples_available": len(replay_all),
        "replay_samples_after_filter": len(replay_pool),
        "replay_samples_selected": len(replay),
        "mixed_samples_before_dedupe": before_dedupe,
        "mixed_samples": len(mixed),
        "actual_replay_ratio": round(len(replay) / len(mixed), 6) if mixed else 0.0,
        "target_replay_ratio": args.replay_ratio,
        "dedupe": args.dedupe,
        "shuffle": args.shuffle,
        "seed": args.seed,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
