#!/usr/bin/env python3
"""Offline audit for Co-Sight reward design.

Examples:
  python scripts/evaluate_cosight_rewards.py --workspace_dir work_space/work_space_20260621_075554
  python scripts/evaluate_cosight_rewards.py --benchmark_reason_file pasted-text.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cosight_rl.rewards.reward_evaluator import classify_benchmark_reason, score_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace_dir", default="", help="Workspace root containing task_* directories.")
    parser.add_argument("--benchmark_reason_file", default="", help="JSONL with qid/reason from judge output.")
    parser.add_argument("--output_jsonl", default="data/rl_reward_audit/reward_audit.jsonl")
    parser.add_argument("--summary_json", default="data/rl_reward_audit/reward_summary.json")
    parser.add_argument("--min_report_chars", type=int, default=10000)
    return parser.parse_args()


def iter_workspace_reports(workspace_dir: Path, min_chars: int) -> Iterable[Dict[str, Any]]:
    for path in sorted(workspace_dir.glob("task_*/*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) < min_chars:
            continue
        task_dir = path.parent
        images = [p.name for p in task_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        yield {
            "kind": "workspace_report",
            "id": f"{task_dir.name}/{path.name}",
            "path": str(path),
            "solution": text,
            "extra_info": {"expected_images": images},
        }


def iter_benchmark_reasons(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            reason = str(record.get("reason") or "")
            classified = classify_benchmark_reason(reason)
            yield {
                "kind": "benchmark_reason",
                "id": record.get("qid") or f"line_{line_no}",
                "reason": reason,
                "classification": classified,
            }


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    score_bins = Counter()
    cap_reasons = Counter()
    reason_tags = Counter()

    if args.workspace_dir:
        for item in iter_workspace_reports(Path(args.workspace_dir), args.min_report_chars):
            reward = score_solution(item["solution"], extra_info=item["extra_info"])
            row = {k: v for k, v in item.items() if k != "solution"}
            row["reward"] = reward
            rows.append(row)
            score = reward["capped_score"]
            score_bins[f"{int(score * 10) / 10:.1f}"] += 1
            cap_reasons.update(reward.get("cap_reasons") or [])

    if args.benchmark_reason_file:
        for item in iter_benchmark_reasons(Path(args.benchmark_reason_file)):
            rows.append(item)
            reason_tags.update(item["classification"].get("tags") or [])

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "records": len(rows),
        "score_bins": dict(sorted(score_bins.items())),
        "cap_reasons": dict(cap_reasons.most_common()),
        "benchmark_reason_tags": dict(reason_tags.most_common()),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
