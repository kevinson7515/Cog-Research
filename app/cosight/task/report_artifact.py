"""Validation helpers for the final Markdown report artifact contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_MIN_REPORT_BYTES = 128


def minimum_report_bytes() -> int:
    """Return the configured minimum report size without trusting bad env input."""

    try:
        return max(1, int(os.environ.get("COSIGHT_MIN_REPORT_BYTES", DEFAULT_MIN_REPORT_BYTES)))
    except (TypeError, ValueError):
        return DEFAULT_MIN_REPORT_BYTES


def validate_report_artifact(
    workspace_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str] | None,
    *,
    min_bytes: Optional[int] = None,
) -> Tuple[bool, str, Optional[str]]:
    """Validate that a report is a non-empty Markdown file inside the workspace.

    Returns ``(valid, reason, resolved_path)``. ``resolved_path`` is only
    populated for valid artifacts, so callers cannot accidentally persist an
    unchecked model-provided path.
    """

    if not report_path:
        return False, "generate_markdown_report did not return report_path", None

    workspace = Path(workspace_path).expanduser().resolve()
    candidate = Path(report_path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate

    try:
        candidate = candidate.resolve()
        if os.path.commonpath((str(workspace), str(candidate))) != str(workspace):
            return False, "report_path is outside the current task workspace", None
    except (OSError, ValueError):
        return False, "report_path could not be resolved safely", None

    if candidate.suffix.lower() not in {".md", ".markdown"}:
        return False, "final report must be a Markdown file", None
    if not candidate.is_file():
        return False, "reported Markdown file does not exist", None

    required_bytes = minimum_report_bytes() if min_bytes is None else max(1, int(min_bytes))
    try:
        size = candidate.stat().st_size
    except OSError:
        return False, "reported Markdown file could not be inspected", None
    if size < required_bytes:
        return False, f"reported Markdown file is too small ({size} < {required_bytes} bytes)", None

    return True, "ok", str(candidate)


__all__ = ["DEFAULT_MIN_REPORT_BYTES", "minimum_report_bytes", "validate_report_artifact"]
