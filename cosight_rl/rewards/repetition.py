"""Normalized repetition detection for Co-Sight reports.

This module is deterministic and cheap enough to run inside offline audits and
VERL reward workers. It targets the report-collapse failure where paragraphs or
sentences repeat while citation ids, URLs, or minor punctuation change.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List


URL_TOKEN = "<url>"

_CITATION_RE = re.compile(
    r"""
    (?:
      \[
      \s*
      (?:\d+|[ivxlcdm]+)
      (?:\s*(?:,|;|-|--|\u2013|\u2014)\s*(?:\d+|[ivxlcdm]+))*
      \s*
      \]
    )+
    """,
    re.IGNORECASE | re.VERBOSE,
)
_URL_RE = re.compile(r"https?://[^\s)\]}>\"'\u3002\uff0c\uff1b\uff1a\uff01\uff1f]+", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((?:https?://|file://)[^)]+\)", re.IGNORECASE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_HTML_TAG_RE = re.compile(r"<[^>\n]+>")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_MARKER_RE = re.compile(r"^\s{0,6}(?:[-*+]|\d+[.)\u3001]|[A-Za-z][.)])\s+")
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?")
_TABLE_DIVIDER_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_SENTENCE_RE = re.compile(r"[^.!?\u3002\uff01\uff1f]+[.!?\u3002\uff01\uff1f]?", re.MULTILINE)
_TOKEN_RE = re.compile(r"[0-9a-zA-Z%]+|[\u4e00-\u9fff]")


def normalize_text_for_repetition(text: str) -> str:
    """Normalize text for duplicate detection while preserving semantics.

    The output is lowercase, whitespace-collapsed text. Markdown citation ids,
    URLs, link destinations, Markdown decorations, and punctuation-only
    differences are removed or normalized so citation-number churn does not hide
    repeated prose.
    """

    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _IMAGE_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(lambda match: f"{match.group(1)} {URL_TOKEN}", text)
    text = _URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _CITATION_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("`", " ")
    text = re.sub(r"[*_~]+", " ", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", " ", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", " ", text)
    text = re.sub(r"(?m)^\s*(?:[-*+]|\d+[.)\u3001]|[A-Za-z][.)])\s+", " ", text)
    text = text.lower()

    placeholder = " zzurltokenzz "
    text = text.replace(URL_TOKEN, placeholder)
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff%]+", " ", text)
    text = text.replace("zzurltokenzz", URL_TOKEN)
    return re.sub(r"\s+", " ", text).strip()


def _is_ignorable_block(block: str) -> bool:
    raw = block.strip()
    if not raw:
        return True
    if _HEADING_RE.match(raw):
        return True
    if raw.startswith(("```", "---", "***")):
        return True
    if _TABLE_DIVIDER_RE.match(raw):
        return True
    stripped = raw.strip("|-: ")
    return not bool(stripped)


def _clean_paragraph_block(block: str) -> str:
    lines: List[str] = []
    in_code = False
    for line in block.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if _HEADING_RE.match(line) or _TABLE_DIVIDER_RE.match(line):
            continue
        line = _BLOCKQUOTE_RE.sub("", line)
        line = _LIST_MARKER_RE.sub("", line)
        lines.append(line.strip())
    return " ".join(part for part in lines if part)


def split_paragraphs(text: str, min_chars: int = 40) -> List[str]:
    """Split Markdown text into normalized, non-trivial paragraphs."""

    paragraphs: List[str] = []
    blocks = re.split(r"\n\s*\n+", str(text or ""))
    for block in blocks:
        if _is_ignorable_block(block):
            continue
        # Long list items are meaningful report body. Split them only when list
        # markers appear at line starts, while keeping wrapped paragraphs intact.
        list_items = re.split(r"(?m)^\s*(?=(?:[-*+]|\d+[.)\u3001])\s+)", block)
        for item in list_items:
            cleaned = _clean_paragraph_block(item)
            normalized = normalize_text_for_repetition(cleaned)
            if len(normalized) >= min_chars:
                paragraphs.append(normalized)
    return paragraphs


def split_sentences(text: str, min_chars: int = 30) -> List[str]:
    """Split Chinese/English prose into normalized sentence-like chunks."""

    compact = re.sub(r"\s+", " ", str(text or ""))
    sentences: List[str] = []
    for match in _SENTENCE_RE.finditer(compact):
        normalized = normalize_text_for_repetition(match.group(0))
        if len(normalized) >= min_chars:
            sentences.append(normalized)
    return sentences


def _top_repeated(counter: Counter[str], limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for text, count in counter.most_common():
        if count <= 1:
            continue
        rows.append({"count": count, "text": text[:240]})
        if len(rows) >= limit:
            break
    return rows


def _token_ngrams(text: str, n: int = 24) -> Counter[tuple[str, ...]]:
    normalized = normalize_text_for_repetition(text)
    tokens = _TOKEN_RE.findall(normalized)
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def compute_soft_repetition_penalty(metrics: Dict[str, Any]) -> float:
    """Compute the soft repetition penalty from normalized metrics."""

    max_repeat_block_score = float(metrics.get("max_repeat_block_score", 0.0))
    if not max_repeat_block_score:
        max_repeat_block_score = min(float(metrics.get("max_repeated_paragraph_count", 0)) / 10.0, 1.0)
    max_sentence_repeat_score = min(float(metrics.get("max_repeated_sentence_count", 0)) / 25.0, 1.0)
    ngram_repeat_score = min(float(metrics.get("duplicate_ngram_ratio", 0.0)) * 2.0, 1.0)
    penalty = (
        0.65 * float(metrics.get("duplicate_paragraph_ratio", 0.0))
        + 0.15 * float(metrics.get("duplicate_sentence_ratio", 0.0))
        + 0.10 * max_repeat_block_score
        + 0.06 * max_sentence_repeat_score
        + 0.04 * ngram_repeat_score
    )
    return max(0.0, min(0.6, penalty))


def compute_repetition_metrics(text: str) -> Dict[str, Any]:
    """Return paragraph/sentence duplicate metrics for a report."""

    paragraphs = split_paragraphs(text)
    sentences = split_sentences(text)
    paragraph_counts = Counter(paragraphs)
    sentence_counts = Counter(sentences)
    ngram_counts = _token_ngrams(text)

    duplicate_paragraph_extra = sum(count - 1 for count in paragraph_counts.values() if count > 1)
    duplicate_sentence_extra = sum(count - 1 for count in sentence_counts.values() if count > 1)
    duplicate_ngram_extra = sum(count - 1 for count in ngram_counts.values() if count > 3)
    max_repeated_paragraph_count = max(paragraph_counts.values(), default=0)
    max_repeated_sentence_count = max(sentence_counts.values(), default=0)
    max_repeated_ngram_count = max(ngram_counts.values(), default=0)
    duplicate_paragraph_ratio = duplicate_paragraph_extra / max(1, len(paragraphs))
    duplicate_sentence_ratio = duplicate_sentence_extra / max(1, len(sentences))
    duplicate_ngram_ratio = duplicate_ngram_extra / max(1, sum(ngram_counts.values()))
    max_repeat_block_score = min(max_repeated_paragraph_count / 10.0, 1.0)

    severe_reasons: List[str] = []
    if duplicate_paragraph_ratio > 0.25:
        severe_reasons.append("duplicate_paragraph_ratio")
    if max_repeated_paragraph_count > 5:
        severe_reasons.append("max_repeated_paragraph_count")
    # Some collapses repeat a sentence cycle inside one huge paragraph. Keep the
    # required paragraph gates above, and add a conservative fallback for these.
    if duplicate_sentence_ratio > 0.35 and max_repeated_sentence_count > 10:
        severe_reasons.append("duplicate_sentence_block")
    if duplicate_sentence_ratio > 0.18 and max_repeated_sentence_count > 30:
        severe_reasons.append("max_repeated_sentence_count")
    if duplicate_ngram_ratio > 0.20 and max_repeated_ngram_count > 20:
        severe_reasons.append("duplicate_ngram_block")

    metrics: Dict[str, Any] = {
        "num_paragraphs": len(paragraphs),
        "num_unique_paragraphs": len(paragraph_counts),
        "duplicate_paragraph_ratio": duplicate_paragraph_ratio,
        "max_repeated_paragraph_count": max_repeated_paragraph_count,
        "num_sentences": len(sentences),
        "num_unique_sentences": len(sentence_counts),
        "duplicate_sentence_ratio": duplicate_sentence_ratio,
        "max_repeated_sentence_count": max_repeated_sentence_count,
        "duplicate_ngram_ratio": duplicate_ngram_ratio,
        "max_repeated_ngram_count": max_repeated_ngram_count,
        "max_repeat_block_score": max_repeat_block_score,
        "top_repeated_paragraphs": _top_repeated(paragraph_counts, 10),
        "top_repeated_sentences": _top_repeated(sentence_counts, 10),
        "severe_repetition": bool(severe_reasons),
        "severe_repetition_reasons": severe_reasons,
    }
    metrics["P_repeat_soft"] = compute_soft_repetition_penalty(metrics)
    return metrics


__all__ = [
    "compute_repetition_metrics",
    "compute_soft_repetition_penalty",
    "normalize_text_for_repetition",
    "split_paragraphs",
    "split_sentences",
]
