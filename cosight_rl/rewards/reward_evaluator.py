"""Deterministic Co-Sight reward evaluator.

The evaluator is designed for two paths:

* offline reward audits over existing workspaces;
* VERL/JADE custom reward loading through ``compute_score``.

It uses transparent heuristics for failures that can be detected from text and
tool metadata. Hard-to-prove failures such as wrong visual identity can be
supplied as judge tags in ``extra_info``.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

try:
    from .repetition import compute_repetition_metrics
except ImportError:  # JADE loads custom rewards via spec_from_file_location.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from cosight_rl.rewards.repetition import compute_repetition_metrics


FAILURE_CAPS = {
    "empty_or_crashed": 0.05,
    "wrong_visual_identity": 0.15,
    "fabricated_source_data": 0.20,
    "repetition_collapse": 0.25,
    "format_instruction_fail": 0.35,
    "core_quant_error": 0.40,
}

REWARD_WEIGHTS = {
    "R_visual_grounding": 0.30,
    "R_evidence_factuality": 0.25,
    "R_task_following": 0.15,
    "R_quantitative_reasoning": 0.12,
    "R_report_quality": 0.10,
    "R_workflow": 0.08,
}


@dataclass
class RewardResult:
    """Serializable reward breakdown used by audit and PPO reward code."""

    final_reward: float
    raw_reward: float
    reward_cap: float
    hard_gate_reasons: List[str] = field(default_factory=list)
    R_visual_grounding: float = 0.5
    R_evidence_factuality: float = 0.5
    R_task_following: float = 0.5
    R_quantitative_reasoning: float = 0.5
    R_report_quality: float = 0.5
    R_workflow: float = 0.5
    P_cost: float = 0.0
    P_invalid_tool: float = 0.0
    P_repeat_soft: float = 0.0
    repetition_metrics: Dict[str, Any] = field(default_factory=dict)
    short_diagnosis: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # VERL's naive reward manager reads "score". The other aliases keep
        # older app.cosight.rl.reward callers working while the new audit uses
        # explicit final_reward/R_* names.
        data["score"] = data["final_reward"]
        data["capped_score"] = data["final_reward"]
        data["cap"] = data["reward_cap"]
        data["cap_reasons"] = data["hard_gate_reasons"]
        data["visual_grounding"] = data["R_visual_grounding"]
        data["evidence_factuality"] = data["R_evidence_factuality"]
        data["task_following"] = data["R_task_following"]
        data["quantitative_reasoning"] = data["R_quantitative_reasoning"]
        data["report_quality"] = data["R_report_quality"]
        data["workflow"] = data["R_workflow"]
        data["cost_penalty"] = data["P_cost"]
        data["invalid_tool_penalty"] = data["P_invalid_tool"]
        data["repetition_penalty"] = data["P_repeat_soft"]
        data["repetition"] = data["repetition_metrics"]
        return data


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp numeric values and treat NaN as ``low``."""

    if isinstance(value, float) and math.isnan(value):
        return low
    return max(low, min(high, float(value)))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def extract_urls(text: str) -> List[str]:
    """Extract http/https URLs from text."""

    return re.findall(r"https?://[^\s)\]}>\"'\u3002\uff0c\uff1b\uff1a\uff01\uff1f]+", text or "")


def broken_url_like_count(urls: Iterable[str]) -> int:
    """Count obvious placeholder, truncated, or malformed URLs."""

    bad = 0
    for url in urls:
        lower = url.lower().rstrip(".,;:")
        if any(token in lower for token in ("example.com", "placeholder", "localhost", "127.0.0.1", "todo")):
            bad += 1
            continue
        if lower.endswith(("...", "/...", "%", "%e")):
            bad += 1
            continue
        if not re.match(r"https?://[^/\s]+\.[^/\s]+", lower):
            bad += 1
    return bad


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _normalize_url(url: str) -> str:
    """Normalize only harmless URL surface differences for exact matching."""

    cleaned = str(url or "").strip().rstrip(".,;:!?，。；：！？")
    try:
        parts = urlsplit(cleaned)
    except ValueError:
        return cleaned
    if not parts.scheme or not parts.netloc:
        return cleaned
    # URL paths and query strings can be case-sensitive. Only the scheme and
    # host are safe to fold while retaining exact source identity.
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, parts.fragment))


def _citation_integrity_details(text: str) -> Dict[str, Any]:
    """Audit the numeric citation contract used by MMDR/TRACE.

    A valid cited index must map to exactly one URL in a reference entry.  This
    deliberately does not attempt network retrieval; claim-page entailment is
    approximated separately by restricting report URLs to the provided local
    evidence pack.
    """

    mapping: Dict[str, List[str]] = {}
    body_lines: List[str] = []
    reference_line_re = re.compile(r"^\s*(?:[-*+]\s*)?\[(\d+)\]\s*(?::|\s)\s*(.*)$")
    for line in (text or "").splitlines():
        match = reference_line_re.match(line)
        line_urls = extract_urls(match.group(2)) if match else []
        if match and line_urls:
            mapping.setdefault(match.group(1), []).extend(_normalize_url(url) for url in line_urls)
        else:
            body_lines.append(line)

    body = "\n".join(body_lines)
    citation_ids = re.findall(r"\[(\d+)\]", body)
    unique_citation_ids = sorted(set(citation_ids), key=lambda item: int(item))
    normalized_mapping = {
        index: sorted(set(url for url in urls if url))
        for index, urls in mapping.items()
    }
    valid_mapping_ids = sorted(
        (index for index, urls in normalized_mapping.items() if len(urls) == 1),
        key=lambda item: int(item),
    )
    resolved_ids = sorted(
        set(unique_citation_ids).intersection(valid_mapping_ids),
        key=lambda item: int(item),
    )
    unresolved_ids = sorted(
        set(unique_citation_ids).difference(resolved_ids),
        key=lambda item: int(item),
    )
    orphan_mapping_ids = sorted(
        set(valid_mapping_ids).difference(unique_citation_ids),
        key=lambda item: int(item),
    )
    mapped_urls = sorted(
        set(url for index in valid_mapping_ids for url in normalized_mapping.get(index, []))
    )

    claim_like_sentences = []
    for sentence in re.split(r"(?<=[.!?。！？])\s*|\n+", body):
        stripped = sentence.strip()
        if len(stripped) < 40 or stripped.startswith(("#", "```", "![")):
            continue
        if re.fullmatch(r"[-*+|\s\d.:]+", stripped):
            continue
        claim_like_sentences.append(stripped)
    cited_claims = sum(bool(re.search(r"\[(\d+)\]", sentence)) for sentence in claim_like_sentences)
    claim_citation_coverage = cited_claims / max(1, len(claim_like_sentences)) if claim_like_sentences else 0.0

    return {
        "citation_count": len(citation_ids),
        "unique_citation_ids": unique_citation_ids,
        "mapping_count": len(normalized_mapping),
        "valid_mapping_ids": valid_mapping_ids,
        "resolved_citation_ids": resolved_ids,
        "unresolved_citation_ids": unresolved_ids,
        "orphan_mapping_ids": orphan_mapping_ids,
        "orphan_mapping_ratio": len(orphan_mapping_ids) / max(1, len(valid_mapping_ids)),
        "mapped_urls": mapped_urls,
        "mapping_integrity": len(resolved_ids) / max(1, len(unique_citation_ids)) if unique_citation_ids else 0.0,
        "claim_like_sentence_count": len(claim_like_sentences),
        "cited_claim_count": cited_claims,
        "claim_citation_coverage": claim_citation_coverage,
    }


def _current_year() -> int:
    env_value = os.getenv("COSIGHT_CURRENT_YEAR")
    if env_value and env_value.isdigit():
        return int(env_value)
    return datetime.now().year


def _future_year_count(text: str) -> int:
    current_year = _current_year()
    years = [int(item) for item in re.findall(r"\b(20[2-9]\d|21\d{2})\b", text or "")]
    return sum(1 for year in years if year > current_year + 1)


def _extract_expected_images(extra_info: Mapping[str, Any]) -> List[str]:
    images: List[str] = []
    for key in ("expected_images", "image_paths", "images", "image_url", "attached_files"):
        for item in _as_list(extra_info.get(key)):
            if isinstance(item, Mapping):
                item = item.get("path") or item.get("url") or item.get("image") or item.get("image_url")
            if not item:
                continue
            value = str(item).replace("file://", "")
            name = Path(value).name
            if name:
                images.append(name)
    return sorted(set(images))


def _image_coverage(text: str, expected_images: List[str]) -> Dict[str, Any]:
    lower = text.lower()
    hits: List[str] = []
    misses: List[str] = []
    for idx, image in enumerate(expected_images):
        stem = Path(image).stem.lower()
        one_based = idx + 1
        markers = [
            image.lower(),
            f"image {idx}",
            f"image {one_based}",
            f"figure {idx}",
            f"figure {one_based}",
            f"fig. {idx}",
            f"fig. {one_based}",
            f"img {idx}",
            f"img {one_based}",
            f"\u56fe{idx}",
            f"\u56fe{one_based}",
            f"\u56fe {idx}",
            f"\u56fe {one_based}",
            f"\u56fe\u7247{one_based}",
            f"\u56fe\u50cf{one_based}",
            f"\u7b2c{one_based}\u5f20",
        ]
        if len(stem) >= 3:
            markers.append(stem)
        if any(marker in lower for marker in markers):
            hits.append(image)
        else:
            misses.append(image)
    coverage = len(hits) / max(1, len(expected_images)) if expected_images else 0.0
    return {"coverage": coverage, "hits": hits, "misses": misses}


def _visual_grounding_score(text: str, expected_images: List[str]) -> float:
    if not expected_images:
        return 0.55
    coverage = _image_coverage(text, expected_images)["coverage"]
    visual_terms = len(
        re.findall(
            r"image|figure|fig\.|visual|visible|chart|axis|legend|unit|curve|"
            r"\u56fe\u50cf|\u56fe\u7247|\u56fe\u8868|\u753b\u9762|\u4e3b\u4f53|"
            r"\u80cc\u666f|\u6784\u56fe|\u5750\u6807|\u56fe\u4f8b|\u5355\u4f4d|"
            r"\u66f2\u7ebf|\u53ef\u89c1",
            text or "",
            re.IGNORECASE,
        )
    )
    uncertainty_terms = len(
        re.findall(
            r"uncertain|not visible|cannot confirm|cannot determine|"
            r"\u4e0d\u786e\u5b9a|\u65e0\u6cd5\u786e\u8ba4|\u770b\u4e0d\u6e05|\u8bc1\u636e\u4e0d\u8db3",
            text or "",
            re.IGNORECASE,
        )
    )
    term_score = min(1.0, visual_terms / max(4, len(expected_images) * 4))
    uncertainty_score = min(1.0, uncertainty_terms / 3) if uncertainty_terms else 0.25
    return clamp(0.65 * coverage + 0.25 * term_score + 0.10 * uncertainty_score)


def _evidence_factuality_score(text: str) -> float:
    urls = extract_urls(text)
    citations = re.findall(r"\[(?:\d+|[A-Za-z][\w-]*)\]", text or "")
    source_sections = len(
        re.findall(
            r"^#{1,3}\s*(references?|sources?|bibliography|"
            r"\u53c2\u8003|\u6765\u6e90|\u53c2\u8003\u6587\u732e)",
            text or "",
            re.IGNORECASE | re.MULTILINE,
        )
    )
    evidence_markers = len(
        re.findall(
            r"evidence|source|according to|citation|"
            r"\u6765\u6e90|\u8bc1\u636e|\u8d44\u6599|\u5f15\u7528|\u6839\u636e",
            text or "",
            re.IGNORECASE,
        )
    )
    no_source_markers = len(
        re.findall(
            r"citation needed|placeholder|no source|no available source|"
            r"\u65e0\u6cd5\u63d0\u4f9b|\u672a\u627e\u5230|\u65e0\u53ef\u7528",
            text or "",
            re.IGNORECASE,
        )
    )
    bad_urls = broken_url_like_count(urls)
    future_years = _future_year_count(text)

    if not urls and not citations and source_sections == 0:
        base = 0.30
    else:
        url_quality = 1.0 - min(1.0, bad_urls / max(1, len(urls)))
        citation_score = min(1.0, len(citations) / 8) if citations else 0.35
        source_score = 0.15 if source_sections else 0.0
        evidence_score = min(0.15, evidence_markers * 0.01)
        base = 0.45 * url_quality + 0.35 * citation_score + source_score + evidence_score
    base -= min(0.25, future_years * 0.05)
    base -= min(0.25, no_source_markers * 0.05)
    return clamp(base)


def _citation_aligned_evidence_factuality_score(text: str, citation_details: Mapping[str, Any]) -> float:
    """Blend the legacy style heuristic with TRACE-like citation integrity.

    The blend keeps the change gentle for already-good reports while removing
    the old exploit where many ``[n]`` markers with no URL received a high
    evidence score.
    """

    legacy = _evidence_factuality_score(text)
    citation_count = int(citation_details.get("citation_count", 0))
    mapped_urls = list(citation_details.get("mapped_urls") or [])
    mapping_integrity = float(citation_details.get("mapping_integrity", 0.0))
    claim_coverage = float(citation_details.get("claim_citation_coverage", 0.0))

    if citation_count == 0:
        aligned = 0.12 if not mapped_urls else 0.20
    elif not mapped_urls:
        aligned = 0.08
    else:
        url_quality = 1.0 - min(1.0, broken_url_like_count(mapped_urls) / max(1, len(mapped_urls)))
        source_diversity = min(1.0, len(mapped_urls) / 6.0)
        has_reference_section = bool(
            re.search(
                r"^#{1,3}\s*(references?|sources?|bibliography|参考|来源|参考文献)",
                text or "",
                re.IGNORECASE | re.MULTILINE,
            )
        )
        aligned = (
            0.05
            + 0.40 * mapping_integrity
            + 0.20 * claim_coverage
            + 0.15 * url_quality
            + 0.10 * source_diversity
            + (0.10 if has_reference_section else 0.0)
        )
        # A long References dump with few or no in-text uses is not evidence
        # fidelity. Keep this a soft penalty so sparse but valid reports are
        # still learnable by GRPO.
        orphan_ratio = float(citation_details.get("orphan_mapping_ratio", 0.0))
        aligned -= min(0.25, 0.25 * orphan_ratio)

    strength = clamp(_env_float("COSIGHT_CITATION_ALIGNMENT_STRENGTH", 0.75))
    return clamp((1.0 - strength) * legacy + strength * clamp(aligned))


def _url_host(url: str) -> str:
    match = re.match(r"https?://([^/?#]+)", url or "", re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).lower().removeprefix("www.")


def _local_evidence_items(extra_info: Mapping[str, Any], ground_truth: Any = None) -> List[Mapping[str, Any]]:
    raw = extra_info.get("local_evidence") or extra_info.get("evidence")
    if not raw and isinstance(ground_truth, Mapping):
        raw = ground_truth.get("local_evidence") or ground_truth.get("evidence")

    items: List[Mapping[str, Any]] = []
    for item in _as_list(raw):
        if isinstance(item, Mapping):
            items.append(item)
        elif item:
            items.append({"text": str(item)})
    return items


def _local_evidence_details(
    text: str,
    local_evidence: List[Mapping[str, Any]],
    citation_details: Mapping[str, Any],
) -> Dict[str, Any]:
    evidence_urls: List[str] = []
    source_refs: List[str] = []
    titles: List[str] = []
    for item in local_evidence:
        evidence_urls.extend(str(url) for url in _as_list(item.get("urls")) if url)
        for key in ("source_ref", "path", "url"):
            if item.get(key):
                source_refs.append(str(item.get(key)))
        if item.get("title"):
            titles.append(str(item.get("title")))

    evidence_urls = sorted(set(_normalize_url(url) for url in evidence_urls if _normalize_url(url)))
    source_refs = sorted(set(source_refs))
    hosts = sorted(set(host for host in (_url_host(url) for url in evidence_urls) if host))
    lower = (text or "").lower()
    report_urls = sorted(set(_normalize_url(url) for url in extract_urls(text) if _normalize_url(url)))
    exact_url_hits = sorted(set(evidence_urls).intersection(report_urls))
    host_hits = sorted(host for host in hosts if host in lower)
    title_hits = sorted(title for title in titles if title and len(title) >= 4 and title.lower() in lower)
    mapped_urls = sorted(set(citation_details.get("mapped_urls") or []))
    supported_mapped_urls = sorted(set(mapped_urls).intersection(evidence_urls))
    unsupported_mapped_urls = sorted(set(mapped_urls).difference(evidence_urls))
    return {
        "item_count": len(local_evidence),
        "url_count": len(evidence_urls),
        "host_count": len(hosts),
        "exact_url_hits": exact_url_hits,
        "host_hits": host_hits,
        "title_hits": title_hits,
        "source_refs": source_refs,
        "mapped_urls": mapped_urls,
        "supported_mapped_urls": supported_mapped_urls,
        "unsupported_mapped_urls": unsupported_mapped_urls,
        "report_url_precision": len(supported_mapped_urls) / max(1, len(mapped_urls)) if mapped_urls else 0.0,
        "evidence_url_coverage": (
            min(1.0, len(supported_mapped_urls) / max(1, min(6, len(evidence_urls))))
            if evidence_urls
            else 0.0
        ),
        "mapping_integrity": float(citation_details.get("mapping_integrity", 0.0)),
        "citation_count": int(citation_details.get("citation_count", 0)),
        "claim_citation_coverage": float(citation_details.get("claim_citation_coverage", 0.0)),
        "orphan_mapping_ratio": float(citation_details.get("orphan_mapping_ratio", 0.0)),
    }


def _legacy_local_evidence_score(text: str, details: Mapping[str, Any]) -> float:
    if not details.get("item_count"):
        return 0.50

    url_count = int(details.get("url_count", 0))
    host_count = int(details.get("host_count", 0))
    exact_hits = len(details.get("exact_url_hits") or [])
    host_hits = len(details.get("host_hits") or [])
    title_hits = len(details.get("title_hits") or [])
    source_section = bool(
        re.search(
            r"^#{1,3}\s*(references?|sources?|bibliography|\u53c2\u8003|\u6765\u6e90|\u53c2\u8003\u6587\u732e)",
            text or "",
            re.IGNORECASE | re.MULTILINE,
        )
    )

    if url_count:
        score = 0.28
        score += 0.42 * min(1.0, host_hits / max(1, min(3, host_count)))
        score += 0.18 * min(1.0, exact_hits / max(1, min(2, url_count)))
        score += 0.07 if source_section else 0.0
        score += 0.05 * min(1.0, title_hits / 2)
    else:
        score = 0.42 + 0.30 * min(1.0, title_hits / 2)
        score += 0.08 if source_section else 0.0
    return clamp(score)


def _local_evidence_score(text: str, details: Mapping[str, Any]) -> float:
    if not details.get("item_count"):
        return 0.50

    legacy = _legacy_local_evidence_score(text, details)
    url_count = int(details.get("url_count", 0))
    mapped_urls = list(details.get("mapped_urls") or [])
    if not url_count:
        aligned = 0.50
    elif not mapped_urls:
        aligned = 0.08
    else:
        claim_coverage = float(details.get("claim_citation_coverage", 0.0))
        aligned = (
            0.05
            + 0.30 * float(details.get("report_url_precision", 0.0))
            + 0.15 * float(details.get("evidence_url_coverage", 0.0))
            + 0.20 * float(details.get("mapping_integrity", 0.0))
            + 0.25 * claim_coverage
        )
        if re.search(
            r"^#{1,3}\s*(references?|sources?|bibliography|参考|来源|参考文献)",
            text or "",
            re.IGNORECASE | re.MULTILINE,
        ):
            aligned += 0.05
        aligned -= min(0.25, 0.25 * float(details.get("orphan_mapping_ratio", 0.0)))

        # Exact URL copying alone is insufficient: require the report to use
        # those mapped references next to claims. This specifically blocks the
        # high-reward/low-EVI failure seen as 100+ orphan reference entries.
        if int(details.get("citation_count", 0)) == 0:
            aligned = min(aligned, 0.20)
        elif float(details.get("mapping_integrity", 0.0)) == 0.0:
            aligned = min(aligned, 0.30)

    strength = clamp(_env_float("COSIGHT_CITATION_ALIGNMENT_STRENGTH", 0.75))
    return clamp((1.0 - strength) * legacy + strength * clamp(aligned))


def _combined_evidence_factuality_score(
    text: str,
    citation_details: Mapping[str, Any],
    local_evidence_details: Mapping[str, Any],
) -> Dict[str, float]:
    """Score evidence fidelity with citation integrity and optional local evidence.

    The older evidence heuristic rewarded citation-looking surface form too
    easily.  Keep it inside ``_citation_aligned_evidence_factuality_score`` for
    continuity, but make the score depend on resolvable citation mappings and,
    when a local evidence pack is available, precision/coverage against that
    pack.
    """

    citation_aligned = _citation_aligned_evidence_factuality_score(text, citation_details)
    local_score = _local_evidence_score(text, local_evidence_details)
    if local_evidence_details.get("item_count"):
        local_strength = clamp(_env_float("COSIGHT_LOCAL_EVIDENCE_STRENGTH", 0.45))
        combined = (1.0 - local_strength) * citation_aligned + local_strength * local_score
    else:
        combined = citation_aligned
    return {
        "combined": clamp(combined),
        "citation_aligned": clamp(citation_aligned),
        "local_evidence": clamp(local_score),
        "legacy": _evidence_factuality_score(text),
    }


def _task_following_score(text: str, task_prompt: str, extra_info: Mapping[str, Any]) -> float:
    headings = len(re.findall(r"^#{1,4}\s+\S", text or "", re.MULTILINE))
    bullets = len(re.findall(r"^\s*(?:[-*+]|\d+[.)\u3001])\s+\S", text or "", re.MULTILINE))
    tables = len(re.findall(r"^\s*\|.+\|\s*$", text or "", re.MULTILINE))
    score = 0.45 + min(0.20, headings * 0.03) + min(0.12, bullets * 0.01) + min(0.08, tables * 0.02)

    prompt = task_prompt or ""
    expected_format = str(extra_info.get("expected_format") or "")
    prompt_and_format = f"{prompt}\n{expected_format}"
    prompt_lower = prompt_and_format.lower()
    if "markdown" in prompt_lower and headings == 0:
        score -= 0.12
    if re.search(r"table|\u8868\u683c|\u8868\u5f62", prompt_and_format, re.IGNORECASE) and tables == 0:
        score -= 0.18
    if re.search(r"\u7f16\u53f7|numbered|bullet|\u5217\u51fa|list", prompt_and_format, re.IGNORECASE) and bullets < 2:
        score -= 0.12
    if re.search(r"3\s*(?:categories|types|\u7c7b|\u4e2a)", prompt_and_format, re.IGNORECASE):
        three_hits = len(re.findall(r"^\s*(?:[123][.)\u3001]|\u4e00[.)\u3001]|\u4e8c[.)\u3001]|\u4e09[.)\u3001])", text or "", re.MULTILINE))
        score += 0.08 if three_hits >= 3 else -0.10
    if re.search(r"2\s*(?:possible|candidates?|\u4e2a|\u79cd)|two possible", prompt_and_format, re.IGNORECASE):
        score += 0.06 if len(re.findall(r"hypothesis|candidate|\u5047\u8bbe|\u5019\u9009|\u53ef\u80fd", text or "", re.IGNORECASE)) >= 2 else -0.08
    if re.search(
        r"\u77ed\u7b54|\u4e00\u5c0f\u6bb5|under\s+\d+\s+(?:chars|characters|words)|"
        r"\u5b57\u4ee5\u5185|\u4e0d\u8d85\u8fc7\s*\d+\s*\u5b57",
        prompt_and_format,
        re.IGNORECASE,
    ) and len(text) > 3500:
        score -= 0.18

    if re.search(r"[\u4e00-\u9fff]", prompt):
        zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
        latin_words = len(re.findall(r"\b[A-Za-z]{3,}\b", text or ""))
        if zh_chars < latin_words:
            score -= 0.10
    return clamp(score)


def _quantitative_reasoning_score(text: str, task_prompt: str) -> float:
    needs_quant = bool(
        re.search(
            r"quant|numeric|calculate|ratio|percent|trend|unit|rank|compare|"
            r"\u5b9a\u91cf|\u6570\u91cf|\u8ba1\u7b97|\u6bd4\u4f8b|\u767e\u5206\u6bd4|"
            r"\u540c\u6bd4|\u73af\u6bd4|\u5355\u4f4d|\u6392\u5e8f|\u6bd4\u8f83|"
            r"\u5cf0\u503c|\u6700\u5927|\u6700\u5c0f",
            task_prompt or "",
            re.IGNORECASE,
        )
    )
    numeric_mentions = len(
        re.findall(
            r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?\s*(?:%|km|kg|gb|mb|usd|rmb|yuan|"
            r"kwh|mwh|mw|gw|\u5143|\u5e74|\u6708|\u65e5)?",
            text or "",
            re.IGNORECASE,
        )
    )
    formula_terms = len(
        re.findall(
            r"=|ratio|percent|\u540c\u6bd4|\u73af\u6bd4|\u589e\u957f|\u4e0b\u964d|"
            r"increase|decrease|per\s+cent",
            text or "",
            re.IGNORECASE,
        )
    )
    uncertainty_terms = len(
        re.findall(
            r"insufficient data|cannot calculate|cannot confirm|"
            r"\u65e0\u6cd5\u8ba1\u7b97|\u65e0\u6cd5\u786e\u8ba4|\u6570\u636e\u4e0d\u8db3|\u4e0d\u786e\u5b9a",
            text or "",
            re.IGNORECASE,
        )
    )
    if needs_quant:
        score = 0.30 + min(0.35, numeric_mentions * 0.015) + min(0.20, formula_terms * 0.03)
        if uncertainty_terms:
            score += 0.05
    else:
        score = 0.55 + min(0.12, numeric_mentions * 0.004)
    return clamp(score)


def _report_quality_score(text: str, task_score: float, repeat_penalty: float) -> float:
    headings = len(re.findall(r"^#{1,4}\s+\S", text or "", re.MULTILINE))
    paragraphs = len(re.split(r"\n\s*\n+", text.strip())) if text.strip() else 0
    contradiction_markers = len(
        re.findall(
            r"contradict|inconsistent|"
            r"\u81ea\u76f8\u77db\u76fe|\u524d\u540e\u77db\u76fe|\u4e0d\u4e00\u81f4",
            text or "",
            re.IGNORECASE,
        )
    )
    uncertainty_markers = len(
        re.findall(
            r"uncertain|limitation|caveat|"
            r"\u4e0d\u786e\u5b9a|\u5c40\u9650|\u8bc1\u636e\u4e0d\u8db3",
            text or "",
            re.IGNORECASE,
        )
    )
    structure = min(0.25, headings * 0.035) + min(0.12, paragraphs * 0.004)
    score = 0.38 + 0.30 * task_score + structure - repeat_penalty * 0.75
    score -= min(0.20, contradiction_markers * 0.08)
    score += min(0.05, uncertainty_markers * 0.01)
    if len(text) > 220000:
        score -= 0.12
    return clamp(score)


def _collect_tool_names(extra_info: Mapping[str, Any]) -> List[str]:
    names: List[str] = []

    def visit(value: Any) -> None:
        if len(names) > 1000:
            return
        if isinstance(value, Mapping):
            possible = value.get("tool_name") or value.get("name") or value.get("function")
            if isinstance(possible, Mapping):
                possible = possible.get("name")
            if possible and isinstance(possible, str):
                names.append(possible)
            for nested_key in ("tool_calls", "tools", "tool_events", "messages", "workflow_tools"):
                if nested_key in value:
                    visit(value[nested_key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value:
            names.append(value)

    for key in ("tool_names", "tools", "tool_events", "workflow_tools", "trace_messages"):
        visit(extra_info.get(key))
    return names


def _workflow_score(extra_info: Mapping[str, Any]) -> float:
    names = set(_collect_tool_names(extra_info))
    if not names:
        return 0.50
    score = 0.30
    if "ask_question_about_image" in names:
        score += 0.16
    if "file_saver" in names:
        score += 0.14
    if "generate_markdown_report" in names:
        score += 0.20
    if "mark_step" in names:
        score += 0.06
    research_tools = {
        "serper_search",
        "tavily_search",
        "search",
        "image_search",
        "fetch_website_content",
        "fetch_website_content_with_images",
        "file_read",
        "extract_document_content",
        "execute_code",
    }
    if names.intersection(research_tools):
        score += 0.12
    return clamp(score)


def classify_benchmark_reason(reason: str) -> Dict[str, Any]:
    """Classify a judge reason into hard-failure tags used by the scorer."""

    reason = reason or ""
    specs = {
        "wrong_visual_identity": [
            r"wrong visual identity",
            r"misidentif",
            r"misreads?.*image",
            r"incorrectly identifies",
            r"fails? to .*visual",
            r"\u56fe\u50cf\u8bef\u8bfb|\u4e3b\u4f53\u8bc6\u522b\u9519\u8bef|\u770b\u9519",
        ],
        "fabricated_source_data": [
            r"fabricated",
            r"invents?",
            r"hallucinat",
            r"non-existent",
            r"false presence",
            r"future publication dates?",
            r"\u4f2a\u9020|\u7f16\u9020|\u5e7b\u89c9|\u4e0d\u5b58\u5728",
        ],
        "core_quant_error": [
            r"calculation error",
            r"numerical error",
            r"inverts?.*ratio",
            r"wrong time period",
            r"\u91cf\u5316\u9519\u8bef|\u8ba1\u7b97\u9519\u8bef|\u5355\u4f4d\u9519\u8bef|\u65f6\u95f4\u8303\u56f4\u9519\u8bef",
        ],
        "format_instruction_fail": [
            r"format violates",
            r"fails? to adhere",
            r"ignores? the requested",
            r"hard requirement",
            r"\u683c\u5f0f.*\u9519\u8bef|\u8fdd\u53cd.*\u8981\u6c42|\u672a\u9075\u5b88",
        ],
        "repetition_collapse": [
            r"excessive repetition",
            r"repeated headings",
            r"copy-paste",
            r"repetitive filler",
            r"\u91cd\u590d\u5d29\u574f|\u5927\u91cf\u91cd\u590d|\u62a5\u544a\u5d29\u574f",
        ],
    }
    tags = [tag for tag, patterns in specs.items() if any(re.search(pattern, reason, re.IGNORECASE | re.DOTALL) for pattern in patterns)]
    cap = min([FAILURE_CAPS[tag] for tag in tags], default=1.0)
    positive = bool(re.search(r"excellent|outstanding|near-perfect|high-quality|accurate", reason, re.IGNORECASE))
    failed = bool(tags or re.search(r"catastrophic|complete failure|fundamentally flawed|invalidates|misleading", reason, re.IGNORECASE))
    return {"tags": tags, "positive": positive, "failed": failed, "cap": cap}


def _judge_tags(extra_info: Mapping[str, Any]) -> List[str]:
    tags: List[str] = []
    for key in ("failure_tags", "tags"):
        for item in _as_list(extra_info.get(key)):
            if isinstance(item, str) and item in FAILURE_CAPS:
                tags.append(item)
    judge = extra_info.get("judge") or extra_info.get("reward_judge") or {}
    if isinstance(judge, str):
        try:
            judge = json.loads(judge)
        except Exception:
            judge = {}
    if isinstance(judge, Mapping):
        for key in FAILURE_CAPS:
            if judge.get(key):
                tags.append(key)
        for item in _as_list(judge.get("failure_tags") or judge.get("tags")):
            if isinstance(item, str) and item in FAILURE_CAPS:
                tags.append(item)
    reason = str(extra_info.get("judge_reason") or extra_info.get("reason") or "")
    if reason:
        tags.extend(classify_benchmark_reason(reason)["tags"])
    return sorted(set(tags))


def _hard_gates(
    text: str,
    task_prompt: str,
    extra_info: Mapping[str, Any],
    repetition_metrics: Mapping[str, Any],
    evidence_details: Mapping[str, Any],
) -> List[str]:
    reasons: List[str] = []
    stripped = text.strip()
    if len(stripped) < 100 or re.search(
        r"traceback|exception|tool crashed|error executing|"
        r"\u7a7a\u8f93\u51fa|\u5d29\u6e83|\u62a5\u9519",
        stripped,
        re.IGNORECASE,
    ):
        reasons.append("empty_or_crashed")
    if repetition_metrics.get("severe_repetition"):
        reasons.append("repetition_collapse")

    urls = evidence_details.get("urls", [])
    bad_urls = int(evidence_details.get("bad_urls", 0))
    future_years = int(evidence_details.get("future_years", 0))
    if (urls and bad_urls >= max(2, math.ceil(len(urls) * 0.5))) or future_years >= 3:
        reasons.append("fabricated_source_data")

    citation_details = evidence_details.get("citation_details") or {}
    local_evidence_details = evidence_details.get("local_evidence_details") or {}
    citation_count = int(citation_details.get("citation_count", 0))
    mapping_count = int(citation_details.get("mapping_count", 0))
    mapping_integrity = float(citation_details.get("mapping_integrity", 0.0))
    claim_coverage = float(citation_details.get("claim_citation_coverage", 0.0))
    orphan_ratio = float(citation_details.get("orphan_mapping_ratio", 0.0))

    # Citation-looking tokens with no resolvable reference list are a common
    # EVI failure: the report appears sourced, but claims cannot be checked.
    if citation_count >= 4 and not urls and mapping_count == 0:
        reasons.append("fabricated_source_data")
    if citation_count >= 40 and mapping_count >= 6 and mapping_integrity < 0.20:
        reasons.append("fabricated_source_data")
    if citation_count >= 80 and claim_coverage < 0.08:
        reasons.append("fabricated_source_data")
    if mapping_count >= 8 and orphan_ratio >= 0.60:
        reasons.append("fabricated_source_data")

    if local_evidence_details.get("item_count"):
        mapped_urls = list(local_evidence_details.get("mapped_urls") or [])
        report_url_precision = float(local_evidence_details.get("report_url_precision", 0.0))
        if len(mapped_urls) >= 4 and report_url_precision < 0.25:
            reasons.append("fabricated_source_data")

    prompt_and_format = f"{task_prompt}\n{extra_info.get('expected_format', '')}"
    if re.search(
        r"under\s+200\s+characters|within\s+200\s+characters|"
        r"200\s*\u5b57\u4ee5\u5185|\u4e0d\u8d85\u8fc7\s*200\s*\u5b57",
        prompt_and_format,
        re.IGNORECASE,
    ) and len(stripped) > 1200:
        reasons.append("format_instruction_fail")
    if re.search(r"table|\u8868\u683c", prompt_and_format, re.IGNORECASE) and not re.search(r"^\s*\|.+\|\s*$", text or "", re.MULTILINE):
        reasons.append("format_instruction_fail")

    reasons.extend(_judge_tags(extra_info))
    return sorted(set(reason for reason in reasons if reason in FAILURE_CAPS))


def _short_diagnosis(reasons: List[str], repetition_metrics: Mapping[str, Any], scores: Mapping[str, float]) -> str:
    reason_text = ", ".join(reasons) if reasons else "no hard gate"
    repeated = (
        f"dup_para={repetition_metrics.get('duplicate_paragraph_ratio', 0):.3f}, "
        f"max_para_repeat={repetition_metrics.get('max_repeated_paragraph_count', 0)}, "
        f"dup_sent={repetition_metrics.get('duplicate_sentence_ratio', 0):.3f}"
    )
    weakest = sorted(scores.items(), key=lambda item: item[1])[:2]
    weak_text = ", ".join(f"{key}={value:.2f}" for key, value in weakest)
    return f"{reason_text}; {repeated}; weakest: {weak_text}"


def evaluate_report(
    report_text: str,
    task_prompt: str = "",
    extra_info: Optional[Mapping[str, Any]] = None,
    ground_truth: Any = None,
) -> Dict[str, Any]:
    """Evaluate a final report and return a decomposed reward dictionary."""

    text = str(report_text or "")
    extra = dict(extra_info or {}) if isinstance(extra_info, Mapping) else {}
    ground_truth_text = _stringify(ground_truth)
    rubric = ""
    if isinstance(ground_truth, Mapping):
        rubric = str(ground_truth.get("rubric") or "")
    rubric = rubric or str(extra.get("rubric") or "")
    prompt = task_prompt or str(
        extra.get("task_prompt")
        or extra.get("body")
        or extra.get("query")
        or extra.get("question")
        or ground_truth_text
        or ""
    )
    prompt_for_scoring = f"{prompt}\n\nRubric:\n{rubric}" if rubric else prompt

    repetition_metrics = compute_repetition_metrics(text)
    expected_images = _extract_expected_images(extra)
    urls = extract_urls(text)
    citation_details = _citation_integrity_details(text)
    local_evidence = _local_evidence_items(extra, ground_truth)
    local_evidence_details = _local_evidence_details(text, local_evidence, citation_details)
    evidence_scores = _combined_evidence_factuality_score(text, citation_details, local_evidence_details)
    evidence_details = {
        "urls": urls,
        "bad_urls": broken_url_like_count(urls),
        "future_years": _future_year_count(text),
        "citation_details": citation_details,
        "local_evidence_details": local_evidence_details,
    }

    visual = _visual_grounding_score(text, expected_images)
    evidence = evidence_scores["combined"]
    task = _task_following_score(text, prompt_for_scoring, extra)
    quant = _quantitative_reasoning_score(text, prompt_for_scoring)
    repeat_penalty = float(repetition_metrics.get("P_repeat_soft", 0.0))
    report_quality = _report_quality_score(text, task, repeat_penalty)
    workflow = _workflow_score(extra)

    cost_penalty = min(0.08, max(0, len(text) - 120000) / 1200000)
    tool_errors = _as_list(extra.get("tool_errors")) + _as_list(extra.get("invalid_tool_calls"))
    invalid_tool_penalty = min(0.20, len(tool_errors) * 0.05)

    scores = {
        "R_visual_grounding": visual,
        "R_evidence_factuality": evidence,
        "R_task_following": task,
        "R_quantitative_reasoning": quant,
        "R_report_quality": report_quality,
        "R_workflow": workflow,
    }
    raw = sum(REWARD_WEIGHTS[key] * value for key, value in scores.items())
    raw -= cost_penalty + invalid_tool_penalty + repeat_penalty
    raw = clamp(raw)

    hard_gate_reasons = _hard_gates(text, prompt_for_scoring, extra, repetition_metrics, evidence_details)
    reward_cap = min([FAILURE_CAPS[reason] for reason in hard_gate_reasons], default=1.0)
    final_reward = clamp(min(raw, reward_cap))

    tool_names = sorted(set(_collect_tool_names(extra)))
    result = RewardResult(
        final_reward=final_reward,
        raw_reward=raw,
        reward_cap=reward_cap,
        hard_gate_reasons=hard_gate_reasons,
        R_visual_grounding=visual,
        R_evidence_factuality=evidence,
        R_task_following=task,
        R_quantitative_reasoning=quant,
        R_report_quality=report_quality,
        R_workflow=workflow,
        P_cost=cost_penalty,
        P_invalid_tool=invalid_tool_penalty,
        P_repeat_soft=repeat_penalty,
        repetition_metrics=dict(repetition_metrics),
        short_diagnosis=_short_diagnosis(hard_gate_reasons, repetition_metrics, scores),
        details={
            "expected_images": expected_images,
            "image_coverage": _image_coverage(text, expected_images),
            "url_count": len(urls),
            "bad_url_count": evidence_details["bad_urls"],
            "future_year_count": evidence_details["future_years"],
            "citation_integrity": citation_details,
            "local_evidence": local_evidence_details,
            "evidence_score_components": evidence_scores,
            "tool_names": tool_names,
            "tool_call_count": len(_collect_tool_names(extra)),
            "text_chars": len(text),
            "estimated_tokens": int(len(text) / 4),
        },
    )
    return result.to_dict()


def score_solution(solution_str: str, ground_truth: Any = None, extra_info: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Score a solution string with optional VERL ground truth and extra info."""

    if isinstance(ground_truth, Mapping):
        task_prompt = str(
            ground_truth.get("task_prompt")
            or ground_truth.get("body")
            or ground_truth.get("query")
            or ground_truth.get("question")
            or ""
        )
    else:
        task_prompt = _stringify(ground_truth)
    return evaluate_report(solution_str, task_prompt=task_prompt, extra_info=extra_info, ground_truth=ground_truth)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Mapping[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """VERL-compatible custom reward function."""

    result = score_solution(solution_str, ground_truth=ground_truth, extra_info=extra_info)
    result["data_source"] = data_source
    result["score"] = result["final_reward"]
    return result


__all__ = [
    "FAILURE_CAPS",
    "REWARD_WEIGHTS",
    "RewardResult",
    "broken_url_like_count",
    "classify_benchmark_reason",
    "compute_score",
    "evaluate_report",
    "extract_urls",
    "score_solution",
]
