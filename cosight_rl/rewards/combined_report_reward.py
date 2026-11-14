"""Single-rollout claim/source verifier for the report evidence component."""

from collections import Counter, defaultdict
import json
import re
from typing import Mapping

from .claim_source_audit import Claim, EvidenceFact, EvidenceIndex, normalized_text, value_key


DECISIONS = {"supported": 1.0, "contradicted": 0.0, "wrong_source": 0.0}
LEXICAL_WEIGHT = 0.25
_MARKERS = re.compile(r"(?:[ \t]*\[\d+\])+")
_STRUCTURED = re.compile(
    r'(?P<subject>[^:：]+)[:：]\s*(?P<field>.+?)\s*(?P<op>!=|=)\s*[“「"](?P<value>[^”」"]+)[”」"]'
)
_IMAGE = r"(?:image|图像|图)\s+(?P<subject>[\w.-]+)"
_VALUE = r'[“「"](?P<value>[^”」"]+)[”」"]'
_OTHER = r'[“「"](?P<other>[^”」"]+)[”」"]'
_PAIRED = tuple(re.compile(pattern, re.I) for pattern in (
    rf"{_IMAGE}\s+gives\s+{_VALUE}\s+for\s+the\s+(?P<field>.+?)\s+and\s+rules out\s+{_OTHER}",
    rf"for\s+the\s+(?P<field>.+?),\s*{_IMAGE}\s+rules out\s+{_OTHER}\s+and\s+lists\s+{_VALUE}",
    rf"{_IMAGE}\s*显示[，,]\s*(?P<field>.+?)为{_VALUE}[，,]\s*可排除{_OTHER}",
    rf"以{_IMAGE}\s*为依据[，,]\s*可以排除{_OTHER}[，,]\s*并确认{_VALUE}是(?P<field>.+?)的记录",
))
_UNITS = re.compile(r"(?<=[.!?。！？])\s+|[。！？]|\n+")
_EN_SCALAR = re.compile(
    r"(?P<subject>[^,:;]{2,90}?)\s+(?:(?P<negation>does not have|did not have)|has|had)\s+(?:an?\s+)?"
    r"(?P<field>mass|weight|height|length|width|diameter|area|population|orbital inclination)"
    r"\s+of\s+(?P<value>[+-]?\d[\d,.]*(?:\s*[a-z%]+)?)", re.I
)
_ZH_SCALAR = re.compile(
    r"(?P<subject>[^，、；：。]{2,45}?)(?:的)?(?P<field>重量|质量|高度|长度|宽度|直径|面积|人口|轨道倾角)"
    r"(?P<negation>不)?(?:为|是)\s*(?P<value>[+-]?\d[\d,.]*(?:\s*[%a-zA-Z\u4e00-\u9fff]{1,4})?)"
)
_WORDS = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_NUMBERS = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_POLARITY = re.compile(r"\b(?:no|not|never|without|false|neither|cannot)\b|并非|没有|不是|不能|不曾|从未|未能")


def _statement(text):
    return normalized_text(text.strip(' \t.。!?！？“”"')).rstrip(".。!?！？")


def _statement_features(text):
    words = _WORDS.findall(text)
    width = 3 if re.search(r"[\u4e00-\u9fff]", text) else 2
    phrases = set(zip(*(words[offset:] for offset in range(width))))
    numbers = Counter(map(value_key, _NUMBERS.findall(text.replace("−", "-"))))
    return phrases, numbers, bool(_POLARITY.search(text))


def _has_lexical_match(text, features):
    phrases, numbers, negative = _statement_features(text)
    if len(phrases) < 5:
        return False
    for other, other_numbers, other_negative in features:
        if numbers != other_numbers or negative != other_negative:
            continue
        overlap = len(phrases & other)
        if overlap >= 5 and overlap / len(phrases) >= 0.55 and overlap / len(other) >= 0.45:
            return True
    return False


def _report_lines(report):
    fenced = False
    for raw in report.splitlines():
        line = raw.strip()
        if line.startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if not fenced and line and not line.startswith(("#", "|", "![")):
            yield line


def _clauses(lines, references):
    for line in lines:
        start = 0
        for marker in _MARKERS.finditer(line):
            chunks = _UNITS.split(line[start:marker.start()].strip().rstrip(".。!?！？"))
            for chunk in chunks[:-1]:
                yield chunk, ()
            ids = re.findall(r"\d+", marker[0])
            urls = [references.get(i, ()) for i in ids]
            sources = tuple(sorted({values[0] for values in urls})) if all(len(v) == 1 for v in urls) else ()
            yield chunks[-1], sources
            start = marker.end()
        for chunk in _UNITS.split(line[start:]):
            yield chunk, ()


def _structured_claims(text, sources):
    if match := _STRUCTURED.fullmatch(text):
        yield Claim(match["subject"], match["field"], match["value"], match["op"] == "=", sources)
        return
    for pattern in _PAIRED:
        if match := pattern.fullmatch(text):
            yield Claim(match["subject"], match["field"], match["value"], True, sources)
            yield Claim(match["subject"], match["field"], match["other"], False, sources)
            return
    match = _EN_SCALAR.fullmatch(text) or _ZH_SCALAR.fullmatch(text)
    if match:
        subject = re.sub(r"^(?:the\s+)", "", match["subject"], flags=re.I)
        yield Claim(subject, match["field"], match["value"], not match["negation"], sources)


def _evidence(raw, normalize_url):
    rows = json.loads(raw) if isinstance(raw, str) else raw
    if rows is None:
        rows = []
    if not isinstance(rows, (list, tuple)):
        raise ValueError("claim_source_evidence must be a list of source records")
    facts, snippets, complete = [], defaultdict(set), set()
    for item in rows:
        if not isinstance(item, Mapping) or not isinstance(item.get("url"), str):
            continue
        url = normalize_url(item["url"])
        if not re.fullmatch(r"https?://[^\s]+", url):
            continue
        if isinstance(item.get("text"), str):
            for sentence in _UNITS.split(item["text"]):
                statement = _statement(sentence)
                if not statement:
                    continue
                snippets[url].add(statement)
                claims = list(_structured_claims(sentence.strip().rstrip(".。!?！？"), (url,)))
                if len(claims) == 1 and claims[0].polarity:
                    claim = claims[0]
                    facts.append(EvidenceFact(url, claim.subject, claim.field, claim.value))
        checked_facts = []
        for fact in item.get("facts") or ():
            if isinstance(fact, Mapping) and all(isinstance(fact.get(k), str) and fact[k].strip()
                                                 for k in ("subject", "field", "value")):
                checked_facts.append(EvidenceFact(url, fact["subject"], fact["field"], fact["value"],
                                                  exclusive=fact.get("exclusive") is True))
        facts.extend(checked_facts)
        if checked_facts and item.get("complete") is True:
            complete.add(url)
    return EvidenceIndex(facts), snippets, complete


def score_combined_evidence(report, base, raw_evidence, reference_pattern, extract_urls, normalize_url):
    """Fill undecidable candidates with the original evidence component."""
    index, snippets, complete = _evidence(raw_evidence, normalize_url)
    features = {url: [_statement_features(text) for text in excerpts] for url, excerpts in snippets.items()}
    references, lines = defaultdict(set), []
    for line in _report_lines(report):
        match = reference_pattern.match(line)
        urls = extract_urls(match[2]) if match else []
        if urls:
            references[match[1]].update(map(normalize_url, urls))
        else:
            lines.append(line)
    references = {label: sorted(urls) for label, urls in references.items()}
    outcomes = {}
    for raw, sources in _clauses(lines, references):
        text = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", raw.strip()).strip(" .。!?！？")
        if not any(c.isalnum() for c in text):
            continue
        claims = list(_structured_claims(text, sources))
        for claim in claims:
            key = (normalized_text(claim.subject), normalized_text(claim.field), normalized_text(claim.value),
                   claim.polarity, sources)
            if key in outcomes:
                continue
            status = index.verify(claim)
            if status == "wrong_source":
                other = Claim(claim.subject, claim.field, claim.value, claim.polarity,
                              tuple(sorted(index.sources.difference(sources))))
                if not set(sources).issubset(complete) or index.verify(other) != "supported":
                    status = "insufficient_evidence"
            outcomes[key] = status
        if not claims:
            statement = _statement(text)
            # Very short snippets can be headings or shared fragments.
            substantive = (
                len(re.findall(r"[a-z0-9]+", statement)) >= 6
                or len(re.findall(r"[\u4e00-\u9fff]", statement)) >= 12
            )
            if substantive and any(statement in snippets.get(url, ()) for url in sources):
                status = "supported"
            elif sources and _has_lexical_match(statement, (f for url in sources for f in features.get(url, ()))):
                status = "lexical_match"
            else:
                status = "unverified"
            outcomes[(statement, sources)] = status
    counts = Counter(outcomes.values())
    decided = sum(counts[status] for status in DECISIONS)
    adjustment = sum(counts[status] * (score - base) for status, score in DECISIONS.items())
    adjustment += LEXICAL_WEIGHT * counts["lexical_match"] * (1.0 - base)
    combined = base + adjustment / max(1, len(outcomes))
    return combined, counts, decided, len(outcomes)
