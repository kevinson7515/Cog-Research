"""Deterministic audit for extracted claims and source-indexed atomic facts."""

from dataclasses import dataclass
from decimal import Decimal
import re
import unicodedata


_NUMBER = re.compile(r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(\s*[%a-z\u4e00-\u9fff/][%a-z0-9\u4e00-\u9fff/]*)?")
_DATE = re.compile(r"(\d{4})(?:年|[-/])(\d{1,2})(?:月|[-/])(\d{1,2})日?")


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def value_key(value):
    """Normalize signed scalars and dates without discarding units or polarity."""
    value = normalized_text(value).replace("−", "-")
    if date := _DATE.fullmatch(value):
        return "date", tuple(int(part) for part in date.groups())
    if number := _NUMBER.fullmatch(value):
        return "number", Decimal(number[1].replace(",", "")), (number[2] or "").strip()
    return "text", value


@dataclass(frozen=True)
class EvidenceFact:
    source: str
    subject: str
    field: str
    value: str
    exclusive: bool = False


@dataclass(frozen=True)
class Claim:
    subject: str
    field: str
    value: str
    polarity: bool
    cited_sources: tuple[str, ...]


class EvidenceIndex:
    def __init__(self, facts):
        self.fields = {}
        self.sources = set()
        for fact in facts:
            key = normalized_text(fact.subject), normalized_text(fact.field)
            self.fields.setdefault(key, []).append(fact)
            self.sources.add(fact.source)

    def verify(self, claim):
        if not claim.cited_sources:
            return "no_citation"
        key = normalized_text(claim.subject), normalized_text(claim.field)
        field = self.fields.get(key, [])
        relevant = [fact for fact in field if fact.source in claim.cited_sources]
        if not relevant:
            if not self.sources.intersection(claim.cited_sources):
                return "source_not_indexed"
            return "wrong_source" if field else "insufficient_evidence"

        target = value_key(claim.value)
        supports, contradicts = False, False
        for fact in relevant:
            if value_key(fact.value) == target:
                supports |= claim.polarity
                contradicts |= not claim.polarity
            elif fact.exclusive:
                supports |= not claim.polarity
                contradicts |= claim.polarity
        if supports and contradicts:
            return "conflicting_evidence"
        if contradicts:
            return "contradicted"
        return "supported" if supports else "insufficient_evidence"
