from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from indw.filter.pii.config import NerConfig

_EMAIL_STRUCT = re.compile(
    r'\b([A-Za-z0-9._%+\-]{1,64})@([A-Za-z0-9.\-]{1,253}\.[A-Za-z]{2,24})\b'
)
_PHONE_STRUCT = re.compile(
    r'\b(?:\+?[\d\u0660-\u0669]{1,4}[\s.\-]?)?'
    r'(?:\(?[\d\u0660-\u0669]{2,4}\)?[\s.\-]?)?'
    r'[\d\u0660-\u0669]{3,4}[\s.\-]?[\d\u0660-\u0669]{3,4}(?:[\s.\-]?[\d\u0660-\u0669]{2,6})?\b'
)
_FINANCIAL_STRUCT = re.compile(r'\b(?:\d[\d\s\-]{11,18}\d)\b')
_ACCOUNT_STRUCT = re.compile(r'\b[A-Za-z0-9][A-Za-z0-9_\-]{7,31}\b')
_ASSIGNMENT = re.compile(
    r'(?i)([a-z][a-z0-9_\-]{0,24})\s*[:=]\s*([^\s\'"]{8,})'
)
_UUID = re.compile(
    r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
    re.I,
)
_HEX_REF = re.compile(r'\b[0-9a-fA-F]{24,}\b')

@dataclass
class ExtractedEntity:
    type: str
    text: str
    start: int
    end: int
    confidence: float
    source: str = 'structural'

    def to_dict(self) -> dict[str, Any]:
        return {
            'type': self.type,
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'confidence': round(self.confidence, 4),
            'source': self.source,
        }

@dataclass
class EntityExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {'entities': [e.to_dict() for e in self.entities]}

    def max_confidence(self) -> float:
        return max((e.confidence for e in self.entities), default=0.0)

    def entity_risk(self) -> float:
        if not self.entities:
            return 0.0
        return min(1.0, max(e.confidence for e in self.entities) * 0.85 + 0.05 * min(len(self.entities), 8))

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def _is_schema_template_email(email: str) -> bool:
    low = email.lower().strip('\'",;')
    local, sep, domain = low.partition('@')
    if not sep or not local or not domain:
        return False
    if domain.endswith('.tld') or domain.endswith('.invalid') or domain == 'domain.tld':
        return True
    if domain.startswith('example.') or '.example.' in f'.{domain}':
        return True
    if len(local) <= 12 and local.isalpha() and 'domain' in domain:
        return True
    return False

def _is_placeholder_value(val: str) -> bool:
    low = val.lower().strip('[]')
    if low in {'xxx', 'xxxx', 'redacted', 'redacted_sample', 'placeholder', 'sample', 'example'}:
        return True
    if re.fullmatch(r'[xX*#.\-]{6,}', val):
        return True
    if 'redacted' in low or 'placeholder' in low or 'sample' in low:
        return True
    return False

def _structural_confidence_email(local: str, domain: str) -> float:
    if len(local) < 1 or len(domain) < 4:
        return 0.0
    if '..' in local or '..' in domain:
        return 0.0
    ent = _shannon_entropy(local + domain)
    return min(1.0, 0.55 + ent / 8.0)

def _structural_confidence_phone(raw: str) -> float:
    digits = sum(c.isdigit() for c in raw)
    if digits < 10:
        return 0.0
    ratio = digits / max(len(raw), 1)
    return min(1.0, 0.5 + ratio * 0.45)

class StructuralEntityExtractor:

    def extract(self, text: str) -> list[ExtractedEntity]:
        found: list[ExtractedEntity] = []
        for m in _EMAIL_STRUCT.finditer(text):
            if _is_schema_template_email(m.group(0)):
                continue
            conf = _structural_confidence_email(m.group(1), m.group(2))
            if conf >= 0.5:
                found.append(
                    ExtractedEntity('EMAIL', m.group(0), m.start(), m.end(), conf, 'structural')
                )
        for m in _PHONE_STRUCT.finditer(text):
            raw = m.group(0)
            if _UUID.search(raw) or _HEX_REF.fullmatch(raw.replace('-', '')):
                continue
            conf = _structural_confidence_phone(raw)
            if conf >= 0.55:
                found.append(
                    ExtractedEntity('PHONE', raw, m.start(), m.end(), conf, 'structural')
                )
        for m in _FINANCIAL_STRUCT.finditer(text):
            digits = sum(c.isdigit() for c in m.group(0))
            if 13 <= digits <= 19:
                found.append(
                    ExtractedEntity(
                        'FINANCIAL_ID',
                        m.group(0),
                        m.start(),
                        m.end(),
                        min(1.0, 0.6 + digits / 20.0),
                        'structural',
                    )
                )
        for m in _ASSIGNMENT.finditer(text):
            label, val = m.group(1), m.group(2)
            if _is_placeholder_value(val):
                continue
            ent = _shannon_entropy(val)
            if ent >= 3.0 and len(val) >= 10:
                found.append(
                    ExtractedEntity(
                        'CREDENTIAL',
                        val,
                        m.start(2),
                        m.end(2),
                        min(1.0, 0.5 + ent / 6.0),
                        'assignment',
                    )
                )
        return _merge_spans(found)

def _merge_spans(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    if not entities:
        return []
    ordered = sorted(entities, key=lambda e: (e.start, -e.end))
    kept: list[ExtractedEntity] = []
    for ent in ordered:
        if kept and ent.start < kept[-1].end and ent.type == kept[-1].type:
            if ent.confidence > kept[-1].confidence:
                kept[-1] = ent
            continue
        kept.append(ent)
    return kept

class EntityExtractor:
    def __init__(self, config: NerConfig):
        del config
        self.structural = StructuralEntityExtractor()

    def extract(self, text: str) -> EntityExtractionResult:
        if not text:
            return EntityExtractionResult()
        return EntityExtractionResult(entities=_merge_spans(self.structural.extract(text)))
