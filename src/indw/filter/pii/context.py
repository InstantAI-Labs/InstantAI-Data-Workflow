from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from indw.filter.pii.config import PiiContextConfig

PiiContextLabel = Literal[
    'educational',
    'example',
    'documentation',
    'configuration',
    'credential_leak',
    'customer_data',
    'production_secret',
]

@dataclass
class PiiContextResult:
    context: PiiContextLabel = 'documentation'
    confidence: float = 0.0
    risk_multiplier: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'context': self.context,
            'confidence': round(self.confidence, 4),
            'risk_multiplier': round(self.risk_multiplier, 4),
        }

_EXAMPLE_MARKERS = re.compile(
    r'(?i)\b(example|sample|placeholder|dummy|test only|for demo|xxx+|000000)\b'
)
_DOC_MARKERS = re.compile(
    r'(?i)\b(documentation|configure|environment variable|field describes|parameter|'
    r'tutorial|guide|in this section|we delve|authentication plugin|implementation)\b'
)
_EDU_MARKERS = re.compile(
    r'(?i)\b(?:tutorial|textbook|lesson|course material|educational|how to|'
    r'we begin by|in this (?:section|chapter|article)|step \d+)\b'
)
_PRODUCTION_MARKERS = re.compile(
    r'(?i)\b(production|live key|real customer|patient record|ssn|social security)\b'
)
_CONFIG_MARKERS = re.compile(
    r'(?i)\b(config|yaml|json|env var|\.env|settings file|configuration block)\b'
)

class PiiContextAnalyzer:
    def __init__(self, config: PiiContextConfig):
        self.config = config

    def analyze(self, text: str, *, nearby: str = '') -> PiiContextResult:
        if not text:
            return PiiContextResult()
        sample = (nearby or text)[:2000]
        if _EXAMPLE_MARKERS.search(sample) and not _PRODUCTION_MARKERS.search(sample):
            return PiiContextResult('example', 0.88, 0.15)
        if _EDU_MARKERS.search(sample) and not _PRODUCTION_MARKERS.search(sample):
            return PiiContextResult('educational', 0.86, 0.16)
        if _DOC_MARKERS.search(sample) and not _PRODUCTION_MARKERS.search(sample):
            return PiiContextResult('documentation', 0.82, 0.2)
        if _CONFIG_MARKERS.search(sample) and not _PRODUCTION_MARKERS.search(sample):
            return PiiContextResult('configuration', 0.8, 0.22)
        if _PRODUCTION_MARKERS.search(sample):
            return PiiContextResult('production_secret', 0.9, 1.15)
        if re.search(r'(?i)\b(credential|api[_ ]?key|password|secret|token leak)\b', sample):
            return PiiContextResult('credential_leak', 0.78, 1.05)
        if re.search(r'(?i)\b(customer|patient|user record|account holder)\b', sample):
            return PiiContextResult('customer_data', 0.75, 1.1)
        return PiiContextResult('documentation', 0.65, 0.85)
