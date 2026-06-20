from __future__ import annotations

import math
import re
_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)

_CODE_FENCE = re.compile(r'```[\s\S]*?```', re.M)

_TABLE_ROW = re.compile(r'(?m)^\|.+\|$')

_STRUCTURED = re.compile(r'(?m)^(?:#{1,6}\s+|={2,}.+={2,})')

_CITATION_FMT = re.compile(r'(?i)(?:doi:\s*\S+|arxiv:\s*\S+|\[\d+\]|\(\d{4}[a-z]?\))')

_YEAR = re.compile(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b')

_URL = re.compile(r'https?://\S+|www\.\S+')

_QA_LINE = re.compile(r'(?im)^\s*(?:question|answer|reply|comment)\s*:')

_NUMERIC_TOKEN = re.compile(r'\d')

_COPULA_DEF = re.compile(r'(?i)\b\w+(?:\s+\w+){0,6}\s+(?:is|are|was|were)\s+(?:a|an|the|defined)\b')

_STEP_LINE = re.compile(r'(?im)^\s*(?:step\s*)?\d+[\.\):]\s+\S')

_EXCLAIM_LINE = re.compile(r'(?m)^[^.!?]{4,}[!?]\s*$')

_FACT_REL = re.compile(r'(?i)\b(?:equals?|converts?|produces?|fixes?|defines?|means?)\b')

_EPS = 1e-9

def _token_estimate(text: str, word_count: int) -> int:
    if word_count <= 0:
        return 1
    return max(1, int(len(text) / (len(text) / word_count)))

def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

def _spread(values: list[float]) -> float:
    if not values:
        return 0.0
    lo, hi = min(values), max(values)
    return (hi - lo) or (sum(values) / len(values))

def _saturate(value: float, peers: list[float]) -> float:
    if value <= 0:
        return 0.0
    baseline = sum(peers) / len(peers) if peers else value
    baseline = max(baseline, value * 0.25)
    return max(0.0, min(1.0, value / (value + baseline)))

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def peer_baseline(values: list[float]) -> float:
    if not values:
        return 0.0
    peers = list(values)
    positive = [v for v in peers if v > 0]
    if not positive:
        return 0.0
    if len(positive) == 1 and len(peers) == 1:
        return min(1.0, positive[0])
    if len(positive) == 1:
        return min(1.0, positive[0] * 0.85)
    return max(_saturate(v, peers) for v in positive)

def evidence_margin(utility: float, threshold: float, uncertainty: float) -> float:
    return utility - threshold * (1.0 - uncertainty)
