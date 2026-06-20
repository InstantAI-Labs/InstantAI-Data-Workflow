from __future__ import annotations

import logging
import math
import re
from typing import Protocol

from indw.filter.language.config import DetectorConfig

logger = logging.getLogger(__name__)
_WS_RE = re.compile(r'\s+')

class LanguageDetectorBackend(Protocol):
    def predict_distribution(self, text: str) -> dict[str, float]: ...

def _normalize_label(label: str) -> str:
    code = str(label or '').strip().lower()
    if code.startswith('__label__'):
        code = code[9:]
    if len(code) > 3 and '_' in code:
        code = code.split('_', 1)[0]
    return code[:2] if len(code) >= 2 else code

def _clean_text(text: str) -> str:
    return _WS_RE.sub(' ', text or '').strip()

def _symbol_heavy(text: str) -> bool:
    sample = text or ''
    if len(sample) < 4:
        return False
    letters = sum(1 for ch in sample if ch.isalpha())
    if letters >= max(8, len(sample) * 0.12):
        return False
    symbols = sum(1 for ch in sample if not ch.isspace() and not ch.isalnum())
    return symbols >= max(6, int(len(sample) * 0.55))

class LangIdBackend:
    _TOP_K = 8

    def __init__(self) -> None:
        self._rank = None
        try:
            import langid

            self._rank = langid.rank
        except Exception as exc:
            logger.warning('langid backend unavailable: %s', exc)

    def predict_distribution(self, text: str, *, cleaned: bool = False) -> dict[str, float]:
        if not self._rank:
            return {}
        if cleaned:
            sample = text or ''
        else:
            sample = text.strip() if text else ''
        if not sample:
            return {}
        ranked = self._rank(sample)[: self._TOP_K]
        if not ranked:
            return {}
        logs = [float(lp) for _, lp in ranked]
        peak = max(logs)
        weights = [math.exp(lp - peak) for lp in logs]
        total = sum(weights) or 1.0
        out: dict[str, float] = {}
        for (label, _), weight in zip(ranked, weights):
            code = _normalize_label(label)
            if not code:
                continue
            out[code] = out.get(code, 0.0) + weight / total
        return out

class FastLanguageDetector:

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._backend = LangIdBackend()

    def predict_distribution(self, text: str, *, min_chars: int | None = None) -> dict[str, float]:
        sample = _clean_text(text)[: self.config.max_chars]
        floor = self.config.min_text_chars if min_chars is None else min_chars
        if len(sample) < floor:
            return {}
        if _symbol_heavy(sample):
            return {'und': 1.0}
        return self._backend.predict_distribution(sample, cleaned=True)
