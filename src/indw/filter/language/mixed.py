from __future__ import annotations

import re
from dataclasses import dataclass

from indw.filter.language.config import MixedLanguageConfig
from indw.filter.language.fast_detector import FastLanguageDetector
from indw.filter.language.script import scan_script_text
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?。！？])\s+|\n+')

@dataclass
class MixedLanguageResult:
    languages: dict[str, float]
    mixed_language: bool
    segments: int = 0

_CJK_FAMILY = frozenset({'cjk', 'hiragana_katakana'})

def _mixed_script_bucket(script: str) -> str:
    norm = script.strip().lower()
    if norm in _CJK_FAMILY:
        return 'cjk_family'
    return norm

def _script_segments(text: str, *, min_chars: int) -> list[tuple[str, str]]:
    return list(scan_script_text(text, segment_min_chars=min_chars).segments)

class MixedLanguageAnalyzer:
    def __init__(self, detector: FastLanguageDetector, config: MixedLanguageConfig):
        self.detector = detector
        self.config = config

    def analyze(
        self,
        text: str,
        *,
        script_segments: tuple[tuple[str, str], ...] | list[tuple[str, str]] | None = None,
    ) -> MixedLanguageResult:
        if not self.config.enabled or not text:
            dist = self.detector.predict_distribution(text)
            primary = max(dist, key=dist.get) if dist else 'und'
            return MixedLanguageResult({primary: 1.0} if primary else {}, False, 0)

        doc_dist = self.detector.predict_distribution(text)
        if not doc_dist:
            return MixedLanguageResult({}, False, 0)

        if script_segments is None:
            segments = _script_segments(text, min_chars=self.config.min_segment_chars)
        else:
            segments = list(script_segments)
        script_buckets = {
            _mixed_script_bucket(b)
            for _, b in segments
            if b not in ('other', 'emoji', 'common')
        }
        if len(script_buckets) < 2:
            return MixedLanguageResult(doc_dist, False, max(len(segments), 1))

        ranked_doc = sorted(doc_dist.values(), reverse=True)
        if ranked_doc and ranked_doc[0] >= 0.9 and script_buckets <= {'cjk_family'}:
            return MixedLanguageResult(doc_dist, False, max(len(segments), 1))

        weights: dict[str, float] = {}
        total = 0
        seg_min = max(2, self.config.min_segment_chars)
        for seg, _script in segments:
            dist = dict(self.detector.predict_distribution(seg, min_chars=seg_min))
            if not dist:
                continue
            w = max(len(seg), 1)
            total += w
            for lang, prob in dist.items():
                weights[lang] = weights.get(lang, 0.0) + prob * w
        if not weights or total <= 0:
            return MixedLanguageResult(doc_dist, True, len(segments))
        merged = {k: v / total for k, v in weights.items()}
        ranked = sorted(merged.values(), reverse=True)
        top = ranked[0] if ranked else 0.0
        second = ranked[1] if len(ranked) > 1 else 0.0
        active = sum(1 for p in merged.values() if p >= self.config.mixed_threshold)
        mixed = len(script_buckets) >= 2 or active >= 2 or (
            second >= self.config.mixed_threshold and top < self.config.dominance_threshold
        )
        return MixedLanguageResult(merged, mixed, len(segments))
