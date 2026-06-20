from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any
from indw.clean.artifact.evidence_util import (
    _CITATION_FMT, _CODE_FENCE, _COPULA_DEF, _EPS, _EXCLAIM_LINE, _FACT_REL,
    _NUMERIC_TOKEN, _QA_LINE, _STEP_LINE, _STRUCTURED, _TABLE_ROW, _URL, _WORD,
    _YEAR, _lines, _mean, _saturate, _spread, _token_estimate,
)


@dataclass
class RawDocumentFeatures:
    text: str
    words: list[str]
    lines: list[str]
    word_count: int
    line_count: int
    char_count: int
    token_estimate: int
    numeric_token_ratio: float = 0.0
    uppercase_token_ratio: float = 0.0
    url_char_ratio: float = 0.0
    fence_char_ratio: float = 0.0
    table_line_ratio: float = 0.0
    structured_line_ratio: float = 0.0
    citation_hits: int = 0
    year_hits: int = 0
    copula_def_hits: int = 0
    step_line_hits: int = 0
    qa_line_hits: int = 0
    exclaim_line_ratio: float = 0.0
    avg_line_len: float = 0.0
    line_len_cv: float = 0.0
    sentence_count: int = 0
    contact_token_ratio: float = 0.0
    nav_line_ratio: float = 0.0
    schedule_token_ratio: float = 0.0
    anchor_density: float = 0.0
    first_person_ratio: float = 0.0
    uniform_line_ratio: float = 0.0
    fact_relation_hits: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in self.__dict__.items()
            if k not in ('text', 'words', 'lines')
        }

_SHARED_EXTRACTOR: DocumentFeatureExtractor | None = None


def shared_feature_extractor() -> DocumentFeatureExtractor:
    global _SHARED_EXTRACTOR
    if _SHARED_EXTRACTOR is None:
        _SHARED_EXTRACTOR = DocumentFeatureExtractor()
    return _SHARED_EXTRACTOR


_SENT_SPLIT = re.compile(r'[.!?]+')


class PopulationAdaptiveScaler:
    @staticmethod
    def rate(count: float, *totals: float) -> float:
        totals = [max(t, 1.0) for t in totals]
        peers = [count / t for t in totals]
        return _saturate(count, peers)

    @staticmethod
    def short_doc_boundary(raw: RawDocumentFeatures) -> float:
        peers = [
            float(raw.sentence_count),
            float(raw.line_count),
            raw.word_count / max(raw.sentence_count, 1),
        ]
        span = _saturate(raw.avg_line_len, [raw.avg_line_len, raw.line_len_cv])
        return max(1.0, _mean(peers) * span)

    @staticmethod
    def capacity(raw: RawDocumentFeatures, density: float, strength: float) -> float:
        peers = [
            density,
            strength,
            raw.sentence_count / max(raw.word_count, 1),
            raw.structured_line_ratio + raw.fence_char_ratio,
        ]
        unit = _saturate(_mean(peers), peers)
        return max(1.0, raw.word_count * unit / max(density, _EPS))

class DocumentFeatureExtractor:
    def extract(self, text: str) -> RawDocumentFeatures:
        if not text or not text.strip():
            return RawDocumentFeatures('', [], [], 0, 0, 0, 1)
        from indw.clean.artifact.evidence_cache import get_raw_feature_cache, raw_feature_cache_key

        key = raw_feature_cache_key(text)
        if key is not None:
            cache = get_raw_feature_cache()
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = self._extract_impl(text)
            cache.put(key, result)
            return result
        return self._extract_impl(text)

    def _extract_impl(self, text: str) -> RawDocumentFeatures:
        words = _WORD.findall(text)
        lines = _lines(text)
        wc = len(words)
        lc = max(len(lines), 1)
        cc = len(text)
        numeric = sum(1 for w in words if _NUMERIC_TOKEN.search(w))
        upper = sum(1 for w in words if w[:1].isupper() and len(w) > 1)
        url_chars = sum(len(m.group(0)) for m in _URL.finditer(text))
        fences = _CODE_FENCE.findall(text)
        fence_chars = sum(len(f) for f in fences)
        table_lines = sum(1 for ln in lines if _TABLE_ROW.match(ln))
        struct_lines = len(_STRUCTURED.findall(text))
        line_lens = [len(ln) for ln in lines] or [0]
        avg_ll = sum(line_lens) / lc
        var = sum((x - avg_ll) ** 2 for x in line_lens) / lc
        cv = math.sqrt(var) / avg_ll if avg_ll > 0 else 0.0
        sents = [p for p in _SENT_SPLIT.split(text) if p.strip()]
        contact = sum(
            1 for w in words
            if '@' in w or sum(ch.isdigit() for ch in w) >= max(7, len(w) // 2)
        )
        nav_lines = sum(
            1 for ln in lines
            if avg_ll > 0
            and len(ln.split()) <= max(3, int(avg_ll / max(len(_WORD.findall(ln)), 1)))
            and sum(ln.count(c) for c in '|>»/\\-:') >= 1
            and len(ln) <= avg_ll * (1.0 + cv)
        )
        schedule = sum(
            1 for w in words
            if ':' in w and any(ch.isdigit() for ch in w)
        )
        uniform = sum(
            1 for ln in line_lens
            if avg_ll > 0 and abs(ln - avg_ll) <= avg_ll * max(cv, _EPS)
        )
        fp = sum(1 for w in words if len(w) <= 2 and w.isalpha() and w.lower() == w)
        anchor = (len(_CITATION_FMT.findall(text)) + len(set(_YEAR.findall(text)))) / max(wc, 1)
        return RawDocumentFeatures(
            text=text,
            words=words,
            lines=lines,
            word_count=wc,
            line_count=lc,
            char_count=cc,
            token_estimate=_token_estimate(text, wc),
            numeric_token_ratio=numeric / max(wc, 1),
            uppercase_token_ratio=upper / max(wc, 1),
            url_char_ratio=url_chars / max(cc, 1),
            fence_char_ratio=fence_chars / max(cc, 1),
            table_line_ratio=table_lines / lc,
            structured_line_ratio=struct_lines / lc,
            citation_hits=len(_CITATION_FMT.findall(text)),
            year_hits=len(set(_YEAR.findall(text))),
            copula_def_hits=len(_COPULA_DEF.findall(text)),
            step_line_hits=len(_STEP_LINE.findall(text)),
            qa_line_hits=len(_QA_LINE.findall(text)),
            exclaim_line_ratio=sum(1 for ln in lines if _EXCLAIM_LINE.match(ln)) / lc,
            avg_line_len=avg_ll,
            line_len_cv=cv,
            sentence_count=max(len(sents), 1),
            contact_token_ratio=contact / max(wc, 1),
            nav_line_ratio=nav_lines / lc,
            schedule_token_ratio=schedule / max(wc, 1),
            anchor_density=anchor,
            first_person_ratio=fp / max(wc, 1),
            uniform_line_ratio=uniform / lc,
            fact_relation_hits=len(_FACT_REL.findall(text)),
        )

@dataclass
class SemanticFeatureBundle:
    raw: RawDocumentFeatures
    filters: Any
    signals: Any | None = None

    def quality_signals(self) -> Any:
        if self.signals is None:
            from indw.filter.score.signals import compute_signals
            self.signals = compute_signals(
                self.raw.text,
                filters=self.filters,
                words=self.raw.words,
                lines=self.raw.lines,
            )
        return self.signals
