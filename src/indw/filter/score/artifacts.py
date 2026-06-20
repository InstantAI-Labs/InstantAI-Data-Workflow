from __future__ import annotations

import unicodedata
from dataclasses import dataclass

@dataclass
class ArtifactSignalBundle:
    ocr_corruption: float = 0.0
    dict_ui: float = 0.0
    broken_unicode: float = 0.0
    scraping_remnant: float = 0.0
    format_break: float = 0.0
    contact_block: float = 0.0
    author_meta: float = 0.0
    news_meta: float = 0.0
    promotional: float = 0.0

    @property
    def severity(self) -> float:
        return min(
            1.0,
            self.ocr_corruption * 0.22
            + self.dict_ui * 0.14
            + self.broken_unicode * 0.12
            + self.scraping_remnant * 0.10
            + self.format_break * 0.08
            + self.contact_block * 0.14
            + self.author_meta * 0.08
            + self.news_meta * 0.06
            + self.promotional * 0.06,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            'ocr_corruption': round(self.ocr_corruption, 4),
            'dict_ui': round(self.dict_ui, 4),
            'broken_unicode': round(self.broken_unicode, 4),
            'scraping_remnant': round(self.scraping_remnant, 4),
            'format_break': round(self.format_break, 4),
            'contact_block': round(self.contact_block, 4),
            'author_meta': round(self.author_meta, 4),
            'news_meta': round(self.news_meta, 4),
            'promotional': round(self.promotional, 4),
            'severity': round(self.severity, 4),
        }

def _word_stats(text: str) -> tuple[int, int, int, int]:
    words = 0
    mixed_case = 0
    punct_heavy = 0
    short_garbage = 0
    for token in text.split():
        tlen = len(token)
        if tlen < 2:
            continue
        words += 1
        alpha = upper = lower = punct = 0
        for c in token:
            if c.isalpha():
                alpha += 1
                if c.isupper():
                    upper += 1
                elif c.islower():
                    lower += 1
            elif not c.isalnum():
                punct += 1
        if alpha >= 3 and upper > 0 and lower > 0 and upper / alpha > 0.25:
            mixed_case += 1
        if punct / tlen >= 0.45:
            punct_heavy += 1
        if tlen <= 3 and punct >= 1:
            short_garbage += 1
    return words, mixed_case, punct_heavy, short_garbage

def _ocr_corruption_score(text: str) -> float:
    if not text.strip():
        return 0.0
    words, mixed, punct_h, garbage = _word_stats(text)
    if words < 8:
        return 0.0
    ratio = (mixed * 0.45 + punct_h * 0.35 + garbage * 0.20) / words
    return min(1.0, ratio * 2.8)

def _broken_unicode_score(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    total = 0
    for ch in text:
        if ch in '\n\r\t ':
            continue
        total += 1
        if ch == '\ufffd':
            bad += 1
            continue
        cat = unicodedata.category(ch)
        if cat.startswith('C') and ch not in '\n\r\t':
            bad += 1
    if total < 20:
        return 0.0
    return min(1.0, bad / total * 6.0)

def _dict_ui_score(lines: list[str]) -> float:
    if len(lines) < 3:
        return 0.0
    hits = 0
    for ln in lines[:40]:
        low = ln.lower()
        if low.startswith('title:') and len(ln.split()) <= 6:
            hits += 1
        if 'webster' in low and len(ln) < 120:
            hits += 1
        if low.startswith('enter a word') or low.startswith('dictionary definition'):
            hits += 1
        words = ln.split()
        if len(words) <= 4 and ln.endswith(':') and ln[0].isupper():
            hits += 1
    return min(1.0, hits / max(len(lines[:40]), 1) * 3.5)

def _scraping_remnant_score(lines: list[str]) -> float:
    if not lines:
        return 0.0
    navish = 0
    for ln in lines[:60]:
        if ln.count('|') >= 2 and len(ln) < 200:
            navish += 1
        if ln.count('http') >= 2 and len(ln) < 180:
            navish += 1
        if len(ln.split()) <= 3 and ln.isupper():
            navish += 1
    return min(1.0, navish / max(len(lines[:60]), 1) * 2.2)

def _format_break_score(text: str) -> float:
    if not text:
        return 0.0
    lines = text.splitlines()
    if len(lines) < 4:
        return 0.0
    empty_runs = 0
    run = 0
    for ln in lines:
        if not ln.strip():
            run += 1
            if run >= 3:
                empty_runs += 1
        else:
            run = 0
    glued = sum(1 for ln in lines if len(ln) > 400 and ln.count(' ') < 8)
    score = empty_runs * 0.08 + glued * 0.12
    return min(1.0, score)

def analyze_artifact_signals(text: str) -> ArtifactSignalBundle:
    from indw.clean.semantic.section_artifacts import score_section_artifact

    all_lines = text.splitlines() or [text]
    content_lines = [ln.strip() for ln in all_lines if ln.strip()]
    profile = score_section_artifact(
        text,
        position_ratio=0.5,
        section_role='body',
        content_lines=content_lines,
        all_lines=all_lines,
    )
    return ArtifactSignalBundle(
        ocr_corruption=_ocr_corruption_score(text),
        dict_ui=_dict_ui_score(content_lines),
        broken_unicode=_broken_unicode_score(text),
        scraping_remnant=_scraping_remnant_score(content_lines),
        format_break=_format_break_score(text),
        contact_block=profile.contact,
        author_meta=profile.author,
        news_meta=profile.news_meta,
        promotional=profile.promotional,
    )
