from __future__ import annotations

import re
from dataclasses import dataclass

from indw.clean.artifact.decompose import compute_layout
from indw.clean.semantic.fingerprints import SemanticFingerprintMatcher
from indw.clean.artifact.evidence_features import RawDocumentFeatures, shared_feature_extractor

_EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
_PHONE = re.compile(r'\+?\d[\d\s().-]{7,}\d')
_URL = re.compile(r'https?://\S+|www\.\S+', re.I)
_YEAR = re.compile(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b')
_STRUCT_KV = re.compile(r'^\s*[\w][\w\s]{0,28}\s*:\s*\S')

_FP_MATCHER: SemanticFingerprintMatcher | None = None

def _fp_matcher() -> SemanticFingerprintMatcher:
    global _FP_MATCHER
    if _FP_MATCHER is None:
        _FP_MATCHER = SemanticFingerprintMatcher()
    return _FP_MATCHER

@dataclass
class SectionArtifactProfile:
    contact: float = 0.0
    author: float = 0.0
    news_meta: float = 0.0
    promotional: float = 0.0
    navigation: float = 0.0
    legal: float = 0.0
    ocr_noise: float = 0.0

    @property
    def severity(self) -> float:
        return min(1.0, max(
            self.contact, self.author, self.news_meta,
            self.promotional, self.navigation, self.legal, self.ocr_noise,
        ))

    def dominant(self) -> tuple[str, float]:
        items = (
            ('contact', self.contact),
            ('author', self.author),
            ('news_meta', self.news_meta),
            ('promotional', self.promotional),
            ('navigation', self.navigation),
            ('legal', self.legal),
            ('ocr_noise', self.ocr_noise),
        )
        return max(items, key=lambda x: x[1])

def _line_contact_score(stripped: str, *, raw: RawDocumentFeatures) -> float:
    if len(stripped) < 6:
        return 0.0
    chars = max(len(stripped), 1)
    contact = sum(len(m.group(0)) for m in _EMAIL.finditer(stripped))
    contact += sum(len(m.group(0)) for m in _PHONE.finditer(stripped))
    ratio = contact / chars
    sched = raw.schedule_token_ratio * 0.45
    return min(1.0, ratio * 2.2 + sched + raw.contact_token_ratio * 0.8)

def _line_author_score(
    stripped: str,
    *,
    position_ratio: float,
    raw: RawDocumentFeatures,
    word_count: int,
) -> float:
    if len(stripped) < 8 or len(stripped) > 220:
        return 0.0
    score = 0.0
    if word_count <= 14 and raw.url_char_ratio > 0.08:
        score += 0.35
    if raw.first_person_ratio > 0.06 and word_count <= 18:
        score += 0.25
    if position_ratio > 0.68 and word_count <= 12 and raw.url_char_ratio > 0.05:
        score += 0.30
    if word_count <= 8 and raw.url_char_ratio > 0.08 and not _YEAR.search(stripped):
        score += 0.22
    if word_count <= 6 and raw.copula_def_hits == 0 and raw.fact_relation_hits == 0:
        if raw.uppercase_token_ratio > 0.25 and position_ratio < 0.22:
            score += 0.28
    return min(1.0, score)

def _line_news_meta_score(
    stripped: str,
    *,
    position_ratio: float,
    raw: RawDocumentFeatures,
    word_count: int,
) -> float:
    if len(stripped) < 8 or len(stripped) > 160:
        return 0.0
    if not _STRUCT_KV.match(stripped):
        return 0.0
    if word_count > 12:
        return 0.0
    score = 0.35
    if _YEAR.search(stripped):
        score += 0.30
    if position_ratio < 0.18:
        score += 0.20
    if raw.citation_hits == 0 and raw.copula_def_hits == 0:
        score += 0.15
    return min(1.0, score)

def _line_promotional_score(
    stripped: str,
    *,
    position_ratio: float,
    raw: RawDocumentFeatures,
    contact_score: float,
    word_count: int,
) -> float:
    if len(stripped) < 10:
        return 0.0
    fp = _fp_matcher().match(stripped, raw=raw)
    promo = max(fp.get('seo', 0.0), fp.get('contact', 0.0) * 0.6)
    score = contact_score * 0.45 + promo * 0.35 + raw.schedule_token_ratio * 0.55
    if raw.exclaim_line_ratio > 0.5 and word_count < 20:
        score += 0.15
    if position_ratio > 0.72:
        score += 0.12
    if word_count <= 8 and raw.url_char_ratio > 0.12:
        score += 0.18
    return min(1.0, score)

def _line_navigation_score(
    stripped: str,
    *,
    raw: RawDocumentFeatures,
    word_count: int,
    pipe_parts: int,
) -> float:
    if not stripped or len(stripped) > 240:
        return 0.0
    score = 0.0
    if stripped.count('|') >= 2 and pipe_parts >= 3:
        score += 0.55
    url_chars = sum(len(m.group(0)) for m in _URL.finditer(stripped))
    if url_chars / max(len(stripped), 1) >= 0.45:
        score += 0.40
    score += raw.nav_line_ratio * 0.35
    if word_count <= 4 and stripped.isupper():
        score += 0.25
    return min(1.0, score)

def _build_line_profile(
    stripped: str,
    raw: RawDocumentFeatures,
    *,
    position_ratio: float,
    section_role: str,
    word_count: int,
    pipe_parts: int,
) -> SectionArtifactProfile:
    from indw.filter.score.artifacts import _ocr_corruption_score

    contact = _line_contact_score(stripped, raw=raw)
    profile = SectionArtifactProfile(
        contact=contact,
        author=_line_author_score(
            stripped, position_ratio=position_ratio, raw=raw, word_count=word_count,
        ),
        news_meta=_line_news_meta_score(
            stripped, position_ratio=position_ratio, raw=raw, word_count=word_count,
        ),
        promotional=_line_promotional_score(
            stripped,
            position_ratio=position_ratio,
            raw=raw,
            contact_score=contact,
            word_count=word_count,
        ),
        navigation=_line_navigation_score(
            stripped, raw=raw, word_count=word_count, pipe_parts=pipe_parts,
        ),
        ocr_noise=_ocr_corruption_score(stripped),
    )
    if section_role == 'legal':
        profile.legal = 0.75
    elif section_role in ('contact', 'footer', 'navigation'):
        profile.contact = max(profile.contact, 0.55)
        profile.navigation = max(profile.navigation, 0.45)
    elif section_role in ('author_info', 'metadata'):
        profile.author = max(profile.author, 0.45)
        profile.news_meta = max(profile.news_meta, 0.35)
    elif section_role == 'promotional':
        profile.promotional = max(profile.promotional, 0.55)
    return profile

def _line_raw(
    stripped: str,
    cache: dict[str, RawDocumentFeatures],
) -> RawDocumentFeatures:
    raw = cache.get(stripped)
    if raw is None:
        raw = shared_feature_extractor().extract(stripped)
        cache[stripped] = raw
    return raw

def score_line_artifact(
    line: str,
    *,
    position_ratio: float = 0.5,
    section_role: str = 'body',
) -> SectionArtifactProfile:
    stripped = line.strip()
    if not stripped:
        return SectionArtifactProfile()
    words = stripped.split()
    return _build_line_profile(
        stripped,
        _line_raw(stripped, {}),
        position_ratio=position_ratio,
        section_role=section_role,
        word_count=len(words),
        pipe_parts=len(stripped.split('|')) if '|' in stripped else 0,
    )

def line_should_remove(
    line: str,
    *,
    position_ratio: float = 0.5,
    section_role: str = 'body',
    preserve_educational: bool = False,
) -> tuple[bool, str]:
    stripped = line.strip()
    if not stripped:
        return False, ''
    words = stripped.split()
    word_count = len(words)
    cache: dict[str, RawDocumentFeatures] = {}
    raw = _line_raw(stripped, cache)
    profile = _build_line_profile(
        stripped,
        raw,
        position_ratio=position_ratio,
        section_role=section_role,
        word_count=word_count,
        pipe_parts=len(stripped.split('|')) if '|' in stripped else 0,
    )
    kind, score = profile.dominant()

    if preserve_educational:
        if raw.copula_def_hits > 0 or raw.step_line_hits > 0 or raw.fact_relation_hits > 0:
            if score < 0.72:
                return False, ''
        if word_count >= 18 and raw.sentence_count >= 2:
            if score < 0.68:
                return False, ''

    thresholds = {
        'contact': 0.36,
        'author': 0.44,
        'news_meta': 0.40,
        'promotional': 0.38,
        'navigation': 0.46,
        'legal': 0.68,
        'ocr_noise': 0.76,
    }
    thr = thresholds.get(kind, 0.55)
    if section_role in ('contact', 'footer', 'navigation', 'promotional', 'author_info', 'metadata'):
        thr *= 0.82
    if score >= thr:
        return True, kind
    return False, ''

def score_section_artifact(
    text: str,
    *,
    position_ratio: float = 0.5,
    section_role: str = 'body',
    content_lines: list[str] | None = None,
    all_lines: list[str] | None = None,
) -> SectionArtifactProfile:
    if content_lines is not None:
        lines = content_lines
        layout_lines = all_lines
    else:
        layout_lines = all_lines or text.splitlines() or [text]
        lines = [ln for ln in layout_lines if ln.strip()]
    if not lines:
        return SectionArtifactProfile()
    acc = SectionArtifactProfile()
    n = len(lines)
    raw_cache: dict[str, RawDocumentFeatures] = {}
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if not stripped:
            continue
        words = stripped.split()
        pos = position_ratio + (i / max(n, 1)) * 0.08
        p = _build_line_profile(
            stripped,
            _line_raw(stripped, raw_cache),
            position_ratio=pos,
            section_role=section_role,
            word_count=len(words),
            pipe_parts=len(stripped.split('|')) if '|' in stripped else 0,
        )
        acc.contact = max(acc.contact, p.contact)
        acc.author = max(acc.author, p.author)
        acc.news_meta = max(acc.news_meta, p.news_meta)
        acc.promotional = max(acc.promotional, p.promotional)
        acc.navigation = max(acc.navigation, p.navigation)
        acc.legal = max(acc.legal, p.legal)
        acc.ocr_noise = max(acc.ocr_noise, p.ocr_noise)
    layout = compute_layout(text, lines=layout_lines)
    if layout.list_ratio > 0.5 and position_ratio > 0.78:
        acc.promotional = max(acc.promotional, 0.35)
    return acc

def find_promotional_tail_start(text: str) -> int | None:
    paras = re.split(r'\n\s*\n+', text.strip())
    if len(paras) < 2:
        return None
    total_len = max(len(text), 1)
    offset = 0
    cut_at: int | None = None
    for para in paras:
        pos = offset / total_len
        if pos < 0.40:
            offset += len(para) + 2
            continue
        profile = score_section_artifact(para, position_ratio=pos, section_role='promotional')
        raw = shared_feature_extractor().extract(para)
        edu = raw.copula_def_hits + raw.fact_relation_hits + raw.step_line_hits
        if profile.promotional >= 0.38 or profile.contact >= 0.44:
            if edu <= 1 or len(para.split()) < 24:
                from indw.extract.structure.analyze import analyze_structure
                st = analyze_structure(para)
                if (
                    para.strip()
                    and para.strip()[-1] in '.!?)"\'»]})'
                    and st.sentence_completeness_mean >= 0.80
                    and len(para.split()) >= 5
                ):
                    offset += len(para) + 2
                    continue
                cut_at = offset
                break
        offset += len(para) + 2
    return cut_at
