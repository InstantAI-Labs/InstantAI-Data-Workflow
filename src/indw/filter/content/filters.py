from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
_LINK_SPAM = re.compile(r'https?://|www\.|utm_', re.I)
_YEAR_BARE = re.compile(r'\b\d{4}\b')
_LISTICLE_LINE = re.compile(r'^\s*\d+[\.\)]\s+\S')
_DISCOURSE_SENT = re.compile(r'[.!?]+\s+')

_TRUNC_MARKERS = re.compile(
    r'\[(?:truncated|continued|cut|excerpt|…+)\]|(?:\.\.\.|…)\s*$|<\s*/?\s*(?:page|div|article)\s*>',
    re.I,
)

_BOILERPLATE = re.compile(
    r'(?i)\b(?:'
    r'cookie(?:s)?\s+(?:policy|consent|banner)|accept\s+(?:all\s+)?cookies|'
    r'privacy\s+policy|terms\s+(?:of\s+)?(?:service|use)|all\s+rights\s+reserved|'
    r'skip\s+to\s+(?:main\s+)?content|sign\s+up\s+for\s+(?:our\s+)?newsletter|'
    r'subscribe\s+to\s+(?:our\s+)?(?:newsletter|channel)|follow\s+us\s+on|'
    r'share\s+(?:this|on)\s+\w+|enable\s+javascript|javascript\s+is\s+(?:disabled|required)|'
    r'home\s*[|›>»/]\s*(?:about|contact|products|services|blog)'
    r')\b'
)

_PRICE = re.compile(r'[$€£¥]\s?\d[\d,]*(?:\.\d{2})?|\b\d{1,3}\s*%\s+off\b')
_TRANSACTION_VERB = re.compile(
    r'(?i)\b(?:subscri|regist|enroll|donat|book|reserv|order|checkout|purchas|membership)\w*\b'
)
_CATALOG_FIELD = re.compile(r'(?i)\b(?:sku|isbn|item\s*#|qty|quantity|in\s+stock|publisher\s*:|ages\s*:\s*\d)\b')
_CHECKOUT_URL = re.compile(r'(?i)(?:checkout|/cart|payment|billing|/buy/|/order/|utm_)')

_DISCOURSE_OPENERS = frozenset({
    'furthermore', 'moreover', 'additionally', 'however', 'therefore',
    'thus', 'hence', 'consequently', 'overall', 'finally', 'firstly',
    'secondly', 'lastly', 'indeed', 'nonetheless', 'meanwhile',
})

_CITATION_ANCHOR = re.compile(
    r'(?i)(?:'
    r'\[\d+\]|\(\d{4}[a-z]?\)|doi:\s*\S+|arxiv:\s*\S+|'
    r'https?://[^\s]+|(?:see|cf\.)\s+(?:section|chapter|table|figure)\s+\d+|'
    r'(?:figure|table|section|chapter|appendix)\s+\d+|'
    r'(?:published|journal|proceedings|conference)\s+(?:in|by)|'
    r'(?:et al\.|ibid\.|op\. cit\.)'
    r')'
)

_UNGROUNDED_CLAIM = re.compile(
    r'(?i)\b(?:'
    r'studies (?:show|prove|confirm|suggest)|research (?:proves|shows|confirms)|'
    r'scientists (?:agree|believe|say)|experts (?:agree|say|believe)|'
    r'it (?:is|has been) (?:proven|widely known|well established)|'
    r'everyone knows|common knowledge|undeniably|'
    r'according to (?:studies|research|experts)(?!\s+(?:from|at|by|in|published))'
    r')\b'
)

_SPECIFICITY = re.compile(
    r'\b(?:'
    r'\d{1,3}(?:\.\d+)?%|\d{4}\b|'
    r'\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d+)?\s*(?:million|billion|thousand|km|kg|mhz|gb|tb)|'
    r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}'
    r')\b'
)

_CATALOG_BLURB = re.compile(
    r'(?i)(?:author\s*:|illustrated\s+by\s*:|publisher\s*:|workbook\s+ages\s*:|how\s+to\s+draw\s+book)'
)

_PIRACY_STRONG = re.compile(
    r'(?i)\b(?:'
    r'keygen|key[\s-]?gen(?:erator)?|warez|nulled|crack(?:ed)?\s+(?:software|version|game|app|program)|'
    r'(?:software|program|game|app)\s+crack|download\s+(?:the\s+)?crack|crack\s+download|'
    r'crack\s+only|serial\s+(?:key|number)\s+crack|license\s+key\s+generator|'
    r'(?:kms|windows)\s+activator|kmspico|repack\s+crack|pirated\s+software|'
    r'(?:bypass|circumvent)\s+(?:license|licensing|activation|drm)|'
    r'activate\s+without\s+(?:a\s+)?(?:license|paying|purchase)|'
    r'torrent\s+(?:download|file|link).{0,40}(?:software|game|app|program|windows|office)|'
    r'(?:software|game|app|program|windows|office).{0,40}torrent\s+(?:download|file|link)'
    r')\b'
)

_PIRACY_INSTRUCTIONAL = re.compile(
    r'(?i)\b(?:'
    r'how\s+to\s+(?:crack|activate|bypass|patch)\s+|'
    r'(?:crack|keygen|patch|activator)\s+(?:for|to\s+activate)|'
    r'download\s+(?:and\s+)?install\s+(?:the\s+)?(?:crack|keygen|patch|activator)|'
    r'use\s+(?:a\s+)?keygen\s+to|apply\s+the\s+(?:crack|patch)\s+to|'
    r'get\s+(?:the\s+)?full\s+version\s+for\s+free(?!\s+trial)'
    r')\b'
)

_COPYRIGHT_STRONG = re.compile(
    r'(?i)\b(?:'
    r'(?:remove|strip|break|circumvent)\s+(?:drm|copy\s*protection|copyright\s+protection)|'
    r'drm\s+remov(?:al|er|e)|'
    r'illegal(?:ly)?\s+download(?:ing)?\s+(?:of\s+)?(?:movies?|films?|ebooks?|books?|albums?|music|songs?|shows?)|'
    r'download\s+(?:the\s+)?(?:full|entire)\s+(?:movie|film|ebook|book|album|show)(?:\s+for\s+free)?|'
    r'pirated\s+(?:copy|ebook|movie|film|album|book|pdf|epub)|'
    r'(?:movie|film|ebook|book|album|show|song)\s+torrent|'
    r'torrent.{0,40}(?:movie|film|ebook|book|album|show|song)|'
    r'watch\s+(?:the\s+)?(?:full\s+)?(?:movie|film|show).{0,35}online\s+free|'
    r'leaked\s+(?:copy|pdf|ebook|epub|album)|'
    r'copyrighted\s+(?:material|content|work).{0,40}(?:download|torrent|free)|'
    r'(?:download|torrent|stream).{0,40}copyrighted\s+(?:material|content|work)'
    r')\b'
)

_COPYRIGHT_INSTRUCTIONAL = re.compile(
    r'(?i)\b(?:'
    r'how\s+to\s+(?:download|rip|copy)\s+(?:copyrighted|protected)\s+|'
    r'how\s+to\s+(?:remove|strip|bypass)\s+(?:drm|copy\s*protection|copyright)|'
    r'how\s+to\s+(?:rip|copy)\s+(?:dvd|blu-?ray|cd)s?|'
    r'bypass\s+dmca|evade\s+copyright|'
    r'download\s+(?:copyrighted|protected)\s+(?:movies?|films?|ebooks?|books?|albums?|music)'
    r')\b'
)

_PIRACY_LEGAL_CONTEXT = re.compile(
    r'(?i)\b(?:'
    r'illegal|copyright\s+infringement|intellectual\s+property|legal\s+consequences|'
    r'law\s+enforcement|anti[- ]piracy|official\s+(?:purchase|subscription|license)|'
    r'free\s+trial|open[- ]source\s+alternative|legitimate\s+alternative|'
    r'fair\s+use|rights\s+holder|licensed\s+under|public\s+domain|creative\s+commons|'
    r'dmca\s+takedown|copyright\s+law|copyright\s+act|copyright\s+holder'
    r')\b'
)

@dataclass
class ContentFilterSignals:
    truncation_score: float = 0.0
    boilerplate_score: float = 0.0
    commercial_score: float = 0.0
    seo_spam_score: float = 0.0
    low_information_score: float = 0.0
    keyword_stuffing_score: float = 0.0
    software_piracy_score: float = 0.0
    ai_verbosity_score: float = 0.0
    discourse_template_score: float = 0.0
    artificial_enthusiasm_score: float = 0.0
    hallucination_risk_score: float = 0.0
    transaction_signal_hits: int = -1

def _unique_word_ratio(words: list[str]) -> float:
    return len(set(words)) / max(len(words), 1)

def _structural_seo_score(
    text: str,
    words: list[str],
    lines: list[str],
    *,
    stuffing: float,
    link_hits: int | None = None,
    shout_lines: int | None = None,
) -> float:
    n_lines = max(len(lines), 1)
    n_words = max(len(words), 1)
    score = stuffing * 0.9
    if link_hits is None:
        link_hits = len(_LINK_SPAM.findall(text))
    score = max(score, min(1.0, link_hits / max(n_lines, 1) * 0.35))
    if shout_lines is None:
        shout_lines = sum(1 for ln in lines if 10 < len(ln) < 80 and ln.isupper())
    if shout_lines >= 2:
        score = max(score, min(1.0, 0.35 + shout_lines / n_lines))
    listicle = sum(1 for ln in lines if len(ln) < 90 and _LISTICLE_LINE.match(ln))
    if listicle >= 3:
        score = max(score, min(1.0, 0.3 + listicle / n_lines))
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    if caps_words / n_words > 0.06:
        score = max(score, min(1.0, caps_words / n_words * 6))
    cta_lines = sum(
        1 for ln in lines
        if len(ln) < 100 and _LINK_SPAM.search(ln) and _TRANSACTION_VERB.search(ln)
    )
    if cta_lines >= 1:
        score = max(score, 0.45)
    return min(1.0, score)

def _structural_low_info_score(
    text: str,
    words: list[str],
    lines: list[str],
    *,
    citation_anchors: int | None = None,
    year_token_hits: int | None = None,
) -> float:
    n_lines = max(len(lines), 1)
    n_words = max(len(words), 1)
    unique = _unique_word_ratio(words)
    if n_words < 45 and unique > 0.72:
        return 0.0
    if citation_anchors is None:
        citation_anchors = len(_CITATION_ANCHOR.findall(text))
    if year_token_hits is None:
        year_token_hits = len(_YEAR_BARE.findall(text))
    fact_anchors = citation_anchors + year_token_hits
    anchor_density = fact_anchors / max(n_words / 80, 1)
    score = 0.0
    if unique < 0.38 and n_words > 80:
        score = max(score, min(1.0, (0.42 - unique) * 3.5))
    if anchor_density < 0.15 and n_words > 120:
        score = max(score, min(1.0, 0.25 + (0.15 - anchor_density) * 2))
    short_para = sum(1 for ln in lines if 20 <= len(ln) <= 120 and not re.search(r'\d', ln))
    if short_para / n_lines > 0.55 and anchor_density < 0.2:
        score = max(score, 0.4)
    if any(re.match(r'(?i)^(?:welcome|hello|hi)\b', ln) for ln in lines[:3]):
        score = max(score, 0.35)
    return min(1.0, score)

def _structural_discourse_score(text: str, lines: list[str]) -> float:
    sentences = [s for s in _DISCOURSE_SENT.split(text) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    openers = 0
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if re.search(
            r'(?i)(?:^|[,;]\s*)(?:in conclusion|to summarize|to sum up|in summary|in addition)\b',
            s[:60],
        ):
            openers += 1
            continue
        parts = s.split()
        if not parts:
            continue
        first = parts[0].lower().rstrip(',:')
        if len(first) <= 14 and first in _DISCOURSE_OPENERS and not re.search(r'\d', s[:40]):
            openers += 1
        elif re.search(r'(?i)\b(?:furthermore|moreover|additionally|however|therefore)\b', s[:50]):
            openers += 1
    ratio = openers / max(len(sentences), 1)
    if ratio < 0.12:
        return 0.0
    return min(1.0, ratio * 2.2)

def _structural_ai_verbosity_score(
    text: str,
    words: list[str],
    lines: list[str],
    *,
    discourse: float,
    low_info: float,
) -> float:
    n_words = max(len(words), 1)
    unique = _unique_word_ratio(words)
    avg_len = n_words / max(len(lines), 1)
    score = 0.0
    if discourse > 0.2:
        score = max(score, discourse * 0.85)
    if low_info > 0.35 and unique < 0.45:
        score = max(score, low_info * 0.75)
    if avg_len > 25 and unique < 0.42:
        score = max(score, min(1.0, (avg_len - 20) / 40 * max(0.0, 0.5 - unique) * 4))
    hedge = len(re.findall(
        r"(?i)\b(?:important to note|worth noting|crucial to|it's essential|delve into|holistic approach)\b",
        text,
    ))
    if hedge >= 2:
        score = max(score, min(1.0, 0.3 + hedge / max(n_words / 150, 1)))
    elif hedge >= 1 and discourse > 0.15:
        score = max(score, 0.32)
    return min(1.0, score)

def _structural_enthusiasm_score(text: str, words: list[str], lines: list[str]) -> float:
    excl_density = text.count('!') / max(len(lines), 1)
    caps = sum(1 for w in words if w.isupper() and len(w) > 3) / max(len(words), 1)
    hype_lines = sum(
        1 for ln in lines
        if '!' in ln and len(ln) < 100 and not re.search(r'\d{4}', ln)
    )
    score = min(1.0, excl_density * 0.35 + caps * 4 + hype_lines / max(len(lines), 1) * 0.8)
    return score if score > 0.15 else 0.0

def _keyword_stuffing_score(words: list[str]) -> float:
    if len(words) < 40:
        return 0.0
    counts = Counter(words)
    if not counts:
        return 0.0
    top_freq = counts.most_common(1)[0][1] / len(words)
    bigrams = Counter((words[i], words[i + 1]) for i in range(len(words) - 1))
    tri_freq = 0.0
    if bigrams:
        tri_freq = bigrams.most_common(1)[0][1] / max(len(words) - 1, 1)
    score = 0.0
    if top_freq > 0.08:
        score = max(score, min(1.0, (top_freq - 0.06) * 8.0))
    if tri_freq > 0.04:
        score = max(score, min(1.0, (tri_freq - 0.03) * 10.0))
    return score

def _truncation_score(text: str) -> float:
    from indw.filter.refine.truncation import base_truncation_signal
    return base_truncation_signal(text)

def _pattern_density(
    text: str,
    pattern: re.Pattern[str],
    *,
    per_lines: bool = True,
    lines: list[str] | None = None,
) -> float:
    if lines is None:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()] or [text]
    hits = len(pattern.findall(text))
    denom = max(len(lines) if per_lines else len(text) / 500, 1)
    return min(1.0, hits / denom)

def count_transaction_signals(text: str) -> tuple[int, int, int, int]:
    return (
        len(_PRICE.findall(text)),
        len(_TRANSACTION_VERB.findall(text)),
        len(_CATALOG_FIELD.findall(text)),
        len(_CHECKOUT_URL.findall(text)),
    )

def _transaction_score(
    text: str,
    *,
    lines: list[str],
    words: list[str],
    txn_counts: tuple[int, int, int, int] | None = None,
) -> float:
    n_lines = max(len(lines), 1)
    n_words = max(len(words), 1)
    if txn_counts is None:
        txn_counts = count_transaction_signals(text)
    price_hits, verb_hits, catalog_hits, checkout_hits = txn_counts
    score = min(1.0, (
        price_hits / max(n_words / 80, 1) * 0.35
        + verb_hits / max(n_words / 90, 1) * 0.30
        + catalog_hits / max(n_words / 100, 1) * 0.25
        + checkout_hits / max(n_lines / 5, 1) * 0.20
    ))
    price_lines = sum(1 for ln in lines if _PRICE.search(ln))
    if price_lines >= 2 and n_words < 400:
        score = max(score, 0.55)
    cta_lines = sum(
        1 for ln in lines
        if len(ln) < 90 and (_PRICE.search(ln) or _CHECKOUT_URL.search(ln)) and _TRANSACTION_VERB.search(ln)
    )
    if cta_lines >= 1:
        score = max(score, 0.42)
    return score

def _ai_style_score(
    text: str,
    *,
    lines: list[str],
    words: list[str],
    low_info: float,
) -> tuple[float, float, float]:
    discourse = _structural_discourse_score(text, lines)
    enthusiasm = _structural_enthusiasm_score(text, words, lines)
    ai = _structural_ai_verbosity_score(
        text, words, lines, discourse=discourse, low_info=low_info,
    )
    n_lines = max(len(lines), 1)
    n_words = max(len(words), 1)
    if discourse > 0.25 and n_words < 1200:
        discourse = max(discourse, min(1.0, 0.30 + discourse * 0.5))
    if ai > 0.2 and n_words < 800:
        ai = max(ai, min(1.0, 0.35 + ai / max(n_lines, 1)))
    return ai, discourse, enthusiasm

def _hallucination_risk_score(
    text: str,
    *,
    n_words: int,
    citation_anchors: int | None = None,
) -> float:
    if n_words < 15:
        return 0.0
    ungrounded = len(_UNGROUNDED_CLAIM.findall(text))
    anchors = citation_anchors if citation_anchors is not None else len(_CITATION_ANCHOR.findall(text))
    specificity = len(_SPECIFICITY.findall(text))
    if ungrounded == 0 and specificity < 4:
        return 0.0
    anchor_density = anchors / max(n_words / 100, 1)
    spec_density = specificity / max(n_words / 80, 1)
    risk = 0.0
    if ungrounded >= 1:
        risk = max(risk, min(1.0, 0.25 + ungrounded * 0.18 - anchor_density * 0.12))
    if spec_density > 1.2 and anchor_density < 0.15:
        risk = max(risk, min(1.0, 0.20 + (spec_density - 1.0) * 0.25))
    if ungrounded >= 2 and anchors == 0 and spec_density > 0.6:
        risk = max(risk, 0.65)
    return min(1.0, risk)

def _software_piracy_score(text: str) -> float:
    strong_hits = len(_PIRACY_STRONG.findall(text)) + len(_COPYRIGHT_STRONG.findall(text))
    instr_hits = len(_PIRACY_INSTRUCTIONAL.findall(text)) + len(_COPYRIGHT_INSTRUCTIONAL.findall(text))
    if strong_hits == 0 and instr_hits == 0:
        return 0.0
    score = min(1.0, strong_hits * 0.55 + instr_hits * 0.65)
    if strong_hits >= 1 and instr_hits >= 1:
        score = max(score, 0.92)
    elif strong_hits >= 2 or instr_hits >= 2:
        score = max(score, 0.88)
    elif strong_hits >= 1:
        score = max(score, 0.78)
    elif instr_hits >= 1:
        score = max(score, 0.72)
    if instr_hits == 0 and _PIRACY_LEGAL_CONTEXT.search(text):
        if strong_hits <= 1:
            return 0.0
        score *= 0.4
    return min(1.0, score)

def analyze_content_filters(
    text: str,
    *,
    words: list[str] | None = None,
    lines: list[str] | None = None,
) -> ContentFilterSignals:
    if not text or not text.strip():
        return ContentFilterSignals(
            truncation_score=1.0,
            low_information_score=1.0,
            transaction_signal_hits=0,
        )
    from indw.clean.artifact.evidence_cache import filters_cache_key, get_filters_cache

    cache_key = filters_cache_key(text, words=words, lines=lines)
    if cache_key is not None:
        cache = get_filters_cache()
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
    if words is None:
        words = [w.lower() for w in _WORD.findall(text)]
    if lines is None:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n_lines = max(len(lines), 1)

    citation_anchors = len(_CITATION_ANCHOR.findall(text))
    link_hits = len(_LINK_SPAM.findall(text))
    shout_lines = sum(1 for ln in lines if len(ln) < 80 and ln.isupper() and len(ln) > 10)

    txn_counts = count_transaction_signals(text)
    trunc = _truncation_score(text)
    boiler = _pattern_density(text, _BOILERPLATE, lines=lines)
    commercial = _transaction_score(text, lines=lines, words=words, txn_counts=txn_counts)
    stuffing = _keyword_stuffing_score(words)
    low_info = _structural_low_info_score(
        text, words, lines,
        citation_anchors=citation_anchors,
        year_token_hits=len(_YEAR_BARE.findall(text)),
    )
    seo = _structural_seo_score(
        text, words, lines, stuffing=stuffing,
        link_hits=link_hits, shout_lines=shout_lines,
    )

    if _CATALOG_BLURB.search(text) and len(text) < 1200:
        commercial = max(commercial, 0.85)

    pipe_lines = sum(1 for ln in lines if ln.count('|') >= 2 and len(ln) < 120)
    if pipe_lines >= 2:
        boiler = max(boiler, min(1.0, 0.5 + pipe_lines / n_lines))

    if link_hits >= 4:
        commercial = max(commercial, min(1.0, 0.35 + link_hits / max(n_lines, 1)))

    if shout_lines >= 2:
        seo = max(seo, min(1.0, 0.4 + shout_lines / n_lines))

    seo = max(seo, stuffing * 0.85)

    if low_info > 0.35 and len(words) < 600:
        low_info = max(low_info, min(1.0, low_info + 0.15))

    piracy = _software_piracy_score(text)
    ai, discourse, enthusiasm = _ai_style_score(
        text, lines=lines, words=words, low_info=low_info,
    )
    hallucination = _hallucination_risk_score(
        text, n_words=len(words), citation_anchors=citation_anchors,
    )

    result = ContentFilterSignals(
        truncation_score=trunc,
        boilerplate_score=boiler,
        commercial_score=commercial,
        seo_spam_score=seo,
        low_information_score=low_info,
        keyword_stuffing_score=stuffing,
        software_piracy_score=piracy,
        ai_verbosity_score=ai,
        discourse_template_score=discourse,
        artificial_enthusiasm_score=enthusiasm,
        hallucination_risk_score=hallucination,
        transaction_signal_hits=sum(txn_counts),
    )
    if cache_key is not None:
        get_filters_cache().put(cache_key, result)
    return result
