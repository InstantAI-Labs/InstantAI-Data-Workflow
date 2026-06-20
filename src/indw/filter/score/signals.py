from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from indw.filter.content.filters import analyze_content_filters
_HTML = re.compile('<[^>]+>|&[a-z]+;|&#\\d+;')
_INJECTION = re.compile('(ignore (all )?previous|system prompt|jailbreak|do anything now|you are now|act as (?:a )?dan)', re.I)
_REASONING = re.compile("\\b(therefore|because|step \\d|first,|second,|thus,|hence,|let's think|reasoning:|analysis:)\\b", re.I)
_CODE_FENCE = re.compile('```|<code>|def |class |import |function\\s*\\(')
_WORD = re.compile("\\b[\\w']+\\b", re.UNICODE)
_DELIMS = re.compile(r'[\[\]\(\)\{\}<>\|`~:;,\.\-_=+/\\*!?]')
_FACTUAL = re.compile(r'\b(according to|study|evidence|dataset|experiment|theorem|proof|measured|benchmark|reported)\b', re.I)
_EDU = re.compile(r'\b(explain|example|intuition|derive|step-by-step|why|how|trade-off|design)\b', re.I)

@dataclass
class QualitySignals:
    char_entropy: float = 0.0
    word_diversity: float = 0.0
    line_repetition: float = 0.0
    char_repetition: float = 0.0
    alpha_ratio: float = 0.0
    html_score: float = 0.0
    injection_score: float = 0.0
    reasoning_density: float = 0.0
    code_density: float = 0.0
    formatting_score: float = 0.0
    delimiter_density: float = 0.0
    token_spam_score: float = 0.0
    coherence_score: float = 0.0
    structural_quality: float = 0.0
    factual_density: float = 0.0
    educational_value: float = 0.0
    repeated_span_score: float = 0.0
    semantic_diversity: float = 0.0
    synthetic_score: float = 0.0
    reasoning_repetition: float = 0.0
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
    template_synthetic_score: float = 0.0
    burstiness_score: float = 0.0
    length: int = 0

def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n * math.log2(c / n) for c in counts.values()))

def _sentence_burstiness(text: str) -> float:
    parts = re.split(r'[.!?]+', text)
    lengths = [len(_WORD.findall(p)) for p in parts if p.strip()]
    if len(lengths) < 3:
        return 0.5
    mean = sum(lengths) / len(lengths)
    if mean <= 0:
        return 0.5
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    cv = math.sqrt(variance) / mean
    return max(0.0, min(1.0, cv / 1.2))

def compute_signals(
    text: str,
    *,
    filters: Any | None = None,
    words: list[str] | None = None,
    lines: list[str] | None = None,
) -> QualitySignals:
    n = len(text)
    if n == 0:
        return QualitySignals()
    if words is None:
        words = _WORD.findall(text.lower())
    else:
        words = [w.lower() for w in words]
    unique_words = len(set(words))
    word_div = unique_words / max(len(words), 1)
    ent = shannon_entropy(text)
    if lines is None:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line_rep = 0.0
    if len(lines) > 2:
        line_counts = Counter(lines)
        line_rep = line_counts.most_common(1)[0][1] / len(lines)
    char_rep = 0.0
    if n > 20:
        grams = Counter((text[i:i + 5] for i in range(0, n - 4, 3)))
        if grams:
            char_rep = grams.most_common(1)[0][1] / len(grams)
    alpha = sum((c.isalpha() for c in text)) / n
    html_hits = len(_HTML.findall(text))
    html_score = min(1.0, html_hits / max(n / 200, 1))
    inj = len(_INJECTION.findall(text))
    injection_score = min(1.0, inj * 0.25)
    reasoning = len(_REASONING.findall(text))
    reasoning_density = min(1.0, reasoning / max(len(lines), 1))
    code_hits = len(_CODE_FENCE.findall(text))
    code_density = min(1.0, code_hits / max(len(lines) / 5, 1))
    delim_density = len(_DELIMS.findall(text)) / max(n, 1)
    factual_hits = len(_FACTUAL.findall(text))
    factual_density = min(1.0, factual_hits / max(len(lines), 1))
    edu_hits = len(_EDU.findall(text))
    educational_value = min(1.0, edu_hits / max(len(lines), 1))
    repeated_span_score = max(line_rep, char_rep)
    semantic_diversity = max(0.0, min(1.0, word_div * (1.0 - repeated_span_score)))
    token_spam_score = min(1.0, max(0.0, repeated_span_score * 0.7 + max(0.0, delim_density - 0.2) * 1.5))
    line_len = [len(ln) for ln in lines] or [0]
    avg_len = sum(line_len) / max(len(line_len), 1)
    variance = sum(((x - avg_len) ** 2 for x in line_len)) / max(len(line_len), 1)
    coherence_score = max(0.0, min(1.0, 1.0 - (variance / max(avg_len * avg_len, 1.0))))
    broken = text.count('�')
    fmt_penalty = min(1.0, broken / max(n / 500, 1))
    formatting_score = 1.0 - fmt_penalty
    structural_quality = max(0.0, min(1.0, formatting_score * (1.0 - html_score) * (1.0 - token_spam_score)))
    reasoning_repetition = min(1.0, repeated_span_score * (0.5 + reasoning_density * 0.5))
    burstiness = _sentence_burstiness(text)
    content = filters if filters is not None else analyze_content_filters(text)
    template_synthetic = min(
        1.0,
        content.ai_verbosity_score * 0.35
        + content.discourse_template_score * 0.25
        + content.artificial_enthusiasm_score * 0.15
        + max(0.0, 0.55 - burstiness) * 0.25,
    )
    synthetic_score = min(
        1.0,
        max(
            template_synthetic * 0.55,
            repeated_span_score * 0.30 + (1.0 - semantic_diversity) * 0.25 + max(0.0, 0.2 - ent / 10.0) * 0.15,
        ),
    )
    return QualitySignals(
        char_entropy=ent,
        word_diversity=word_div,
        line_repetition=line_rep,
        char_repetition=char_rep,
        alpha_ratio=alpha,
        html_score=html_score,
        injection_score=injection_score,
        reasoning_density=reasoning_density,
        code_density=code_density,
        formatting_score=formatting_score,
        delimiter_density=delim_density,
        token_spam_score=token_spam_score,
        coherence_score=coherence_score,
        structural_quality=structural_quality,
        factual_density=factual_density,
        educational_value=educational_value,
        repeated_span_score=repeated_span_score,
        semantic_diversity=semantic_diversity,
        synthetic_score=synthetic_score,
        reasoning_repetition=reasoning_repetition,
        truncation_score=content.truncation_score,
        boilerplate_score=content.boilerplate_score,
        commercial_score=content.commercial_score,
        seo_spam_score=content.seo_spam_score,
        low_information_score=content.low_information_score,
        keyword_stuffing_score=content.keyword_stuffing_score,
        software_piracy_score=content.software_piracy_score,
        ai_verbosity_score=content.ai_verbosity_score,
        discourse_template_score=content.discourse_template_score,
        artificial_enthusiasm_score=content.artificial_enthusiasm_score,
        hallucination_risk_score=content.hallucination_risk_score,
        template_synthetic_score=template_synthetic,
        burstiness_score=burstiness,
        length=n
    )
