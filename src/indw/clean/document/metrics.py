from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any

from indw.clean.document.config import CleaningConfig

from indw.clean.document.patterns import _CODE_FENCE, _UI_LINE, _WORD

from indw.clean.document.adaptive import adaptive_quality_score
from indw.clean.document.normalize import meaningful_char_count
from indw.filter.content.filters import analyze_content_filters
from indw.clean.document.value import analyze_content_value

_BOILERPLATE_HINT = re.compile(

    r'(?i)\b(?:cookie|privacy\s+policy|sign\s+up|advertisement|skip\s+to\s+content)\b'

)

@dataclass

class ChunkMetrics:

    word_count: int = 0

    token_estimate: int = 0

    code_ratio: float = 0.0

    duplicate_ratio: float = 0.0

    ui_noise_ratio: float = 0.0

    boilerplate_ratio: float = 0.0

    language: str = 'unknown'

    quality_score: float = 0.0

    educational_score: float = 0.0

    technical_score: float = 0.0

    semantic_density: float = 0.0

    information_density: float = 0.0

    storytelling_score: float = 0.0

    entertainment_score: float = 0.0

    code_score: float = 0.0

    duplicate_score: float = 0.0

    boilerplate_score: float = 0.0

    spam_probability: float = 0.0

    commercial_probability: float = 0.0

    overall_score: float = 0.0

    reference_score: float = 0.0

    information_density_per_token: float = 0.0

    category: str = 'blog'

    domain: str = 'web'

    meaningful_chars: int = 0

    def to_dict(self) -> dict[str, float | int | str]:

        return {

            'word_count': self.word_count,

            'token_estimate': self.token_estimate,

            'code_ratio': round(self.code_ratio, 4),

            'duplicate_ratio': round(self.duplicate_ratio, 4),

            'ui_noise_ratio': round(self.ui_noise_ratio, 4),

            'boilerplate_ratio': round(self.boilerplate_ratio, 4),

            'language': self.language,

            'quality_score': round(self.quality_score, 4),

            'educational_score': round(self.educational_score, 4),

            'technical_score': round(self.technical_score, 4),

            'semantic_density': round(self.semantic_density, 4),

            'information_density': round(self.information_density, 4),

            'storytelling_score': round(self.storytelling_score, 4),

            'entertainment_score': round(self.entertainment_score, 4),

            'code_score': round(self.code_score, 4),

            'duplicate_score': round(self.duplicate_score, 4),

            'boilerplate_score': round(self.boilerplate_score, 4),

            'spam_probability': round(self.spam_probability, 4),

            'commercial_probability': round(self.commercial_probability, 4),

            'overall_score': round(self.overall_score, 4),

            'reference_score': round(self.reference_score, 4),

            'information_density_per_token': round(self.information_density_per_token, 4),

            'category': self.category,

            'domain': self.domain,

        }

def _ui_noise_ratio(text: str) -> float:
    from indw.clean.artifact.engine import get_artifact_engine
    return get_artifact_engine().ui_noise_ratio(text)

def _code_ratio(text: str) -> float:

    if not text:

        return 0.0

    code_chars = sum(len(m.group(0)) for m in _CODE_FENCE.finditer(text))

    if not code_chars and re.search(r'(?m)^(?: {4}|\t)\S', text):

        code_chars = len(re.findall(r'(?m)^(?: {4}|\t).+$', text)) * 20

    return min(1.0, code_chars / max(len(text), 1))

def _semantic_density(words: list[str], *, boiler: float, duplicate_ratio: float) -> float:

    if not words:

        return 0.0

    unique_ratio = len(set(w.lower() for w in words)) / len(words)

    return max(0.0, min(1.0, unique_ratio * 1.15 - boiler * 0.35 - duplicate_ratio * 0.25))

def _domain_from_category(category: str, code_ratio: float) -> str:

    mapping = {

        'programming': 'code',

        'documentation': 'docs',

        'forum': 'qa',

        'reference': 'wiki',

        'historical': 'wiki',

        'scientific': 'reasoning',

        'tutorial': 'docs',

        'educational': 'docs',

        'technical': 'reasoning',

    }

    if code_ratio >= 0.12:

        return 'code'

    return mapping.get(category, 'web')

def compute_metrics(

    text: str,

    cfg: CleaningConfig,

    *,

    duplicate_ratio: float = 0.0,

    language: str = 'unknown',

    source: str = '',

    content_value: Any | None = None,

    analysis_bundle: Any | None = None,

) -> ChunkMetrics:

    words = _WORD.findall(text)

    word_count = len(words)

    token_estimate = max(1, int(len(text) / max(cfg.chars_per_token_estimate, 1.0)))

    if analysis_bundle is not None:
        filters = analysis_bundle.filters
        value = content_value or analyze_content_value(
            text, source=source, duplicate_ratio=duplicate_ratio, bundle=analysis_bundle,
        )
    else:
        filters = analyze_content_filters(text)
        value = content_value or analyze_content_value(text, source=source, duplicate_ratio=duplicate_ratio)

    ui = _ui_noise_ratio(text)

    code_r = _code_ratio(text)

    boiler = max(filters.boilerplate_score, filters.commercial_score * 0.5)

    spam = max(filters.seo_spam_score, filters.low_information_score * 0.85)

    sem_density = _semantic_density(words, boiler=boiler, duplicate_ratio=duplicate_ratio)

    quality = adaptive_quality_score(
        cv=value,
        sem_density=sem_density,
        ui=ui,
        duplicate_ratio=duplicate_ratio,
    )

    return ChunkMetrics(

        word_count=word_count,

        token_estimate=token_estimate,

        code_ratio=code_r,

        duplicate_ratio=duplicate_ratio,

        ui_noise_ratio=ui,

        boilerplate_ratio=boiler,

        language=language,

        quality_score=quality,

        educational_score=value.educational_score,

        technical_score=value.technical_score,

        semantic_density=sem_density,

        information_density=value.information_density,

        storytelling_score=value.storytelling_score,

        entertainment_score=value.entertainment_score,

        code_score=value.code_score,

        duplicate_score=duplicate_ratio,

        boilerplate_score=boiler,

        spam_probability=spam,

        commercial_probability=filters.commercial_score,

        overall_score=value.overall_value_score,

        reference_score=value.reference_score,

        information_density_per_token=value.information_density_per_token,

        category=value.category,

        domain=_domain_from_category(value.category, code_r),

        meaningful_chars=meaningful_char_count(text),

    )
