from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Any

from indw.filter.language.config import LanguagePolicyConfig
from indw.filter.language.fast_detector import FastLanguageDetector
from indw.filter.language.mixed import MixedLanguageAnalyzer

from indw.util.stats import wilson_ci

_WILSON_Z = 1.96


@lru_cache(maxsize=1)
def _language_stack() -> tuple[FastLanguageDetector, MixedLanguageAnalyzer, LanguagePolicyConfig]:
    policy = LanguagePolicyConfig.resolve()
    detector = FastLanguageDetector(policy.detector)
    return detector, MixedLanguageAnalyzer(detector, policy.mixed), policy


def _forum_flags(text: str) -> bool:
    from indw.clean.artifact.registry import get_artifact_registry
    return 'forum_junk' in get_artifact_registry().audit_flags(text)


from indw.util.stats import wilson_ci

def estimate_population(rate: float, total: int) -> int:
    return int(round(rate * total))


def detect_lang(text: str) -> str:
    detector, _, _ = _language_stack()
    dist = detector.predict_distribution(text)
    if not dist:
        return 'unknown'
    return max(dist, key=dist.get)


def is_mixed_language(text: str) -> bool:
    _, mixed, _ = _language_stack()
    return mixed.analyze(text).mixed_language


def norm_dedup(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())[:1500]


@dataclass
class DocRecord:
    line_no: int
    text: str
    source: str
    char_len: int


@dataclass
class SampleCounters:
    n: int = 0
    empty: int = 0
    json_errors: int = 0
    total_chars: int = 0
    char_lens: list[int] = field(default_factory=list)
    exact_hashes: Counter = field(default_factory=Counter)
    norm_hashes: Counter = field(default_factory=Counter)
    langs: Counter = field(default_factory=Counter)
    mixed_lang: int = 0
    sources: Counter = field(default_factory=Counter)
    flags: Counter = field(default_factory=Counter)
    trunc_none: int = 0
    trunc_slight: int = 0
    trunc_heavy: int = 0
    trunc_examples: list[dict[str, Any]] = field(default_factory=list)
    code_prose: int = 0
    code_educational: int = 0
    code_mixed: int = 0
    code_raw_dump: int = 0
    knowledge_sum: float = 0.0
    educational_sum: float = 0.0
    factual_sum: float = 0.0
    overall_sum: float = 0.0
    completeness_sum: float = 0.0
    category_hits: Counter = field(default_factory=Counter)
    domain_hits: Counter = field(default_factory=Counter)
    metadata_flags: Counter = field(default_factory=Counter)
    signal_sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    utility_sum: float = 0.0
    confidence_sum: float = 0.0
    evidence_n: int = 0
    preserve_count: int = 0
    semantic_discard: Counter = field(default_factory=Counter)
    discovery_artifact_hits: int = 0
    discovery_learned_flags: Counter = field(default_factory=Counter)
    discovery_ratio_sum: float = 0.0
    best: list[tuple[float, DocRecord, str]] = field(default_factory=list)
    worst: list[tuple[float, DocRecord, str]] = field(default_factory=list)


def count_lines(path: Path) -> int:
    count = 0
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            count += chunk.count(b'\n')
    return count
