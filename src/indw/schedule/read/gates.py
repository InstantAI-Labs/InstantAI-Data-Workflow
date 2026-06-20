from __future__ import annotations

import threading
import time
from typing import Any

from indw.config.defaults import MIN_CHARS_GATE
from indw.clean.document.normalize import meaningful_char_count
from indw.clean.corpus import CleaningResult
from indw.filter.language.detect import LanguageAssessment
from indw.filter.score.analysis import _sample_text
from indw.filter.spec.quality import QualityPipelineConfig
from indw.filter.content.domain import domain_from_source
from indw.filter.gate.quality import QualityGate
from indw.schedule.monitor.budget import resolve_doc_max_chars
from indw.clean.document.value import build_analysis_bundle
from indw.store.io.jsonl import parse_jsonl_line

MergeLineKind = str

_LANG_WARM_LOCK = threading.Lock()
_LANG_WARMED = False


def _early_lang_sample_chars(gate: QualityGate) -> int:
    pol = gate._language_policy
    return min(pol.detector.max_chars, 8192 if pol.english_only else 4096)


def early_language_gate(
    text: str,
    gate: QualityGate,
    *,
    meaningful_chars: int | None = None,
) -> tuple[str | None, LanguageAssessment | None]:
    pol = gate._language_policy
    if not pol.enabled:
        return None, None
    _ensure_language_warm(gate)
    ident = gate._language_identifier
    if ident is None:
        return None, None
    sample_limit = _early_lang_sample_chars(gate)
    sample = text[:sample_limit]
    if len(sample) < 40:
        if meaningful_chars is not None and meaningful_chars < MIN_CHARS_GATE:
            return 'too_short', None
        return None, None
    t0 = time.perf_counter()
    if pol.english_only:
        assessment = ident.assess_english_fast(sample)
    else:
        assessment = ident.assess(sample)
    gate.language_stats.record_detection(time.perf_counter() - t0)
    if assessment.should_reject:
        reason = assessment.reject_reason or 'language'
        gate.language_stats.record_early_reject(reason)
        return reason, None
    return None, assessment


def parse_merge_jsonl_line(line: str) -> tuple[MergeLineKind, dict[str, Any] | None]:
    return parse_jsonl_line(line)


def worker_quality_config(
    cfg: QualityPipelineConfig,
    *,
    merge_work: str = '',
) -> QualityPipelineConfig:
    from copy import deepcopy

    wcfg = deepcopy(cfg)
    if not wcfg.cleaning.artifact_discovery:
        return wcfg
    if merge_work:
        wcfg.cleaning.artifact_discovery_corpus_dir = merge_work
        return wcfg
    wcfg.cleaning.artifact_discovery_corpus_dir = ''
    if wcfg.cleaning.artifact_discovery_trim and not wcfg.cleaning.artifact_discovery_shadow:
        wcfg.cleaning.artifact_discovery_shadow = True
    return wcfg


def early_language_reject(text: str, gate: QualityGate) -> str | None:
    reason, _ = early_language_gate(text, gate)
    return reason


def warmup_language_detector(gate: QualityGate) -> None:
    pol = gate._language_policy
    if not pol.enabled:
        return
    ident = gate._language_identifier
    if ident is None:
        return
    sample = 'Language detector warmup sample text.'
    if pol.english_only:
        ident.assess_english_fast(sample)
    else:
        ident.assess(sample)


def _ensure_language_warm(gate: QualityGate) -> None:
    global _LANG_WARMED
    if _LANG_WARMED:
        return
    with _LANG_WARM_LOCK:
        if _LANG_WARMED:
            return
        warmup_language_detector(gate)
        _LANG_WARMED = True


def attach_analysis_cache(clean_result: CleaningResult, cfg: QualityPipelineConfig) -> None:
    if clean_result.dropped or not clean_result.text:
        return
    if clean_result.analysis_bundle is not None:
        return
    th = cfg.thresholds
    scan, full_len = _sample_text(
        clean_result.text,
        min_chars=th.min_chars,
        sample_limit=max(th.min_chars, th.score_sample_chars),
    )
    clean_result.analysis_scan = scan
    clean_result.analysis_full_len = full_len
    clean_result.analysis_bundle = build_analysis_bundle(scan)


def early_document_size_gate(
    text: str,
    src_name: str,
    *,
    meaningful_chars: int | None = None,
    domain: str | None = None,
) -> str | None:
    if (domain if domain is not None else domain_from_source(src_name)) == 'code':
        return None
    if meaningful_chars is not None:
        mc = meaningful_chars
    elif len(text) < MIN_CHARS_GATE:
        return 'too_short'
    else:
        mc = meaningful_char_count(text)
    if mc < MIN_CHARS_GATE:
        return 'too_short'
    return None


def early_document_max_gate(
    text: str,
    src_name: str,
    *,
    domain: str | None = None,
    max_chars: int | None = None,
) -> str | None:
    if (domain if domain is not None else domain_from_source(src_name)) == 'code':
        return None
    limit = max_chars if max_chars is not None else resolve_doc_max_chars()
    if len(text) > limit:
        return 'document_too_large'
    return None
