from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from indw.tools.reports.fast.patterns import (
    _AI_SLOP, _CHARS_PER_TOKEN, _COOKIE, _FORUM, _GARBAGE_UNICODE, _HTML_TAG, _NAV,
    _OCR, _RANDOM_PUNCT, _REPEATED_SYM, _REPEATED_WORD, _SEO, _WORD_SALAD,
)
from indw.clean.meta.patterns import (
    _BOILERPLATE_LINE, _COPYRIGHT_LINE, _EDITORIAL_LINE, _LEGAL_FOOTER, _REPO_LINE,
)
from indw.clean.meta.foundation import metadata_noise_ratio
from indw.filter.content.code import analyze_code_dump
from indw.clean.document.value import analyze_content_value
from indw.filter.refine.corpus import DocumentAnalysisCache, compute_document_metrics, score_to_grade
from indw.filter.score.signals import compute_signals
from indw.filter.refine.truncation import analyze_truncation
from indw.tools.reports.fast.stats import (
    DocRecord, SampleCounters, _forum_flags, count_lines, detect_lang,
    estimate_population, is_mixed_language, norm_dedup, wilson_ci,
)
from indw.tools.reports.fast.sample import parse_sample_lines, reservoir_sample_lines

def _scan_metadata(text: str) -> list[str]:
    found: list[str] = []
    lines = text.splitlines()
    for ln in lines[:40] + lines[-20:]:
        if _COPYRIGHT_LINE.search(ln):
            found.append('copyright_notice')
        if _EDITORIAL_LINE.search(ln):
            found.append('editorial_metadata')
        if _BOILERPLATE_LINE.search(ln):
            found.append('navigation_boilerplate')
        if _REPO_LINE.search(ln):
            found.append('repo_metadata')
    if _LEGAL_FOOTER.search(text[:3000]) or _LEGAL_FOOTER.search(text[-2000:]):
        found.append('legal_footer')
    if metadata_noise_ratio(text) > 0.12:
        found.append('metadata_noise')
    return list(dict.fromkeys(found))

def _scan_filtering_flags(text: str) -> list[str]:
    from indw.clean.artifact.registry import get_artifact_registry

    flags: list[str] = []
    for audit_flag in get_artifact_registry().audit_flags(text):
        if audit_flag == 'website_artifact':
            flags.append('navigation_menu')
        elif audit_flag == 'forum_junk':
            flags.append('forum_junk')
        elif audit_flag == 'copyright_notice':
            flags.append('cookie_banner')
        elif audit_flag == 'repo_metadata':
            flags.append('repo_metadata')
        else:
            flags.append(audit_flag)
    if _HTML_TAG.search(text):
        flags.append('html')
    if _COOKIE.search(text) and 'cookie_banner' not in flags:
        flags.append('cookie_banner')
    if _NAV.search(text) and 'navigation_menu' not in flags:
        flags.append('navigation_menu')
    if _SEO.search(text):
        flags.append('seo_spam')
    if (_FORUM.search(text) or _forum_flags(text)) and 'forum_junk' not in flags:
        flags.append('forum_junk')
    if _AI_SLOP.search(text):
        flags.append('ai_slop')
    if _REPEATED_WORD.search(text):
        flags.append('repeated_words')
    if _REPEATED_SYM.search(text):
        flags.append('repeated_punctuation')
    if _GARBAGE_UNICODE.search(text) or text.count('\ufffd') >= 2:
        flags.append('encoding_corruption')
    if _OCR.search(text):
        flags.append('ocr_noise')
    if _RANDOM_PUNCT.search(text):
        flags.append('random_symbols')
    if _WORD_SALAD.search(text):
        flags.append('word_salad')
    sig = compute_signals(text)
    if sig.boilerplate_score > 0.35:
        flags.append('boilerplate')
    if sig.html_score > 0.20:
        flags.append('html_score_high')
    if sig.seo_spam_score > 0.25:
        flags.append('seo_score_high')
    if sig.synthetic_score > 0.45:
        flags.append('synthetic_slop')
    return flags

def _heap_push(heap: list, item: tuple, k: int = 10, reverse: bool = False) -> None:
    heap.append(item)
    heap.sort(key=lambda x: x[0], reverse=reverse)
    del heap[k:]

def analyze_sample(
    docs: list[DocRecord],
    *,
    total_docs: int,
) -> dict[str, Any]:
    st = SampleCounters()
    t0 = time.time()

    for doc in docs:
        text = doc.text
        if not text.strip():
            st.empty += 1
            continue
        st.n += 1
        nchar = doc.char_len
        st.total_chars += nchar
        st.char_lens.append(nchar)
        st.sources[doc.source] += 1

        if nchar < 100:
            st.flags['tiny_doc'] += 1
        if nchar > 32000:
            st.flags['huge_doc'] += 1

        st.exact_hashes[hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()] += 1
        st.norm_hashes[hashlib.md5(norm_dedup(text).encode()).hexdigest()] += 1

        lang = detect_lang(text)
        st.langs[lang] += 1
        if is_mixed_language(text):
            st.mixed_lang += 1

        for flag in _scan_filtering_flags(text):
            st.flags[flag] += 1
        for flag in _scan_metadata(text):
            st.metadata_flags[flag] += 1

        try:
            from indw.clean.artifact.discovery_engine import get_discovery_engine
            eng = get_discovery_engine()
            ratio = eng.document_artifact_ratio(text)
            st.discovery_ratio_sum += ratio
            for flag in eng.audit_flags(text):
                st.discovery_learned_flags[flag] += 1
                st.discovery_artifact_hits += 1
        except Exception:
            pass

        trunc = analyze_truncation(text)
        if trunc.severity == 'none':
            st.trunc_none += 1
        elif trunc.severity == 'slight':
            st.trunc_slight += 1
            if len(st.trunc_examples) < 5:
                st.trunc_examples.append({'severity': 'medium', 'preview': text[-120:]})
        else:
            st.trunc_heavy += 1
            if len(st.trunc_examples) < 8:
                st.trunc_examples.append({'severity': 'high', 'preview': text[-120:]})

        dump = analyze_code_dump(text)
        if dump.classification == 'prose':
            st.code_prose += 1
        elif dump.classification == 'educational_code':
            st.code_educational += 1
        elif dump.classification == 'mixed':
            st.code_mixed += 1
        else:
            st.code_raw_dump += 1

        cache = DocumentAnalysisCache()
        cache.bundle_for(text)
        cv = analyze_content_value(text, source=doc.source, bundle=cache.bundle_for(text))
        metrics = compute_document_metrics(
            text, source=doc.source, truncation=trunc, code_dump=dump, cache=cache,
        )
        if cv.evidence is not None:
            st.evidence_n += 1
            st.utility_sum += cv.evidence.utility
            st.confidence_sum += cv.evidence.confidence
            if cv.evidence.preserve:
                st.preserve_count += 1
            elif cv.evidence.discard_reason:
                st.semantic_discard[cv.evidence.discard_reason] += 1
        st.knowledge_sum += metrics.knowledge_density
        st.educational_sum += metrics.educational_value
        st.factual_sum += metrics.factual_density
        st.overall_sum += metrics.overall_quality
        st.completeness_sum += max(0.0, 100.0 - trunc.probability * 100.0)
        st.category_hits[getattr(cv, 'category', 'unknown')] += 1

        for domain, attr in (
            ('programming', 'code_score'),
            ('scientific', 'technical_score'),
            ('historical', 'reference_score'),
            ('mathematics', 'technical_score'),
            ('documentation', 'reference_score'),
        ):
            if float(getattr(cv, attr, 0.0)) > 0.18:
                st.domain_hits[domain] += 1
        if cv.category in ('programming', 'tutorial', 'documentation', 'scientific', 'historical', 'mathematics'):
            st.domain_hits[str(cv.category)] += 1

        sig = cache.signals_for(text)
        for fld in sig.__dataclass_fields__:
            val = getattr(sig, fld)
            if isinstance(val, (int, float)):
                st.signal_sums[fld] += float(val)

        reason = f"knowledge={metrics.knowledge_density:.0f} trunc={trunc.severity} flags={','.join(_scan_filtering_flags(text)[:3])}"
        _heap_push(st.best, (metrics.overall_quality, doc, reason), reverse=True)
        _heap_push(st.worst, (metrics.overall_quality, doc, reason), reverse=False)

    elapsed = time.time() - t0
    n = max(st.n, 1)
    lens = sorted(st.char_lens)

    def pct(p: int) -> int:
        if not lens:
            return 0
        idx = min(len(lens) - 1, int(len(lens) * p / 100))
        return lens[idx]

    def rate_from(counter: Counter, key: str) -> dict[str, Any]:
        c = counter.get(key, 0)
        ci = wilson_ci(c, n)
        return {
            'sample_count': c,
            'sample_rate_pct': round(ci['rate'] * 100, 3),
            'ci95_low_pct': round(ci['low'] * 100, 3),
            'ci95_high_pct': round(ci['high'] * 100, 3),
            'estimated_population_docs': estimate_population(ci['rate'], total_docs),
        }

    exact_dup = sum(c - 1 for c in st.exact_hashes.values() if c > 1)
    norm_dup = sum(c - 1 for c in st.norm_hashes.values() if c > 1)
    avg_sig = {k: round(v / n, 4) for k, v in st.signal_sums.items()}

    avg_knowledge = st.knowledge_sum / n
    avg_educational = st.educational_sum / n
    avg_overall = st.overall_sum / n
    avg_completeness = st.completeness_sum / n
    en_rate = st.langs.get('en', 0) / n

    cleanliness = max(
        0.0,
        min(
            100.0,
            100
            - avg_sig.get('html_score', 0) * 35
            - avg_sig.get('boilerplate_score', 0) * 30
            - avg_sig.get('seo_spam_score', 0) * 25
            - (st.flags.get('encoding_corruption', 0) / n) * 120
            - (st.metadata_flags.get('copyright_notice', 0) / n) * 40,
        ),
    )
    english_purity = max(0.0, min(100.0, en_rate * 100 - (st.mixed_lang / n) * 30))
    code_quality = max(
        0.0,
        min(100.0, 55 + (st.code_educational / n) * 35 - (st.code_raw_dump / n) * 80),
    )
    training_readiness = max(
        0.0,
        min(
            100.0,
            avg_overall * 0.35
            + cleanliness * 0.20
            + avg_knowledge * 0.20
            + english_purity * 0.15
            + avg_completeness * 0.10,
        ),
    )
    overall_score = max(
        0.0,
        min(
            100.0,
            cleanliness * 0.18
            + avg_knowledge * 0.20
            + avg_educational * 0.15
            + english_purity * 0.15
            + code_quality * 0.10
            + avg_completeness * 0.12
            + (100 - (exact_dup / n) * 200) * 0.10,
        ),
    )
    grade = score_to_grade(overall_score)

    scale = total_docs / n
    est_chars = int(st.total_chars * scale)
    est_tokens = int(est_chars / _CHARS_PER_TOKEN)

    issues_critical: list[str] = []
    issues_major: list[str] = []
    issues_minor: list[str] = []

    if st.code_raw_dump / n > 0.08:
        issues_major.append(f'Raw code dumps ~{100 * st.code_raw_dump / n:.1f}% (target <5%)')
    if st.trunc_heavy / n > 0.10:
        issues_major.append(f'Heavily truncated docs ~{100 * st.trunc_heavy / n:.1f}%')
    if en_rate < 0.85:
        issues_major.append(f'English rate ~{100 * en_rate:.1f}% — language filtering may be needed')
    if exact_dup / n > 0.02:
        issues_major.append(f'Exact duplicates in sample ~{100 * exact_dup / n:.1f}%')
    if st.flags.get('html', 0) / n > 0.05:
        issues_minor.append('HTML remnants above 5%')
    if st.flags.get('cookie_banner', 0) / n > 0.03:
        issues_minor.append('Cookie banners still present')
    if st.metadata_flags.get('copyright_notice', 0) / n > 0.08:
        issues_minor.append('Copyright notices remain in >8% of docs')
    if st.flags.get('ai_slop', 0) / n > 0.04:
        issues_minor.append('AI slop patterns detected')
    if avg_knowledge < 35:
        issues_major.append(f'Low average knowledge density ({avg_knowledge:.1f}/100)')

    pipeline_checks = {
        'low_quality_removed': avg_overall >= 45,
        'spam_reduced': (st.flags.get('seo_spam', 0) + st.flags.get('forum_junk', 0)) / n < 0.05,
        'garbage_text_low': (st.flags.get('word_salad', 0) + st.flags.get('random_symbols', 0)) / n < 0.03,
        'corruption_low': st.flags.get('encoding_corruption', 0) / n < 0.02,
        'html_mostly_removed': st.flags.get('html', 0) / n < 0.05,
        'duplicates_acceptable': exact_dup / n < 0.03,
        'metadata_reduced': st.metadata_flags.get('copyright_notice', 0) / n < 0.10,
        'code_dumps_low': st.code_raw_dump / n < 0.06,
        'truncation_acceptable': st.trunc_heavy / n < 0.08,
        'english_dominant': en_rate >= 0.88,
    }

    return {
        'meta': {
            'total_documents': total_docs,
            'sampled_documents': n,
            'confidence_level': '95%',
            'sampling_method': 'reservoir + stratified by source',
            'analysis_elapsed_sec': round(elapsed, 2),
        },
        'basic_statistics': {
            'estimated_total_tokens': est_tokens,
            'estimated_total_chars': est_chars,
            'avg_doc_length_chars': round(st.total_chars / n),
            'median_chars': pct(50),
            'p10_chars': pct(10),
            'p50_chars': pct(50),
            'p90_chars': pct(90),
            'p99_chars': pct(99),
            'empty_docs_sample': st.empty,
            'tiny_docs_lt100': rate_from(st.flags, 'tiny_doc'),
            'huge_docs_gt32k': rate_from(st.flags, 'huge_doc'),
        },
        'filtering_validation': {
            'html': rate_from(st.flags, 'html'),
            'cookie_banner': rate_from(st.flags, 'cookie_banner'),
            'navigation_menu': rate_from(st.flags, 'navigation_menu'),
            'seo_spam': rate_from(st.flags, 'seo_spam'),
            'forum_junk': rate_from(st.flags, 'forum_junk'),
            'ai_slop': rate_from(st.flags, 'ai_slop'),
            'boilerplate': rate_from(st.flags, 'boilerplate'),
            'encoding_corruption': rate_from(st.flags, 'encoding_corruption'),
            'ocr_noise': rate_from(st.flags, 'ocr_noise'),
            'repeated_punctuation': rate_from(st.flags, 'repeated_punctuation'),
            'word_salad': rate_from(st.flags, 'word_salad'),
            'random_symbols': rate_from(st.flags, 'random_symbols'),
            'synthetic_slop': rate_from(st.flags, 'synthetic_slop'),
        },
        'truncation': {
            'none_pct': round(100 * st.trunc_none / n, 2),
            'medium_slight_pct': round(100 * st.trunc_slight / n, 2),
            'high_heavy_pct': round(100 * st.trunc_heavy / n, 2),
            'estimated_heavy_population': estimate_population(st.trunc_heavy / n, total_docs),
            'examples': st.trunc_examples,
        },
        'language': {
            'distribution': dict(st.langs.most_common()),
            'english_sample_rate_pct': round(100 * en_rate, 2),
            'mixed_language_sample_rate_pct': round(100 * st.mixed_lang / n, 2),
            'english_ci95': wilson_ci(st.langs.get('en', 0), n),
            'recommendation': (
                'Language filtering recommended — English below 88%'
                if en_rate < 0.88
                else 'English purity acceptable for 150M English foundation model'
            ),
        },
        'knowledge_density': {
            'avg_score_0_100': round(avg_knowledge, 2),
            'avg_educational_value': round(avg_educational, 2),
            'avg_factual_density': round(st.factual_sum / n, 2),
            'category_distribution': dict(st.category_hits.most_common(12)),
            'domain_hits_sample': dict(st.domain_hits.most_common()),
        },
        'semantic_evidence': {
            'utility_mean': round(st.utility_sum / max(st.evidence_n, 1), 4),
            'confidence_mean': round(st.confidence_sum / max(st.evidence_n, 1), 4),
            'preserve_rate': round(st.preserve_count / max(st.evidence_n, 1), 4),
            'dominant_discard_reasons': dict(st.semantic_discard.most_common(8)),
        },
        'code_quality': {
            'prose_pct': round(100 * st.code_prose / n, 2),
            'educational_code_pct': round(100 * st.code_educational / n, 2),
            'mixed_pct': round(100 * st.code_mixed / n, 2),
            'raw_dump_pct': round(100 * st.code_raw_dump / n, 2),
            'raw_dump_ci95': wilson_ci(st.code_raw_dump, n),
        },
        'deduplication': {
            'exact_duplicates_in_sample': exact_dup,
            'exact_dup_rate_pct': round(100 * exact_dup / n, 3),
            'near_duplicates_in_sample': norm_dup,
            'near_dup_rate_pct': round(100 * norm_dup / n, 3),
            'estimated_exact_dup_population': estimate_population(exact_dup / n, total_docs),
        },
        'copyright_metadata': {
            k: rate_from(st.metadata_flags, k)
            for k in (
                'copyright_notice',
                'editorial_metadata',
                'navigation_boilerplate',
                'repo_metadata',
                'legal_footer',
                'metadata_noise',
            )
        },
        'training_suitability': {
            '150M': {'suitable': overall_score >= 55, 'expected_ppl_range': '25-45' if overall_score >= 60 else '40-80'},
            '500M': {'suitable': overall_score >= 65, 'expected_ppl_range': '18-35' if overall_score >= 65 else '35-70'},
            '1B': {'suitable': overall_score >= 72, 'expected_ppl_range': '15-28' if overall_score >= 72 else '30-60'},
            'strengths': [
                s for s in [
                    'Strong English ratio' if en_rate >= 0.9 else None,
                    'Good educational content' if avg_educational >= 50 else None,
                    'Low HTML contamination' if st.flags.get('html', 0) / n < 0.02 else None,
                    'Acceptable dedup' if exact_dup / n < 0.01 else None,
                ] if s
            ],
            'weaknesses': issues_major + issues_minor,
            'risks': [
                'Non-English contamination may hurt English-only 150M model' if en_rate < 0.92 else None,
                'Code dump leakage' if st.code_raw_dump / n > 0.05 else None,
                'Truncation noise in training signal' if st.trunc_heavy / n > 0.06 else None,
            ],
        },
        'pipeline_validation': pipeline_checks,
        'best_documents': [
            {
                'score': round(sc, 1),
                'source': doc.source,
                'chars': doc.char_len,
                'preview': doc.text[:400],
                'reason': reason,
            }
            for sc, doc, reason in st.best[:10]
        ],
        'worst_documents': [
            {
                'score': round(sc, 1),
                'source': doc.source,
                'chars': doc.char_len,
                'preview': doc.text[:400],
                'reason': reason,
            }
            for sc, doc, reason in st.worst[:10]
        ],
        'scores': {
            'cleanliness': round(cleanliness, 1),
            'knowledge_density': round(avg_knowledge, 1),
            'educational_value': round(avg_educational, 1),
            'language_quality': round(english_purity, 1),
            'code_quality': round(code_quality, 1),
            'document_completeness': round(avg_completeness, 1),
            'english_purity': round(english_purity, 1),
            'training_readiness': round(training_readiness, 1),
            'overall_dataset_quality': round(overall_score, 1),
        },
        'verdict': {
            'grade': grade,
            'ready_for_training': grade in ('A', 'B', 'C') and training_readiness >= 50,
            'critical_issues': issues_critical,
            'major_issues': issues_major,
            'minor_issues': issues_minor,
            'suggested_fixes': [
                fix for fix in [
                'Run refine_filtered_jsonl.py to trim/remove truncation and code dumps' if st.trunc_heavy / n > 0.05 or st.code_raw_dump / n > 0.04 else None,
                'Run clean_filtered_jsonl.py for remaining metadata/copyright' if st.metadata_flags.get('copyright_notice', 0) / n > 0.05 else None,
                'Enable quality_english_150m language gate at merge' if en_rate < 0.88 else None,
                'Increase dedup strictness if near-dup rate high' if norm_dup / n > 0.05 else None,
            ] if fix],
            'estimated_model_quality_impact': (
                'Low risk — suitable for 150M pretrain'
                if grade in ('A', 'B')
                else 'Moderate risk — expect noisy generation without further cleaning'
                if grade == 'C'
                else 'High risk — further cleaning strongly recommended'
            ),
        },
        'sources_in_sample': dict(st.sources.most_common(15)),
        'avg_signals': avg_sig,
        'artifact_discovery': {
            'avg_discovery_ratio': round(st.discovery_ratio_sum / n, 4),
            'learned_artifact_hits': st.discovery_artifact_hits,
            'learned_flags': dict(st.discovery_learned_flags.most_common(12)),
            'novel_artifact_discovery_rate_pct': round(100 * st.discovery_artifact_hits / n, 3),
        },
    }

def run_fast_audit(
    path: Path,
    *,
    sample_size: int = 6000,
    seed: int = 42,
    skip_line_count: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    total_docs = count_lines(path) if not skip_line_count else sample_size * 10
    _, lines = reservoir_sample_lines(path, sample_size, seed)
    docs = parse_sample_lines(lines)
    report = analyze_sample(docs, total_docs=total_docs)
    report['meta']['corpus_path'] = str(path.resolve())
    report['meta']['file_size_gb'] = round(path.stat().st_size / 1e9, 3)
    report['meta']['total_elapsed_sec'] = round(time.time() - t0, 2)
    return report
