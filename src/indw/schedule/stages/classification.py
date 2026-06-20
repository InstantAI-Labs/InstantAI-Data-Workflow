from __future__ import annotations

import re

from indw.filter.score.signals import compute_signals
from indw.clean.document.value import analyze_content_value, build_analysis_bundle
from indw.filter.license.classifier import classify_document_type
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import ContentClassification, CorpusDocument

def classify_document(doc: CorpusDocument, policy: PipelinePolicy) -> CorpusDocument:
    if not policy.classification.enabled or not doc.text:
        return doc.with_stage('content_classification')
    th = policy.classification.thresholds
    text = doc.text
    bundle = build_analysis_bundle(text)
    cv = analyze_content_value(text, source=doc.provenance.source, bundle=bundle)
    sig = compute_signals(text)
    doc_type = classify_document_type(text=text, source=doc.provenance.source, url=doc.provenance.url)
    from indw.tools.reports.fast.analyze import _scan_filtering_flags
    scan_flags = list(_scan_filtering_flags(text))
    commercial_dominant = (
        (cv.commercial_score > th.content_commercial or sig.commercial_score > th.signals_commercial)
        and cv.overall_value_score < th.max_overall_value
        and cv.educational_score < th.max_educational
        and cv.technical_score < th.max_technical
    )
    if commercial_dominant:
        scan_flags.append('commercial_content')
    if re.search(r'(?i)(?:\$[\d.]+|sku\s*:|checkout\?|%\s*off\b|register to reserve)', text):
        if cv.overall_value_score < th.low_value_overall:
            scan_flags.append('commercial_content')
    if cv.category in ('forum', 'entertainment') and cv.overall_value_score < th.low_value_overall:
        scan_flags.append('low_value_news')
    if doc_type in ('government',) and cv.overall_value_score < th.government_low_value:
        scan_flags.append('low_value_news')
    if cv.category == 'entertainment' or 'entertainment' in text.lower()[:200]:
        scan_flags.append('entertainment_clickbait')
    if (
        'answer:' in text.lower()
        and cv.word_count < th.scaffold_max_words
        and 'you are an ai assistant' in text.lower()
    ):
        scan_flags.append('instruction_scaffold_only')
    from indw.filter.content.code import analyze_code, vendor_sdk_hits
    code_sig = analyze_code(text)
    structured_algo = (
        code_sig is not None
        and code_sig.syntax_balance >= th.structured_code_syntax
        and 'procedure' in text.lower()
        and (
            code_sig.educational_score >= th.structured_code_educational
            or 'spark_mode' in text.lower()
            or re.search(r'(?i)\bprocedure\s+sort\b', text)
        )
    )
    if (vendor_sdk_hits(text) >= 1 or 'package body' in text.lower()) and not structured_algo:
        scan_flags.append('vendor_sdk_dump')
    prior_flags = set(doc.flags)
    cleaned_answer = len(doc.raw_text.split()) >= th.scaffold_max_words and cv.word_count < 25
    if (
        cv.word_count < th.metadata_only_max_words
        and cv.fact_count < th.metadata_only_max_facts
        and cv.educational_score < th.metadata_only_max_educational
        and 'qa_normalized' not in prior_flags
        and 'qa_tail_removed' not in prior_flags
        and not cleaned_answer
    ):
        scan_flags.append('license_or_metadata_only')
    from indw.clean.meta.foundation import is_metadata_only_document
    if is_metadata_only_document(text):
        scan_flags.append('license_or_metadata_only')
    license_hits = len(re.findall(
        r'(?i)(?:\bmit license\b|\bapache license\b|permission is hereby granted|all rights reserved|without warranty)',
        text,
    ))
    if license_hits >= th.license_min_hits and cv.overall_value_score < th.license_max_overall:
        scan_flags.append('license_or_metadata_only')
    if re.search(r'(?i)(?:\(\d{3}\)\s*\d{3}-\d{4}|www\.\S+\s+for reservations)', text):
        if cv.educational_score < th.commercial_edu_floor:
            scan_flags.append('commercial_content')
    if sig.synthetic_score > policy.rewrite.max_synthetic_score:
        scan_flags.append('synthetic_spam')
    flags = tuple(dict.fromkeys(list(doc.flags) + scan_flags))
    content_type = (
        'code' if sig.code_density > th.code_density_text
        else ('mixed' if sig.code_density > th.code_density_mixed else 'text')
    )
    classification = ContentClassification(
        category=cv.category,
        document_type=doc_type,
        content_type=content_type,
        language='unknown',
        flags=flags,
    )
    return doc.with_classification(classification).with_flags(flags)
