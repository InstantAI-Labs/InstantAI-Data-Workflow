from __future__ import annotations

from typing import Any

from indw.clean.gate.evaluate import document_gate_raw
from indw.dedup.normalize import content_hash
from indw.filter.content.domain import domain_from_source
from indw.filter.stage0.admission import evaluate_admission
from indw.filter.stage0.engine import (
    run_stage0_content_filters,
    worker_doc_max_chars,
    worker_gate_policy,
)
from indw.schedule.read.gates import (
    early_document_max_gate,
    early_document_size_gate,
    early_language_gate,
)
from indw.schedule.read.ingest import enrich_ingest_meta
from indw.schedule.state.context import MergeDocumentContext
from indw.schedule.dispatch.alloc import (
    STAGE_DOC_DEDUP,
    STAGE_FAST_FILTER,
    STAGE_METADATA,
    STAGE_STRUCTURAL_FILTER,
    STAGE_ADMISSION,
)


def run_tier01_gates(
    ctx: MergeDocumentContext,
    *,
    gate: Any,
    src_name: str,
    row: dict[str, Any] | None = None,
) -> str | None:
    domain = domain_from_source(src_name)
    max_chars = worker_doc_max_chars()
    pol = worker_gate_policy()

    ctx.mark(STAGE_FAST_FILTER)
    size_reason = early_document_size_gate(
        ctx.text, src_name, meaningful_chars=ctx.meaningful_chars, domain=domain,
    )
    if size_reason:
        ctx.reject(size_reason)
        return size_reason

    max_reason = early_document_max_gate(ctx.text, src_name, domain=domain, max_chars=max_chars)
    if max_reason:
        ctx.reject(max_reason)
        return max_reason

    lang_reason, lang_assessment = early_language_gate(
        ctx.text, gate, meaningful_chars=ctx.meaningful_chars,
    )
    ctx.language_assessment = lang_assessment
    if lang_reason:
        ctx.reject(lang_reason)
        return lang_reason

    ctx.mark(STAGE_DOC_DEDUP)
    ctx.doc_content_hash = content_hash(ctx.text)
    from indw.schedule.dispatch.workers import fast_doc_dedup_check
    if fast_doc_dedup_check(digest=ctx.doc_content_hash, source=src_name):
        ctx.reject('exact_doc_dup')
        return 'exact_doc_dup'

    ctx.mark(STAGE_STRUCTURAL_FILTER)
    ctx.raw_features = document_gate_raw(ctx.text)
    reject = run_stage0_content_filters(
        ctx.text,
        meaningful_chars=ctx.meaningful_chars,
        raw=ctx.raw_features,
        pol=pol,
        normalized=True,
    )
    if reject:
        ctx.reject(reject)
        return reject

    ctx.mark(STAGE_METADATA)
    if ctx.ingest_meta is not None:
        ctx.ingest_meta = enrich_ingest_meta(ctx.ingest_meta, row=row or ctx.row, text=ctx.text)

    ctx.mark(STAGE_ADMISSION)
    decision = evaluate_admission(meaningful_chars=ctx.meaningful_chars)
    ctx.doc_tier = decision.tier
    ctx.admission = decision.to_dict()
    return None
