from __future__ import annotations

from typing import Any

from indw.clean.gate.evaluate import document_gate_raw
from indw.dedup.normalize import content_hash
from indw.filter.stage0.admission import evaluate_admission
from indw.filter.stage0.engine import run_stage0_content_filters, worker_gate_policy
from indw.schedule.dispatch.alloc import (
    STAGE_DOC_DEDUP,
    STAGE_METADATA,
    STAGE_STRUCTURAL_FILTER,
    STAGE_ADMISSION,
    STAGE_FAST_FILTER,
)
from indw.schedule.read.gates import early_language_gate, _ensure_language_warm
from indw.schedule.read.ingest import enrich_ingest_meta
from indw.schedule.state.context import MergeDocumentContext
from indw.schedule.stages.engine import _processed_result
from indw.schedule.admission.tiers import TIER0, TIER1


def process_stage0_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    from indw.schedule.dispatch.workers import fast_doc_dedup_check, _FAST_CTX
    from indw.filter.gate.quality import QualityGate

    gate: QualityGate | None = None
    if _FAST_CTX is not None:
        gate = _FAST_CTX.get('gate')
    if gate is not None:
        _ensure_language_warm(gate)

    terminal: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    pol = worker_gate_policy()
    for item in batch:
        ctx = MergeDocumentContext.from_survivor_payload(item)
        src_name = str(item['src_name'])

        if gate is not None and ctx.language_assessment is None:
            ctx.mark(STAGE_FAST_FILTER)
            lang_reason, lang_assessment = early_language_gate(
                ctx.text, gate, meaningful_chars=ctx.meaningful_chars,
            )
            ctx.language_assessment = lang_assessment
            if lang_reason:
                ctx.reject(lang_reason)
                terminal.append(_terminal(ctx, TIER1))
                continue

        ctx.mark(STAGE_DOC_DEDUP)
        ctx.doc_content_hash = content_hash(ctx.text)
        if fast_doc_dedup_check(digest=ctx.doc_content_hash, source=src_name):
            ctx.reject('exact_doc_dup')
            terminal.append(_terminal(ctx, TIER1))
            continue

        if not item.get('stage0_cleared'):
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
                terminal.append(_terminal(ctx, TIER1))
                continue
        elif item.get('raw_features') is not None:
            ctx.raw_features = item.get('raw_features')

        ctx.mark(STAGE_METADATA)
        if ctx.ingest_meta is not None:
            ctx.ingest_meta = enrich_ingest_meta(ctx.ingest_meta, row=ctx.row, text=ctx.text)
        ctx.mark(STAGE_ADMISSION)
        decision = evaluate_admission(meaningful_chars=ctx.meaningful_chars)
        ctx.doc_tier = decision.tier
        ctx.admission = decision.to_dict()
        payload = ctx.survivor_payload(work_dir=item.get('_work_dir'))
        payload['stage0_cleared'] = True
        if ctx.raw_features is not None:
            payload['raw_features'] = ctx.raw_features
        survivors.append(payload)
    return {'terminal': terminal, 'survivors': survivors}


def _terminal(ctx: MergeDocumentContext, tier: int) -> dict[str, Any]:
    out = _processed_result(ctx)
    out['_reject_tier'] = tier
    return out
