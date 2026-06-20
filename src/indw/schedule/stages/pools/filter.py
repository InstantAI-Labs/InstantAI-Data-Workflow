from __future__ import annotations

from typing import Any

from indw.filter.content.domain import domain_from_source
from indw.schedule.state.context import MergeDocumentContext
from indw.schedule.dispatch.alloc import STAGE_FAST_FILTER
from indw.filter.stage0.engine import worker_doc_max_chars
from indw.schedule.read.gates import (
    early_document_max_gate,
    early_document_size_gate,
)
from indw.schedule.admission.tiers import TIER0


def process_filter_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    terminal: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    for item in batch:
        ctx = MergeDocumentContext.from_survivor_payload(item)
        ctx.text = str(item.get('text') or ctx.text)
        ctx.meaningful_chars = int(item.get('meaningful_chars') or ctx.meaningful_chars or 0)
        src_name = str(item['src_name'])
        ctx.mark(STAGE_FAST_FILTER)
        domain = domain_from_source(src_name)
        max_chars = worker_doc_max_chars()
        size_reason = early_document_size_gate(
            ctx.text, src_name, meaningful_chars=ctx.meaningful_chars, domain=domain,
        )
        if size_reason:
            ctx.reject(size_reason)
            terminal.append(_processed(ctx, tier=TIER0))
            continue
        max_reason = early_document_max_gate(ctx.text, src_name, domain=domain, max_chars=max_chars)
        if max_reason:
            ctx.reject(max_reason)
            terminal.append(_processed(ctx, tier=TIER0))
            continue
        survivors.append(ctx.survivor_payload(work_dir=item.get('_work_dir')))
    return {'terminal': terminal, 'survivors': survivors}


def _processed(ctx: MergeDocumentContext, *, tier: int = TIER0) -> dict[str, Any]:
    out: dict[str, Any] = {
        'seq': ctx.seq,
        'src_name': ctx.src_name,
        'line_no': ctx.line_no,
        'kind': 'processed',
        'cleaning_rejects': ctx.cleaning_rejects,
        'chunks': ctx.chunks,
        'raw_text': ctx.text,
        'row': ctx.row,
        'provenance': ctx.provenance,
        'stage_trace': ctx.stage_trace,
        'language_assessment': ctx.language_assessment,
        '_reject_tier': tier,
    }
    return out
