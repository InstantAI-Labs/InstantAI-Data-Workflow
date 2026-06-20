from __future__ import annotations

from typing import Any

from indw.clean.corpus import CorpusCleaningPipeline
from indw.clean.document.stats import CleaningStats
from indw.extract.core.context import clear_document_context
from indw.filter.stage0.verify import validate_survivor_payload
from indw.schedule.dispatch.lanes import survivor_lane
from indw.schedule.monitor.doc import monitor_end_doc
from indw.schedule.stages.engine import _processed_result, _stage4_high_quality
from indw.schedule.state.context import MergeDocumentContext


def process_clean_batch(survivors: list[dict[str, Any]]) -> dict[str, Any]:
    from indw.schedule.dispatch.workers import _HEAVY_CTX
    if _HEAVY_CTX is None:
        raise RuntimeError('clean pool worker not initialized')
    cleaning_pipeline: CorpusCleaningPipeline = _HEAVY_CTX['cleaning_pipeline']
    cleaning_pipeline.stats = CleaningStats()
    work_dir = (_HEAVY_CTX or {}).get('work_dir')
    out: list[dict[str, Any]] = []
    for raw in survivors:
        payload = dict(raw)
        if work_dir:
            payload.setdefault('_work_dir', work_dir)
        if not str(payload.get('text') or '').strip():
            from indw.filter.stage0.verify import resolve_survivor_payload_text
            resolved = resolve_survivor_payload_text(payload)
            if resolved:
                payload['text'] = resolved
        validate_survivor_payload(payload)
        ctx = MergeDocumentContext.from_survivor_payload(payload)
        src_name = str(payload['src_name'])
        outcome = 'ok'
        try:
            from indw.extract.core.context import DocumentExecutionContext, bind_document_context
            bind_document_context(DocumentExecutionContext(
                normalized_text=ctx.text,
                pci_fp=ctx.pci_fp,
                source=src_name,
                gate_raw=ctx.raw_features,
            ))
            reject = _stage4_high_quality(
                ctx, cleaning_pipeline=cleaning_pipeline, src_name=src_name,
            )
            if reject:
                outcome = reject
            out.append(_processed_result(ctx))
        finally:
            monitor_end_doc(outcome=outcome)
            clear_document_context()
    discovery_cal = cleaning_pipeline.end_discovery_batch()
    result: dict[str, Any] = {'items': out, 'cleaning_stats': cleaning_pipeline.stats}
    if discovery_cal is not None:
        result['discovery_calibration'] = discovery_cal
    return result
