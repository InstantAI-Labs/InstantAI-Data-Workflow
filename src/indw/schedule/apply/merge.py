from __future__ import annotations

import logging
from typing import Any

from indw.schedule.row.index import mixture_index_row
from indw.schedule.row.reject import record_merge_reject
from indw.schedule.apply.serialize import preprocessed_line_to_objects

logger = logging.getLogger(__name__)
_intel_log = logging.getLogger('merge.intel')


def apply_merge_preprocessed_line(
    line: dict[str, Any],
    *,
    cleaning_pipeline: Any,
    gate: Any,
    cfg: Any,
    checkpoint: Any,
    exact: Any,
    fuzzy: Any,
    semantic: Any,
    embed_semantic: Any = None,
    sink: Any,
    index_file: Any,
    scan_counters: dict[str, int],
    license_policy: Any = None,
    validation_collector: Any = None,
    pipeline_metrics: Any = None,
    stop_requested: bool = False,
    reject_log: Any = None,
    pci: Any = None,
) -> bool:
    chunks_in = line.get('chunks') or []
    if not line.get('_merge_objects_ready'):
        if any(isinstance(chunk.get('clean_result'), dict) for chunk in chunks_in):
            line = preprocessed_line_to_objects(line)
            line['_merge_objects_ready'] = True
    if pci is not None:
        try:
            pci.observe_preprocessed(line)
        except Exception as exc:
            from indw.schedule.monitor.obs import is_debug, is_validation
            if is_validation() or is_debug():
                _intel_log.warning(
                    'observe_preprocessed failed seq=%s: %s',
                    line.get('seq'),
                    exc,
                )
    src_name = line['src_name']
    line_no = line['line_no']
    src_state = checkpoint.source(src_name)
    src_state.line_offset = line_no + 1
    kind = line.get('kind', 'processed')

    if kind == 'blank':
        return False
    if kind == 'parse_error':
        scan_counters['skipped_parse'] += 1
        return False
    if kind == 'empty_text':
        src_state.scanned += 1
        scan_counters['total_scanned'] += 1
        gate.stats.record_reject('empty', 0)
        src_state.rejected += 1
        return False

    src_state.scanned += 1
    scan_counters['total_scanned'] += 1
    chunks = line.get('chunks') or []
    cleaning_rejects = line.get('cleaning_rejects') or []
    for reason, text_len in cleaning_rejects:
        gate.stats.record_reject(reason, text_len)
        src_state.rejected += 1
        record_merge_reject(
            reject_log,
            reason=reason,
            source=src_name,
            text=line.get('raw_text') or '',
        )

    kept_any = write_merge_chunks(
        chunks=chunks,
        src_name=src_name,
        src_state=src_state,
        cfg=cfg,
        gate=gate,
        exact=exact,
        fuzzy=fuzzy,
        semantic=semantic,
        embed_semantic=embed_semantic,
        sink=sink,
        index_file=index_file,
        license_policy=license_policy,
        validation_collector=validation_collector,
        pipeline_metrics=pipeline_metrics,
        stop_requested=stop_requested,
        provenance=line.get('provenance'),
        language_assessment=line.get('language_assessment'),
        reject_log=reject_log,
    )
    if kind == 'processed' and not chunks and not cleaning_rejects:
        gate.stats.record_reject('empty_after_cleaning', 0)
        src_state.rejected += 1
        record_merge_reject(
            reject_log,
            reason='empty_after_cleaning',
            source=src_name,
            text=line.get('raw_text') or '',
        )
    return kept_any


def write_merge_chunks(
    *,
    chunks: list[dict[str, Any]],
    src_name: str,
    src_state: Any,
    cfg: Any,
    gate: Any,
    exact: Any,
    fuzzy: Any,
    semantic: Any,
    embed_semantic: Any = None,
    sink: Any,
    index_file: Any,
    license_policy: Any = None,
    validation_collector: Any = None,
    pipeline_metrics: Any = None,
    stop_requested: bool = False,
    provenance: dict[str, Any] | None = None,
    language_assessment: Any = None,
    reject_log: Any = None,
) -> bool:
    from indw.schedule.read.gates import attach_analysis_cache
    from indw.schedule.dispatch.alloc import STAGE_FINAL_VALIDATION, STAGE_OUTPUT
    from indw.clean.document.validate import validate_chunk
    from indw.filter.gate.scorer import score_document
    from indw.filter.license.record import build_provenance_record

    kept_any = False
    from indw.schedule.monitor.doc import set_doc_stage
    set_doc_stage(STAGE_FINAL_VALIDATION)
    for chunk in chunks:
        if stop_requested:
            break
        chunk_text = chunk['chunk_text']
        clean_result = chunk['clean_result']
        text_len = len(chunk_text)
        if cfg.dedup.exact and exact.is_duplicate(
            chunk_text,
            source=src_name,
            digest=chunk.get('content_hash'),
        ):
            gate.stats.record_reject('exact_dup', text_len)
            src_state.rejected += 1
            record_merge_reject(
                reject_log, reason='exact_dup', source=src_name, text=chunk_text,
            )
            continue
        doc = chunk.get('doc')
        if doc is None:
            if getattr(clean_result, 'analysis_bundle', None) is None:
                attach_analysis_cache(clean_result, cfg)
            score_kwargs: dict[str, Any] = {}
            bundle = getattr(clean_result, 'analysis_bundle', None)
            if bundle is not None:
                score_kwargs['analysis_bundle'] = bundle
                score_kwargs['analysis_scan'] = getattr(clean_result, 'analysis_scan', '') or None
                full_len = int(getattr(clean_result, 'analysis_full_len', 0) or 0)
                if full_len > 0:
                    score_kwargs['analysis_full_len'] = full_len
            if language_assessment is not None and gate._language_policy.skip_post_clean_detection:
                score_kwargs['prechecked_language'] = language_assessment
                gate.language_stats.skipped_post_clean += 1
            doc = score_document(
                chunk_text,
                source=src_name,
                duplicate_ratio=clean_result.metrics.duplicate_ratio,
                thresholds=cfg.thresholds,
                gate=gate,
                provenance=provenance,
                **score_kwargs,
            )
            chunk['doc'] = doc
        skip_near_dedup = (
            cfg.dedup.skip_within_document_chunks
            and clean_result.chunk_index > 0
        )
        ok, doc = gate.finalize_scored_document(doc, chunk_text, source=src_name)
        if pipeline_metrics is not None:
            pipeline_metrics.inc_document()
        if not ok:
            src_state.rejected += 1
            reason = doc.reject_reason or 'rejected'
            record_merge_reject(
                reject_log,
                reason=reason,
                source=src_name,
                text=chunk_text,
                doc=doc,
            )
            if validation_collector is not None:
                validation_collector.record_rejected(
                    src_name, chunk_text, reason, doc
                )
            continue
        valid, validation_reason = validate_chunk(
            chunk_text,
            cfg.cleaning,
            skip_quality_check=True,
            gate_approved=True,
            quality_score=doc.score,
            gate_min_chars=cfg.thresholds.min_chars,
        )
        if not valid:
            reason = f'validation_{validation_reason}'
            gate.stats.record_reject(reason, text_len)
            src_state.rejected += 1
            record_merge_reject(
                reject_log,
                reason=reason,
                source=src_name,
                text=chunk_text,
                doc=doc,
            )
            continue
        if not skip_near_dedup:
            if fuzzy is not None and not fuzzy.should_keep(chunk_text, doc.quality_score_10):
                gate.stats.compensate_pre_gate_keep(doc, source=src_name)
                gate.stats.record_reject('near_dup_fuzzy', text_len)
                src_state.rejected += 1
                record_merge_reject(
                    reject_log, reason='near_dup_fuzzy', source=src_name, text=chunk_text, doc=doc,
                )
                continue
            if semantic is not None and not semantic.should_keep(chunk_text, doc.quality_score_10):
                gate.stats.compensate_pre_gate_keep(doc, source=src_name)
                gate.stats.record_reject('near_dup_semantic', text_len)
                src_state.rejected += 1
                record_merge_reject(
                    reject_log, reason='near_dup_semantic', source=src_name, text=chunk_text, doc=doc,
                )
                continue
            if embed_semantic is not None and not embed_semantic.evaluate_and_register(
                chunk_text,
                doc.quality_score_10,
                doc=doc,
            ):
                gate.stats.compensate_pre_gate_keep(doc, source=src_name)
                gate.stats.record_reject('near_dup_embed', text_len)
                src_state.rejected += 1
                record_merge_reject(
                    reject_log, reason='near_dup_embed', source=src_name, text=chunk_text, doc=doc,
                )
                continue
        if not skip_near_dedup or clean_result.chunk_index == 0:
            if fuzzy is not None:
                fuzzy.register(chunk_text, doc.quality_score_10)
            if semantic is not None:
                semantic.register(chunk_text, doc.quality_score_10)
        if (
            license_policy is not None
            and license_policy.include_provenance_in_jsonl
            and doc.license_assessment is not None
        ):
            set_doc_stage(STAGE_OUTPUT)
            record = build_provenance_record(chunk_text, doc.license_assessment)
            sink.write_row(chunk_text, record=record)
        else:
            set_doc_stage(STAGE_OUTPUT)
            sink.write_row(chunk_text)
        src_state.kept += 1
        kept_any = True
        if validation_collector is not None:
            validation_collector.record_accepted(src_name, chunk_text, doc)
        if index_file is not None:
            index_file.write(mixture_index_row(
                src_name=src_name,
                doc=doc,
                clean_result=clean_result,
            ))
    return kept_any
