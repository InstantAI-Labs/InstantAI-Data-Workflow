from __future__ import annotations

import time
from typing import Any

from indw.dedup.normalize import content_hash
from indw.clean.corpus import CorpusCleaningPipeline, extract_row_text
from indw.clean.document.normalize import minimal_normalize_detail
from indw.filter.spec.quality import QualityPipelineConfig
from indw.filter.gate.quality import QualityGate
from indw.schedule.monitor.budget import (
    clear_doc_deadline,
    doc_budget_exceeded,
    resolve_doc_wall_budget_sec,
    set_doc_deadline,
)
from indw.schedule.monitor.doc import monitor_begin_doc, monitor_end_doc, set_doc_stage
from indw.schedule.intel.session import get_acim_session
from indw.schedule.intel.pci import build_fingerprint_bundle_detail
from indw.schedule.read.gates import parse_merge_jsonl_line
from indw.extract.core.context import (
    DocumentExecutionContext,
    bind_document_context,
    clear_document_context,
)
from indw.schedule.state.context import MergeDocumentContext
from indw.schedule.dispatch.alloc import (
    STAGE_FAST_PREPROCESS,
    STAGE_HIGH_QUALITY,
    STAGE_INTEL_PREVIEW,
    STAGE_INTERMEDIATE,
)


def _blank_result(ctx: MergeDocumentContext, *, kind: str) -> dict[str, Any]:
    return {
        'seq': ctx.seq,
        'src_name': ctx.src_name,
        'line_no': ctx.line_no,
        'kind': kind,
        'cleaning_rejects': [],
        'chunks': [],
        'stage_trace': ctx.stage_trace,
    }


def _processed_result(ctx: MergeDocumentContext) -> dict[str, Any]:
    out: dict[str, Any] = {
        'seq': ctx.seq,
        'src_name': ctx.src_name,
        'line_no': ctx.line_no,
        'kind': ctx.kind,
        'cleaning_rejects': ctx.cleaning_rejects,
        'chunks': ctx.chunks,
        'raw_text': ctx.text,
        'row': ctx.row,
        'provenance': ctx.provenance,
        'stage_trace': ctx.stage_trace,
    }
    if ctx.language_assessment is not None:
        out['language_assessment'] = ctx.language_assessment
    if ctx.pci_fp is not None:
        out['pci_fp'] = ctx.pci_fp
    if ctx.acim_intel is not None:
        out['acim_intel'] = ctx.acim_intel
    if ctx.acim_route is not None:
        out['acim_route'] = ctx.acim_route
    if ctx.lci is not None:
        out['lci'] = ctx.lci
    if ctx.doc_tier:
        out['doc_tier'] = ctx.doc_tier
    if ctx.admission is not None:
        out['admission'] = ctx.admission
    if ctx.ingest_meta is not None:
        out['ingest_meta'] = ctx.ingest_meta
    return out


def _stage1_fast_preprocess(
    ctx: MergeDocumentContext,
    *,
    line: str,
    row: dict[str, Any] | None,
) -> str | None:
    ctx.mark(STAGE_FAST_PREPROCESS)
    set_doc_stage('fast_preprocess')
    if row is None:
        kind, row = parse_merge_jsonl_line(line)
        if kind == 'blank':
            ctx.kind = 'blank'
            return 'stop'
        if kind == 'parse_error':
            ctx.kind = 'parse_error'
            return 'stop'
        ctx.row = row
    text, meaningful = minimal_normalize_detail(extract_row_text(row or {}))
    ctx.meaningful_chars = meaningful
    if not text:
        ctx.kind = 'empty_text'
        return 'stop'
    ctx.text = text
    return None


def _stage3_intermediate(ctx: MergeDocumentContext, *, src_name: str) -> None:
    ctx.mark(STAGE_INTERMEDIATE)
    set_doc_stage('intermediate_intel')
    fp, scan = build_fingerprint_bundle_detail(ctx.text, raw=ctx.raw_features)
    ctx.fp = fp
    ctx.fp_scan = scan
    ctx.pci_fp = fp.to_dict()
    doc_id = f'{src_name}:{ctx.line_no}'
    monitor_begin_doc(
        seq=ctx.seq,
        src_name=src_name,
        line_no=ctx.line_no,
        chars=len(ctx.text),
        fingerprint=fp.structural,
        doc_id=doc_id,
        words=scan.word_count,
    )
    bind_document_context(DocumentExecutionContext(
        normalized_text=ctx.text,
        pci_fp=ctx.pci_fp,
        source=src_name,
        gate_raw=ctx.raw_features,
    ))


def _stage4_intel_preview(ctx: MergeDocumentContext, *, src_name: str) -> None:
    ctx.mark(STAGE_INTEL_PREVIEW)
    acim_sess = get_acim_session()
    if acim_sess is None or not acim_sess.enabled or ctx.fp is None:
        return
    acim_intel, acim_route, lci_payload = acim_sess.preview_for_cleaning(
        ctx.text, source=src_name, fp=ctx.fp, scan=ctx.fp_scan,
    )
    ctx.acim_intel = acim_intel
    ctx.acim_route = acim_route
    ctx.lci = lci_payload
    if acim_intel is not None:
        bind_document_context(DocumentExecutionContext(
            normalized_text=ctx.text,
            pci_fp=acim_intel.get('fp'),
            source=src_name,
            gate_raw=ctx.raw_features,
        ))
        ctx.pci_fp = acim_intel.get('fp')


def _stage4_high_quality(
    ctx: MergeDocumentContext,
    *,
    cleaning_pipeline: CorpusCleaningPipeline,
    src_name: str,
) -> str | None:
    ctx.mark(STAGE_HIGH_QUALITY)
    set_doc_stage('cleaning')
    set_doc_deadline(time.perf_counter() + resolve_doc_wall_budget_sec())
    budget_exceeded = False
    try:
        clean_results = cleaning_pipeline.process(
            ctx.text, source=src_name, row=ctx.row, pre_normalized=True,
        )
        budget_exceeded = doc_budget_exceeded()
    finally:
        clear_doc_deadline()

    if budget_exceeded:
        ctx.reject('document_budget_exceeded')
        return 'document_budget_exceeded'

    for dropped in clean_results:
        if dropped.dropped:
            ctx.reject(dropped.drop_reason or 'cleaning')

    set_doc_stage('chunk_attach')
    for clean_result in clean_results:
        if clean_result.dropped or not clean_result.text:
            continue
        chunk_text = clean_result.text
        ctx.chunks.append({
            'chunk_text': chunk_text,
            'content_hash': content_hash(chunk_text),
            'clean_result': clean_result,
            'doc': None,
        })
    return None


def fast_terminal_line(ctx: MergeDocumentContext, needs_heavy: bool) -> dict[str, Any] | None:
    if needs_heavy:
        return None
    if ctx.kind in ('blank', 'parse_error', 'empty_text'):
        return _blank_result(ctx, kind=ctx.kind)
    return _processed_result(ctx)


def run_fast_stages(
    *,
    line: str,
    src_name: str,
    line_no: int,
    seq: int,
    gate: QualityGate,
    row: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    ingest_meta: dict[str, Any] | None = None,
    stage_profile: Any = None,
) -> tuple[MergeDocumentContext, bool]:
    t0 = time.perf_counter()
    ctx = MergeDocumentContext(
        seq=seq,
        src_name=src_name,
        line_no=line_no,
        row=row,
        provenance=provenance,
        ingest_meta=ingest_meta,
    )
    if stage_profile is not None:
        with stage_profile.timed(STAGE_FAST_PREPROCESS):
            stop = _stage1_fast_preprocess(ctx, line=line, row=row)
    else:
        stop = _stage1_fast_preprocess(ctx, line=line, row=row)
    if stop == 'stop':
        _audit_fast(ctx, False, t0)
        return ctx, False

    from indw.schedule.admission.tier01 import run_tier01_gates
    reject = run_tier01_gates(ctx, gate=gate, src_name=src_name, row=row)
    if reject:
        _audit_fast(ctx, False, t0)
        return ctx, False

    _audit_fast(ctx, True, t0)
    return ctx, True


def _audit_fast(ctx: MergeDocumentContext, needs_heavy: bool, t0: float) -> None:
    from indw.filter.stage0.audit import audit_enabled, record_fast_exit
    if audit_enabled():
        record_fast_exit(ctx, needs_heavy, wall_ms=(time.perf_counter() - t0) * 1000.0)


def run_heavy_stages(
    ctx: MergeDocumentContext,
    *,
    cleaning_pipeline: CorpusCleaningPipeline,
    src_name: str,
    stage_profile: Any = None,
    audit_lane: str = '',
    audit_path: str = 'heavy_pool',
) -> dict[str, Any]:
    from indw.filter.stage0.audit import audit_enabled, record_heavy_enter, record_heavy_exit
    heavy_t0 = time.perf_counter()
    if audit_enabled():
        record_heavy_enter(ctx, lane=audit_lane, path=audit_path)
    outcome = 'ok'

    def _run(stage: str, fn) -> None:
        if stage_profile is not None:
            with stage_profile.timed(stage):
                fn()
        else:
            fn()

    try:
        _run(STAGE_INTERMEDIATE, lambda: _stage3_intermediate(ctx, src_name=src_name))
        _run(STAGE_INTEL_PREVIEW, lambda: _stage4_intel_preview(ctx, src_name=src_name))
        reject = None
        if stage_profile is not None:
            with stage_profile.timed(STAGE_HIGH_QUALITY):
                reject = _stage4_high_quality(
                    ctx, cleaning_pipeline=cleaning_pipeline, src_name=src_name,
                )
        else:
            reject = _stage4_high_quality(
                ctx, cleaning_pipeline=cleaning_pipeline, src_name=src_name,
            )
        if reject:
            outcome = reject
        result = _processed_result(ctx)
        if audit_enabled():
            record_heavy_exit(ctx, wall_ms=(time.perf_counter() - heavy_t0) * 1000.0, path=audit_path)
        return result
    finally:
        monitor_end_doc(outcome=outcome)
        clear_document_context()


def run_progressive_preprocess(
    *,
    line: str,
    src_name: str,
    line_no: int,
    seq: int,
    cleaning_pipeline: CorpusCleaningPipeline,
    gate: QualityGate,
    cfg: QualityPipelineConfig,
    provenance: dict[str, Any] | None = None,
    row: dict[str, Any] | None = None,
    stage_profile: Any = None,
    ingest_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del cfg
    ctx, needs_heavy = run_fast_stages(
        line=line,
        src_name=src_name,
        line_no=line_no,
        seq=seq,
        gate=gate,
        row=row,
        provenance=provenance,
        ingest_meta=ingest_meta,
        stage_profile=stage_profile,
    )
    terminal = fast_terminal_line(ctx, needs_heavy)
    if terminal is not None:
        return terminal
    return run_heavy_stages(
        ctx,
        cleaning_pipeline=cleaning_pipeline,
        src_name=src_name,
        stage_profile=stage_profile,
    )
