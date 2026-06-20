from __future__ import annotations

from typing import Any

from indw.clean.corpus import extract_row_text
from indw.clean.document.normalize import minimal_normalize_detail
from indw.schedule.read.gates import parse_merge_jsonl_line
from indw.schedule.state.context import MergeDocumentContext
from indw.schedule.dispatch.alloc import STAGE_FAST_PREPROCESS


def process_preprocess_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    terminal: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    for item in batch:
        line = str(item.get('line') or '')
        row = item.get('row')
        ctx = MergeDocumentContext(
            seq=int(item['seq']),
            src_name=str(item['src_name']),
            line_no=int(item['line_no']),
            row=row,
            provenance=item.get('provenance'),
            ingest_meta=item.get('ingest_meta'),
        )
        ctx.mark(STAGE_FAST_PREPROCESS)
        if row is None:
            kind, row = parse_merge_jsonl_line(line)
            if kind == 'blank':
                ctx.kind = 'blank'
                terminal.append(_terminal(ctx))
                continue
            if kind == 'parse_error':
                ctx.kind = 'parse_error'
                terminal.append(_terminal(ctx))
                continue
            ctx.row = row
        text, meaningful = minimal_normalize_detail(extract_row_text(row or {}))
        ctx.meaningful_chars = meaningful
        if not text:
            ctx.kind = 'empty_text'
            terminal.append(_terminal(ctx))
            continue
        ctx.text = text
        survivors.append(_survivor(ctx, item))
    return {'terminal': terminal, 'survivors': survivors}


def _terminal(ctx: MergeDocumentContext) -> dict[str, Any]:
    return {
        'seq': ctx.seq,
        'src_name': ctx.src_name,
        'line_no': ctx.line_no,
        'kind': ctx.kind,
        'cleaning_rejects': [],
        'chunks': [],
        'stage_trace': ctx.stage_trace,
    }


def _survivor(ctx: MergeDocumentContext, item: dict[str, Any]) -> dict[str, Any]:
    out = ctx.survivor_payload(work_dir=item.get('_work_dir'))
    if item.get('provenance') is not None:
        out['provenance'] = item['provenance']
    if item.get('ingest_meta') is not None:
        out['ingest_meta'] = item['ingest_meta']
    return out
