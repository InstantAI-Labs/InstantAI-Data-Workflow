from __future__ import annotations

from typing import Any

from indw.clean.document.metrics import ChunkMetrics
from indw.clean.corpus import CleaningResult
from indw.clean.document.stats import CleaningStats, StageStats


def merge_cleaning_stats(target: CleaningStats, source: CleaningStats) -> None:
    target.input_documents += source.input_documents
    target.output_chunks += source.output_chunks
    target.dropped_chunks += source.dropped_chunks
    for name in (
        'html', 'ui_noise', 'metadata', 'boilerplate', 'source_processing',
        'structure', 'compression', 'conversation', 'deduplication',
        'segmentation', 'quality_filter', 'document_gate', 'artifacts',
    ):
        dst: StageStats = getattr(target, name)
        src: StageStats = getattr(source, name)
        dst.in_docs += src.in_docs
        dst.out_docs += src.out_docs
        dst.dropped += src.dropped
        dst.lines_removed += src.lines_removed
        dst.chars_removed += src.chars_removed
    for reason, count in source.document_gate_reasons.items():
        target.document_gate_reasons[reason] = (
            target.document_gate_reasons.get(reason, 0) + count
        )


def cleaning_result_from_dict(row: dict[str, Any]) -> CleaningResult:
    metrics_raw = row.get('metrics') or {}
    metrics = ChunkMetrics(**metrics_raw) if isinstance(metrics_raw, dict) else metrics_raw
    result = CleaningResult(
        text=str(row.get('text') or ''),
        metrics=metrics,
        dropped=bool(row.get('dropped', False)),
        drop_reason=str(row.get('drop_reason') or ''),
        document_id=str(row.get('document_id') or ''),
        chunk_id=str(row.get('chunk_id') or ''),
        source=str(row.get('source') or ''),
        chunk_index=int(row.get('chunk_index', 0)),
    )
    scan = row.get('analysis_scan')
    if isinstance(scan, str) and scan:
        result.analysis_scan = scan
    full_len = row.get('analysis_full_len')
    if full_len is not None:
        result.analysis_full_len = int(full_len)
    bundle = row.get('analysis_bundle')
    if bundle is not None:
        result.analysis_bundle = bundle
    return result


def preprocessed_line_to_objects(line: dict[str, Any]) -> dict[str, Any]:
    chunks = []
    for chunk in line.get('chunks') or []:
        clean_result = chunk['clean_result']
        if isinstance(clean_result, dict):
            clean_result = cleaning_result_from_dict(clean_result)
        chunks.append({
            'chunk_text': chunk['chunk_text'],
            'content_hash': chunk['content_hash'],
            'clean_result': clean_result,
            'doc': chunk['doc'],
        })
    cleaning_stats = line.get('cleaning_stats')
    if isinstance(cleaning_stats, dict):
        cleaning_stats = CleaningStats(**{
            k: v for k, v in cleaning_stats.items()
            if k in CleaningStats.__dataclass_fields__
        })
    return {
        **line,
        'chunks': chunks,
        'cleaning_stats': cleaning_stats,
    }
