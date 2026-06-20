from __future__ import annotations

from typing import Any

from indw.clean.corpus import CorpusCleaningPipeline
from indw.filter.spec.quality import QualityPipelineConfig
from indw.filter.gate.quality import QualityGate
from indw.schedule.read.gates import (
    attach_analysis_cache,
    early_document_max_gate,
    early_document_size_gate,
    early_language_gate,
    parse_merge_jsonl_line,
    worker_quality_config,
)
from indw.schedule.stages.engine import run_progressive_preprocess

__all__ = [
    'attach_analysis_cache',
    'early_document_max_gate',
    'early_document_size_gate',
    'early_language_gate',
    'parse_merge_jsonl_line',
    'preprocess_merge_line',
    'worker_quality_config',
]


def preprocess_merge_line(
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
) -> dict[str, Any]:
    return run_progressive_preprocess(
        line=line,
        src_name=src_name,
        line_no=line_no,
        seq=seq,
        cleaning_pipeline=cleaning_pipeline,
        gate=gate,
        cfg=cfg,
        provenance=provenance,
        row=row,
        stage_profile=stage_profile,
    )
