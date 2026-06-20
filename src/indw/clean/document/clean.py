from __future__ import annotations

from typing import Any, Optional

from indw.clean.document.boilerplate import remove_boilerplate
from indw.clean.document.config import CleaningConfig
from indw.clean.meta.foundation import clean_foundation_document, strip_social_promo_prefix
from indw.clean.corpus import CleaningResult, CorpusCleaningPipeline
from indw.clean.document.ui import remove_low_value_lines, remove_metadata, remove_ui_noise
from indw.filter.spec.pipeline import PipelinePolicy

def clean_document_artifact_layer(text: str, policy: PipelinePolicy) -> str:
    cfg = policy.artifact_cleaning
    if not text:
        return text
    working = text
    if cfg.get('strip_ui_noise'):
        working = remove_ui_noise(working)
    if cfg.get('strip_metadata_lines'):
        working = remove_metadata(working)
    if cfg.get('strip_boilerplate'):
        working = remove_boilerplate(working)
        working = remove_low_value_lines(working)
    if cfg.get('strip_social_promo'):
        working, _ = strip_social_promo_prefix(working)
    if cfg.get('foundation_metadata_clean'):
        cleaned, _stats = clean_foundation_document(working)
        if cleaned:
            working = cleaned
    return working

def clean_document(
    text: str,
    *,
    config: Optional[CleaningConfig] = None,
    policy: Optional[PipelinePolicy] = None,
    pipeline: Optional[CorpusCleaningPipeline] = None,
    source: str = '',
    row: Optional[dict[str, Any]] = None,
    document_id: str = '',
) -> list[CleaningResult]:
    if pipeline is not None:
        return pipeline.process(text, source=source, row=row, document_id=document_id)
    if config is not None and config.enabled:
        return CorpusCleaningPipeline(config).process(
            text,
            source=source,
            row=row,
            document_id=document_id,
        )
    if policy is not None:
        cleaned = clean_document_artifact_layer(text, policy)
        return [CleaningResult(text=cleaned, source=source)]
    if text and text.strip():
        return [CleaningResult(text=text.strip(), source=source)]
    return []
