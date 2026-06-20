from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from indw.clean.document.config import CleaningConfig

logger = logging.getLogger(__name__)

@dataclass
class CleaningStageManifest:
    normalize: bool = False
    inline_detection: bool = False
    ui_cleaning: bool = False
    metadata_cleaning: bool = False
    boilerplate_cleaning: bool = False
    legacy_structural: bool = False
    semantic_cleaning: bool = False
    knowledge_extraction: bool = False
    html_cleaning: bool = False
    artifact_cleaning: bool = False
    pretraining_metadata: bool = False
    quality_gate: bool = False
    semantic_dedup: bool = False

    def active_stages(self) -> list[str]:
        labels = (
            ('Normalize', self.normalize),
            ('Inline Detection', self.inline_detection),
            ('UI Cleaning', self.ui_cleaning),
            ('Metadata Cleaning', self.metadata_cleaning),
            ('Boilerplate Cleaning', self.boilerplate_cleaning),
            ('Legacy Structural', self.legacy_structural),
            ('Semantic Cleaning', self.semantic_cleaning),
            ('Knowledge Extraction', self.knowledge_extraction),
            ('HTML Cleaning', self.html_cleaning),
            ('Artifact Cleaning', self.artifact_cleaning),
            ('Pretraining Metadata', self.pretraining_metadata),
            ('Quality Gate', self.quality_gate),
            ('Semantic Dedup', self.semantic_dedup),
        )
        return [name for name, on in labels if on]

    def validate_required(self, *, require_semantic: bool = False) -> None:
        required = ['Normalize', 'Artifact Cleaning']
        if require_semantic and not self.knowledge_extraction:
            required.append('Semantic Cleaning')
        if self.knowledge_extraction:
            required.append('Knowledge Extraction')
        elif not require_semantic:
            required.append('Inline Detection')
        active = set(self.active_stages())
        missing = [r for r in required if r not in active]
        if missing:
            raise RuntimeError(
                f'Cleaning pipeline missing required stages: {missing}; active={sorted(active)}'
            )

def manifest_from_cleaning_config(cfg: CleaningConfig) -> CleaningStageManifest:
    minimal = cfg.minimal or not cfg.enabled
    return CleaningStageManifest(
        normalize=not minimal,
        inline_detection=not minimal and cfg.inline_artifact_removal,
        ui_cleaning=not minimal and cfg.ui_noise_removal,
        metadata_cleaning=not minimal and cfg.metadata_removal,
        boilerplate_cleaning=not minimal,
        legacy_structural=not minimal and cfg.legacy_regex_cleaning,
        semantic_cleaning=not minimal and cfg.semantic_cleaning,
        knowledge_extraction=not minimal and cfg.knowledge_extraction,
        html_cleaning=cfg.html_cleaning and not minimal,
        artifact_cleaning=cfg.artifact_cleaning,
        pretraining_metadata=cfg.pretraining_metadata_cleaning,
    )

def log_cleaning_manifest(cfg: CleaningConfig, *, dedup_semantic: bool = False) -> CleaningStageManifest:
    manifest = manifest_from_cleaning_config(cfg)
    manifest.semantic_dedup = dedup_semantic
    stages = manifest.active_stages()
    logger.info('Cleaning stages: %s', ' | '.join(f'{s} ON' for s in stages))
    if cfg.enabled and not cfg.minimal:
        manifest.validate_required(require_semantic=cfg.semantic_cleaning and not cfg.knowledge_extraction)
    return manifest

def merge_stage_manifest(
    cleaning: CleaningConfig,
    *,
    quality_gate: bool = True,
    dedup_semantic: bool = False,
) -> dict[str, Any]:
    m = manifest_from_cleaning_config(cleaning)
    m.quality_gate = quality_gate
    m.semantic_dedup = dedup_semantic
    return {'stages': m.active_stages(), 'manifest': m}
