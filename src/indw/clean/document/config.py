from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from indw.config.defaults import HARD_MAX_CHARS, MIN_CHARS_AFTER_CLEAN

@dataclass
class CleaningConfig:
    enabled: bool = True
    min_tokens: int = 1000
    max_tokens: int = 3000
    chunk_overlap_ratio: float = 0.12
    min_words: int = 200
    max_words: int = 1500
    hard_max_chars: int = HARD_MAX_CHARS
    min_chars_after_clean: int = MIN_CHARS_AFTER_CLEAN
    max_ui_noise_ratio: float = 0.45
    max_boilerplate_ratio: float = 0.55
    max_duplicate_ratio: float = 0.35
    min_code_ratio_keep: float = 0.0
    chars_per_token_estimate: float = 4.0
    extract_qa: bool = True
    max_extra_answers: int = 1
    split_long_documents: bool = True
    dedupe_paragraphs: bool = True
    drop_acknowledgements: bool = True
    drop_moderator_notices: bool = True
    drop_quoted_replies: bool = True
    preserve_code_fences: bool = True
    output_messages_format: bool = False
    html_cleaning: bool = True
    ui_noise_removal: bool = True
    metadata_removal: bool = True
    content_compression: bool = True
    minimal: bool = False
    document_gate: bool = True
    artifact_cleaning: bool = True
    pretraining_metadata_cleaning: bool = True
    strip_code_license_headers: bool = True
    artifact_discovery: bool = True
    artifact_discovery_shadow: bool = False
    artifact_discovery_trim: bool = True
    artifact_discovery_corpus_dir: str = ''
    artifact_discovery_min_trim_confidence: float = 0.92
    artifact_discovery_promote_doc_freq: int = 8
    artifact_discovery_demote_weight: float = 0.08
    artifact_discovery_decay: float = 0.95
    artifact_discovery_max_trim_ratio: float = 0.40
    artifact_discovery_min_doc_chars: int = 200
    truncation_repair: bool = True
    code_preservation: bool = True
    strip_license_blocks: bool = True
    strip_copyright_lines: bool = True
    preserve_spdx_identifier: bool = False
    license_remove_confidence: float = 0.82
    license_review_confidence: float = 0.55
    license_validate_syntax: bool = True
    semantic_cleaning: bool = False
    legacy_regex_cleaning: bool = True
    inline_artifact_removal: bool = True
    document_understanding: bool = True
    knowledge_extraction: bool = False
    semantic_embedded: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> CleaningConfig:
        if not raw:
            return cls()
        seg = raw.get('segmentation') or {}
        qa = raw.get('qa_extraction') or {}
        stages = raw.get('stages') or {}
        thresholds = raw.get('thresholds') or {}
        min_tok = int(seg.get('min_tokens', raw.get('min_tokens', 1000)))
        max_tok = int(seg.get('max_tokens', raw.get('max_tokens', 3000)))
        return cls(
            enabled=bool(raw.get('enabled', True)),
            min_tokens=min_tok,
            max_tokens=max_tok,
            chunk_overlap_ratio=float(
                seg.get('chunk_overlap_ratio', raw.get('chunk_overlap_ratio', 0.12))
            ),
            min_words=int(seg.get('min_words', raw.get('min_words', max(120, min_tok // 2)))),
            max_words=int(seg.get('max_words', raw.get('max_words', max(400, max_tok // 2)))),
            hard_max_chars=int(seg.get('hard_max_chars', raw.get('hard_max_chars', HARD_MAX_CHARS))),
            min_chars_after_clean=int(
                thresholds.get('min_chars_after_clean', raw.get('min_chars_after_clean', MIN_CHARS_AFTER_CLEAN))
            ),
            max_ui_noise_ratio=float(thresholds.get('max_ui_noise_ratio', raw.get('max_ui_noise_ratio', 0.45))),
            max_boilerplate_ratio=float(thresholds.get('max_boilerplate_ratio', raw.get('max_boilerplate_ratio', 0.55))),
            max_duplicate_ratio=float(thresholds.get('max_duplicate_ratio', raw.get('max_duplicate_ratio', 0.35))),
            min_code_ratio_keep=float(thresholds.get('min_code_ratio_keep', 0.0)),
            chars_per_token_estimate=float(raw.get('chars_per_token_estimate', 4.0)),
            extract_qa=bool(qa.get('enabled', raw.get('extract_qa', True))),
            max_extra_answers=int(qa.get('max_extra_answers', raw.get('max_extra_answers', 1))),
            split_long_documents=bool(seg.get('enabled', raw.get('split_long_documents', True))),
            dedupe_paragraphs=bool(raw.get('dedupe_paragraphs', True)),
            drop_acknowledgements=bool(raw.get('drop_acknowledgements', True)),
            drop_moderator_notices=bool(raw.get('drop_moderator_notices', True)),
            drop_quoted_replies=bool(raw.get('drop_quoted_replies', True)),
            preserve_code_fences=bool(raw.get('preserve_code_fences', True)),
            output_messages_format=bool(raw.get('output_messages_format', False)),
            html_cleaning=bool(stages.get('html_cleaning', True)),
            ui_noise_removal=bool(stages.get('ui_noise_removal', True)),
            metadata_removal=bool(stages.get('metadata_removal', True)),
            content_compression=bool(stages.get('content_compression', raw.get('content_compression', True))),
            minimal=bool(raw.get('minimal', False)),
            document_gate=bool(raw.get('document_gate', True)),
            artifact_cleaning=bool(raw.get('artifact_cleaning', True)),
            pretraining_metadata_cleaning=bool(
                raw.get('pretraining_metadata_cleaning', stages.get('pretraining_metadata_cleaning', True))
            ),
            strip_code_license_headers=bool(raw.get('strip_code_license_headers', True)),
            artifact_discovery=bool(raw.get('artifact_discovery', True)),
            artifact_discovery_shadow=bool(raw.get('artifact_discovery_shadow', False)),
            artifact_discovery_trim=bool(raw.get('artifact_discovery_trim', True)),
            artifact_discovery_corpus_dir=str(raw.get('artifact_discovery_corpus_dir', '') or ''),
            artifact_discovery_min_trim_confidence=float(
                raw.get('artifact_discovery_min_trim_confidence', 0.92)
            ),
            artifact_discovery_promote_doc_freq=int(
                raw.get('artifact_discovery_promote_doc_freq', 8)
            ),
            artifact_discovery_demote_weight=float(
                raw.get('artifact_discovery_demote_weight', 0.08)
            ),
            artifact_discovery_decay=float(raw.get('artifact_discovery_decay', 0.95)),
            artifact_discovery_max_trim_ratio=float(
                raw.get('artifact_discovery_max_trim_ratio', 0.40)
            ),
            artifact_discovery_min_doc_chars=int(
                raw.get('artifact_discovery_min_doc_chars', 200)
            ),
            truncation_repair=bool(raw.get('truncation_repair', True)),
            code_preservation=bool(raw.get('code_preservation', True)),
            strip_license_blocks=bool(raw.get('strip_license_blocks', True)),
            strip_copyright_lines=bool(raw.get('strip_copyright_lines', True)),
            preserve_spdx_identifier=bool(raw.get('preserve_spdx_identifier', False)),
            license_remove_confidence=float(raw.get('license_remove_confidence', 0.82)),
            license_review_confidence=float(raw.get('license_review_confidence', 0.55)),
            license_validate_syntax=bool(raw.get('license_validate_syntax', True)),
            semantic_cleaning=bool(raw.get('semantic_cleaning', False)),
            legacy_regex_cleaning=bool(raw.get('legacy_regex_cleaning', True)),
            inline_artifact_removal=bool(raw.get('inline_artifact_removal', True)),
            document_understanding=bool(raw.get('document_understanding', True)),
            knowledge_extraction=bool(raw.get('knowledge_extraction', False)),
            semantic_embedded=dict(raw.get('semantic_embedded') or {}),
        )
