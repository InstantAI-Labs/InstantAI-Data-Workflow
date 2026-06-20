from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from indw.clean.artifact.strip import clean_document_artifacts
from indw.clean.meta.clean import clean_pretraining_metadata
from indw.clean.meta.foundation import is_metadata_only_document
from indw.clean.document.boilerplate import remove_boilerplate
from indw.clean.gate.evaluate import evaluate_document_gate
from indw.clean.document.normalize import meaningful_char_count, normalize_text
from indw.clean.document.compression import compress_content
from indw.clean.document.config import CleaningConfig
from indw.clean.document.conversation import extract_conversation
from indw.clean.document.dedup import dedupe_paragraphs
from indw.clean.document.html import clean_html
from indw.clean.document.adaptive import adaptive_chunk_keep
from indw.clean.document.metrics import ChunkMetrics, compute_metrics
from indw.clean.document.value import analyze_content_value
from indw.clean.document.segment import segment_text
from indw.clean.structure import apply_structural_processors
from indw.clean.document.stats import CleaningStats
from indw.clean.structure.extract import extract_structure
from indw.clean.document.ui import remove_low_value_lines, remove_metadata, remove_ui_noise
from indw.clean.document.code_preservation import preserve_code_blocks
from indw.clean.artifact.engine import get_artifact_engine
from indw.clean.document.stage_manifest import log_cleaning_manifest
from indw.clean.artifact.discovery_config import DiscoveryConfig
from indw.clean.artifact.discovery_engine import ArtifactDiscoveryEngine, get_discovery_engine
from indw.clean.semantic.config import SemanticCleaningConfig
from indw.clean.semantic.pipeline import SemanticCleaningPipeline
from indw.filter.refine.truncation import analyze_truncation, repair_truncation


@dataclass
class CleaningResult:
    text: str = ''
    messages: Optional[list[dict[str, str]]] = None
    metrics: ChunkMetrics = field(default_factory=ChunkMetrics)
    dropped: bool = False
    drop_reason: str = ''
    document_id: str = ''
    chunk_id: str = ''
    source: str = ''
    chunk_index: int = 0
    section_class: str = ''
    knowledge_metrics: Optional[dict[str, Any]] = None
    analysis_scan: str = ''
    analysis_full_len: int = 0
    analysis_bundle: Any = None


class CorpusCleaningPipeline:
    def __init__(
        self,
        config: Optional[CleaningConfig] = None,
        *,
        score_thresholds: Any = None,
    ):
        self.config = config or CleaningConfig()
        self._score_thresholds = score_thresholds
        self.stats = CleaningStats()
        self._discovery: ArtifactDiscoveryEngine | None = None
        self._semantic_pipeline: SemanticCleaningPipeline | None = None
        if self.config.artifact_discovery:
            dcfg = DiscoveryConfig.from_cleaning(self.config)
            self._discovery = get_discovery_engine(dcfg, corpus_dir=dcfg.corpus_dir)

    def _semantic_pipeline_for(self) -> SemanticCleaningPipeline:
        if self._semantic_pipeline is None:
            cfg = self.config
            sem_cfg = SemanticCleaningConfig.from_dict({
                'enabled': True,
                'legacy_regex_fallback': False,
                'embedded': cfg.semantic_embedded or None,
            })
            self._semantic_pipeline = SemanticCleaningPipeline(sem_cfg)
        return self._semantic_pipeline

    @property
    def discovery_engine(self) -> ArtifactDiscoveryEngine | None:
        return self._discovery

    def end_discovery_batch(self) -> dict[str, Any] | None:
        if self._discovery is None:
            return None
        report = self._discovery.end_batch()
        return report.to_dict()

    def _chunk_content_value(
        self,
        chunk: str,
        *,
        source: str,
        duplicate_ratio: float,
    ) -> tuple[Any, str, int, Any | None]:
        from indw.filter.score.analysis import _sample_text
        from indw.clean.document.value import analyze_content_value, resolve_analysis_bundle

        th = self._score_thresholds
        if th is not None:
            scan, full_len = _sample_text(
                chunk,
                min_chars=th.min_chars,
                sample_limit=max(th.min_chars, th.score_sample_chars),
            )
            bundle = resolve_analysis_bundle(scan)
            cv = analyze_content_value(
                chunk, source=source, duplicate_ratio=duplicate_ratio, bundle=bundle,
            )
            return cv, scan, full_len, bundle
        bundle = resolve_analysis_bundle(chunk)
        cv = analyze_content_value(
            chunk, source=source, duplicate_ratio=duplicate_ratio, bundle=bundle,
        )
        return cv, '', 0, bundle

    @staticmethod
    def _document_id(source: str, row: Optional[dict[str, Any]], text: str) -> str:
        if row and row.get('id'):
            return f'{source}:{row["id"]}'
        digest = hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()[:16]
        return f'{source}:{digest}'

    def _extract_knowledge_units(
        self,
        text: str,
        *,
        source: str,
        row: Optional[dict[str, Any]],
        doc_id: str,
        pre_normalized: bool = False,
    ) -> list[CleaningResult]:
        from indw.extract.nav.context import NavigationContext
        from indw.extract.structure.aggregate import AggregationContext
        from indw.extract.core.units import extract_knowledge
        from indw.extract.sections.classify import PRIMARY_CLASSES

        cfg = self.config
        working = text.strip()
        if not cfg.minimal and not pre_normalized:
            working = normalize_text(working, preserve_code_fences=cfg.preserve_code_fences)

        if cfg.code_preservation:
            working, _ = preserve_code_blocks(working)

        acc = self._discovery.accumulator if self._discovery is not None else None
        nav_ctx = NavigationContext(accumulator=acc)
        agg_ctx = AggregationContext(accumulator=acc)
        with self.stats.structure.timed():
            extraction = extract_knowledge(
                working, cfg=cfg, row=row, source=source, nav_ctx=nav_ctx, agg_ctx=agg_ctx,
                ke_stats=self.stats.knowledge_extraction,
            )
        km_dict = extraction.metrics.to_dict()

        if extraction.dropped_all or not extraction.units:
            self.stats.dropped_chunks += 1
            return [
                CleaningResult(
                    dropped=True,
                    drop_reason=extraction.drop_reason or 'no_knowledge',
                    document_id=doc_id,
                    source=source,
                    knowledge_metrics=km_dict,
                )
            ]

        results: list[CleaningResult] = []
        kept_idx = 0
        for unit in extraction.units:
            chunk = unit.text
            if not chunk:
                continue
            duplicate_ratio = 0.0
            if cfg.dedupe_paragraphs:
                chunk, duplicate_ratio = dedupe_paragraphs(chunk, stats=self.stats.deduplication)
                if not chunk:
                    continue
                if duplicate_ratio > 0.0:
                    from indw.extract.sections.integrity import (
                        finalize_semantic_unit,
                        score_chunk_integrity,
                    )
                    from indw.extract.sections.semantic import analyze_completion_cached
                    pre_score = score_chunk_integrity(chunk)
                    pre_inc = analyze_completion_cached(chunk).incomplete_probability
                    refinalized, integrity = finalize_semantic_unit(
                        chunk, min_chars=max(40, cfg.min_chars_after_clean // 2),
                    )
                    post_score = score_chunk_integrity(refinalized) if refinalized else 0.0
                    if (
                        refinalized
                        and not integrity.rejected
                        and len(refinalized) + 20 < len(chunk)
                        and pre_score >= 0.52
                    ):
                        refinalized = chunk
                    elif (integrity.rejected or not refinalized) and pre_score >= 0.55 and pre_inc < 0.62:
                        refinalized = chunk
                    elif integrity.rejected or not refinalized:
                        continue
                    chunk = refinalized
            cv, analysis_scan, analysis_full_len, analysis_bundle = self._chunk_content_value(
                chunk, source=source, duplicate_ratio=duplicate_ratio,
            )
            metrics = compute_metrics(
                chunk,
                cfg,
                duplicate_ratio=duplicate_ratio,
                source=source,
                content_value=cv,
                analysis_bundle=analysis_bundle,
            )
            code_heavy = (
                metrics.code_ratio >= max(cfg.min_code_ratio_keep, 0.08)
                or unit.source_kind == 'code'
            )
            meaningful = metrics.meaningful_chars
            primary = unit.section_class in {c.value for c in PRIMARY_CLASSES}
            ke_recovered = cfg.knowledge_extraction and primary and unit.retention_score >= 0.10
            min_keep = cfg.min_chars_after_clean
            if code_heavy:
                min_keep = max(40, cfg.min_chars_after_clean // 4)
            elif primary and unit.retention_score >= 0.12:
                min_keep = max(40, cfg.min_chars_after_clean // 2)
            elif ke_recovered:
                min_keep = max(40, cfg.min_chars_after_clean // 3)
            if meaningful < min_keep and not code_heavy:
                continue
            keep, drop_reason = adaptive_chunk_keep(chunk, metrics, cfg, cv=cv, code_heavy=code_heavy)
            if not keep and unit.retention_score >= 0.12 and primary:
                keep = True
            if not keep and unit.retention_score < 0.12:
                self.stats.quality_filter.dropped += 1
                results.append(CleaningResult(dropped=True, drop_reason=drop_reason, metrics=metrics))
                continue
            if not keep:
                continue
            chunk_id = f'{doc_id}#{kept_idx}'
            results.append(
                CleaningResult(
                    text=chunk,
                    metrics=metrics,
                    document_id=doc_id,
                    chunk_id=chunk_id,
                    source=source,
                    chunk_index=kept_idx,
                    section_class=unit.section_class,
                    knowledge_metrics=km_dict if kept_idx == 0 else None,
                    analysis_scan=analysis_scan,
                    analysis_full_len=analysis_full_len,
                    analysis_bundle=analysis_bundle,
                )
            )
            kept_idx += 1
            self.stats.output_chunks += 1
            self.stats.quality_filter.out_docs += 1

        if not results:
            self.stats.dropped_chunks += 1
            return [
                CleaningResult(
                    dropped=True,
                    drop_reason='no_knowledge_after_filter',
                    document_id=doc_id,
                    source=source,
                    knowledge_metrics=km_dict,
                )
            ]
        return results

    def process(
        self,
        text: str,
        *,
        source: str = '',
        row: Optional[dict[str, Any]] = None,
        document_id: str = '',
        pre_normalized: bool = False,
    ) -> list[CleaningResult]:
        cfg = self.config
        if not cfg.enabled or not text or not text.strip():
            if not text or not text.strip():
                return []
            stripped = text.strip()
            result = CleaningResult(text=stripped, source=source)
            if self._score_thresholds is not None:
                from indw.filter.score.analysis import _sample_text
                from indw.clean.document.value import build_analysis_bundle

                th = self._score_thresholds
                scan, full_len = _sample_text(
                    stripped,
                    min_chars=th.min_chars,
                    sample_limit=max(th.min_chars, th.score_sample_chars),
                )
                result.analysis_scan = scan
                result.analysis_full_len = full_len
                result.analysis_bundle = build_analysis_bundle(scan)
            return [result]

        self.stats.input_documents += 1
        working = text.strip()
        doc_id = document_id or self._document_id(source, row, working)

        if cfg.knowledge_extraction and not cfg.minimal:
            if cfg.document_gate:
                from indw.schedule.monitor.doc import set_doc_stage
                from indw.extract.core.context import get_document_context
                set_doc_stage('s2_fast_filter')
                doc_ctx = get_document_context()
                gate_raw = doc_ctx.gate_raw if doc_ctx is not None else None
                gate_eval = evaluate_document_gate(working, raw=gate_raw)
                self.stats.document_gate.in_docs += 1
                if not gate_eval.keep:
                    self.stats.document_gate.dropped += 1
                    self.stats.dropped_chunks += 1
                    reason = gate_eval.reason or 'document_gate'
                    self.stats.document_gate_reasons[reason] = (
                        self.stats.document_gate_reasons.get(reason, 0) + 1
                    )
                    return [
                        CleaningResult(
                            dropped=True,
                            drop_reason=reason,
                            document_id=doc_id,
                            source=source,
                        )
                    ]
                self.stats.document_gate.out_docs += 1
            from indw.schedule.monitor.doc import set_doc_stage
            set_doc_stage('s4_high_quality')
            return self._extract_knowledge_units(
                working, source=source, row=row, doc_id=doc_id, pre_normalized=pre_normalized,
            )

        if not cfg.minimal and not pre_normalized:
            working = normalize_text(working, preserve_code_fences=cfg.preserve_code_fences)

        if cfg.document_gate:
            from indw.schedule.monitor.doc import set_doc_stage
            from indw.extract.core.context import get_document_context
            set_doc_stage('s2_fast_filter')
            doc_ctx = get_document_context()
            gate_raw = doc_ctx.gate_raw if doc_ctx is not None else None
            gate = evaluate_document_gate(working, raw=gate_raw)
            self.stats.document_gate.in_docs += 1
            if not gate.keep:
                self.stats.document_gate.dropped += 1
                self.stats.dropped_chunks += 1
                reason = gate.reason or 'document_gate'
                self.stats.document_gate_reasons[reason] = (
                    self.stats.document_gate_reasons.get(reason, 0) + 1
                )
                return [
                    CleaningResult(
                        dropped=True,
                        drop_reason=reason,
                        document_id=doc_id,
                        source=source,
                    )
                ]
            self.stats.document_gate.out_docs += 1

        if cfg.html_cleaning:
            from indw.schedule.monitor.doc import set_doc_stage
            set_doc_stage('s1_fast_preprocess')
            working = clean_html(working, stats=self.stats.html)

        if cfg.truncation_repair:
            trunc = analyze_truncation(working)
            if trunc.severity != 'none':
                repaired, trunc_result = repair_truncation(working)
                if trunc_result.severity == 'heavy' and not trunc_result.trimmed:
                    self.stats.truncation_repair.dropped += 1
                    self.stats.dropped_chunks += 1
                    return [
                        CleaningResult(
                            dropped=True,
                            drop_reason='heavily_truncated',
                            document_id=doc_id,
                            source=source,
                        )
                    ]
                removed = len(working) - len(repaired)
                if removed > 0:
                    self.stats.truncation_repair.chars_removed += removed
                working = repaired
            self.stats.truncation_repair.in_docs += 1
            self.stats.truncation_repair.out_docs += 1

        if cfg.code_preservation:
            before_cp = len(working)
            working, _cp_stats = preserve_code_blocks(working)
            self.stats.code_preservation.in_docs += 1
            self.stats.code_preservation.out_docs += 1 if working.strip() else 0
            self.stats.code_preservation.chars_removed += max(0, before_cp - len(working))

        if cfg.artifact_discovery and self._discovery is not None:
            from indw.clean.gate.evaluate import compute_artifact_ratio

            legacy_ratio, _ = compute_artifact_ratio(working, include_discovery=False)
            report = self._discovery.discover(
                working, doc_id=doc_id, legacy_ratio=legacy_ratio,
            )
            self.stats.discovery.in_docs += 1
            self.stats.discovery.out_docs += 1
            if abs(report.discovery_ratio - legacy_ratio) > 0.15:
                self.stats.discovery_shadow_disagreements += 1
            if cfg.artifact_discovery_trim and not cfg.artifact_discovery_shadow:
                trimmed = self._discovery.apply_trim(working, report)
                self.stats.discovery.chars_removed += len(working) - len(trimmed)
                if report.trim:
                    self.stats.discovery.lines_removed += report.trim.units_removed
                working = trimmed
            if len(self.stats.discovery_reports) < 500:
                self.stats.discovery_reports.append(report.to_dict())

        if cfg.artifact_cleaning:
            working, _artifact_stats = clean_document_artifacts(
                working,
                preserve_code_fences=cfg.preserve_code_fences,
                html_already_extracted=cfg.html_cleaning,
            )
            self.stats.artifacts.in_docs += 1
            self.stats.artifacts.out_docs += 1 if working.strip() else 0
            if not working.strip():
                self.stats.artifacts.dropped += 1
                self.stats.dropped_chunks += 1
                return [
                    CleaningResult(
                        dropped=True,
                        drop_reason='empty_after_clean',
                        document_id=doc_id,
                        source=source,
                    )
                ]

        if cfg.pretraining_metadata_cleaning:
            working, _meta_stats = clean_pretraining_metadata(
                working,
                preserve_code_fences=cfg.preserve_code_fences,
                strip_code_comments=cfg.strip_code_license_headers,
                strip_license_blocks=cfg.strip_license_blocks,
                strip_copyright_lines=cfg.strip_copyright_lines,
                license_remove_confidence=cfg.license_remove_confidence,
                license_review_confidence=cfg.license_review_confidence,
                license_validate_syntax=cfg.license_validate_syntax,
            )
            if _meta_stats.license_regions_flagged:
                self.stats.metadata.lines_removed += _meta_stats.license_regions_flagged
            if not working.strip():
                self.stats.artifacts.dropped += 1
                self.stats.dropped_chunks += 1
                return [
                    CleaningResult(
                        dropped=True,
                        drop_reason='empty_after_metadata_clean',
                        document_id=doc_id,
                        source=source,
                    )
                ]
            if is_metadata_only_document(working):
                self.stats.artifacts.dropped += 1
                self.stats.dropped_chunks += 1
                return [
                    CleaningResult(
                        dropped=True,
                        drop_reason='metadata_only',
                        document_id=doc_id,
                        source=source,
                    )
                ]

        use_semantic = cfg.semantic_cleaning and not cfg.minimal
        use_legacy_structural = cfg.legacy_regex_cleaning and not cfg.minimal

        if use_semantic:
            from indw.clean.gate.evaluate import compute_artifact_ratio

            legacy_ratio, _ = compute_artifact_ratio(working, include_discovery=False)
            if legacy_ratio < 0.08:
                use_semantic = False

        if not cfg.minimal:
            from indw.schedule.monitor.doc import set_doc_stage
            set_doc_stage('s3_intermediate')
            if cfg.inline_artifact_removal:
                engine = get_artifact_engine()
                if cfg.document_understanding and self._discovery is not None:
                    engine.bind_discovery(
                        self._discovery,
                        corpus_dir=cfg.artifact_discovery_corpus_dir,
                    )
                working, inline_stats = engine.strip_inline(
                    working,
                    preserve_code_fences=cfg.preserve_code_fences,
                    doc_id=doc_id,
                )
                if inline_stats.spans_removed:
                    self.stats.ui_noise.in_docs += 1
                    self.stats.ui_noise.lines_removed += inline_stats.spans_removed
            if cfg.ui_noise_removal:
                working = remove_ui_noise(working, stats=self.stats.ui_noise)
            if cfg.metadata_removal:
                working = remove_metadata(working, stats=self.stats.metadata)
            working = remove_boilerplate(working, stats=self.stats.boilerplate)
            if use_legacy_structural:
                working = apply_structural_processors(working, cfg=cfg, stats=self.stats.source_processing)

        if use_semantic:
            from indw.schedule.monitor.doc import set_doc_stage
            set_doc_stage('s4_high_quality')
            with self.stats.semantic_cleaning.timed():
                sem = self._semantic_pipeline_for().process(working)
            working = sem.text
            self.stats.semantic_cleaning.in_docs += 1
            self.stats.semantic_cleaning.out_docs += 1 if working.strip() else 0
            self.stats.semantic_cleaning.chunks_removed += sem.removed_chunks
            self.stats.semantic_cleaning.chunks_downweighted += sem.downweighted_chunks
            self.stats.semantic_cleaning.lines_removed += sem.cleaned_chunks

        if not cfg.minimal:
            if not use_semantic and not use_legacy_structural:
                working = apply_structural_processors(working, cfg=cfg, stats=self.stats.source_processing)
            working = extract_structure(
                working,
                source='' if use_semantic else source,
                stats=self.stats.structure,
                generic_only=use_semantic,
            )
            if cfg.content_compression:
                working = compress_content(working, stats=self.stats.compression)
            working = remove_low_value_lines(
                working,
                drop_ack=cfg.drop_acknowledgements,
                drop_mod=cfg.drop_moderator_notices,
                drop_quotes=cfg.drop_quoted_replies,
                stats=self.stats.conversation,
            )
        working = normalize_text(working, preserve_code_fences=cfg.preserve_code_fences)

        duplicate_ratio = 0.0
        if cfg.dedupe_paragraphs:
            working, duplicate_ratio = dedupe_paragraphs(working, stats=self.stats.deduplication)

        units: list[str] = []
        messages_out: list[list[dict[str, str]]] = []
        if cfg.extract_qa:
            pair = extract_conversation(
                working,
                row=row,
                max_extra_answers=cfg.max_extra_answers,
                stats=self.stats.conversation,
            )
            if pair is not None:
                if cfg.output_messages_format:
                    messages_out.append(pair.to_messages())
                units.append(pair.to_text())
            else:
                units.append(working)
        else:
            units.append(working)

        chunks: list[str] = []
        for unit in units:
            if not unit:
                continue
            for chunk in segment_text(unit, cfg, stats=self.stats.segmentation):
                if chunk:
                    chunks.append(chunk)

        results: list[CleaningResult] = []
        kept_idx = 0
        for chunk in chunks:
            if not chunk:
                self.stats.quality_filter.dropped += 1
                self.stats.dropped_chunks += 1
                continue
            cv, analysis_scan, analysis_full_len, analysis_bundle = self._chunk_content_value(
                chunk, source=source, duplicate_ratio=duplicate_ratio,
            )
            metrics = compute_metrics(
                chunk,
                cfg,
                duplicate_ratio=duplicate_ratio,
                source=source,
                content_value=cv,
                analysis_bundle=analysis_bundle,
            )
            code_heavy = metrics.code_ratio >= max(cfg.min_code_ratio_keep, 0.08)
            meaningful = metrics.meaningful_chars
            if meaningful < cfg.min_chars_after_clean and not code_heavy:
                self.stats.quality_filter.dropped += 1
                self.stats.dropped_chunks += 1
                continue
            keep, drop_reason = adaptive_chunk_keep(
                chunk, metrics, cfg, cv=cv, code_heavy=code_heavy,
            )
            if not keep:
                self.stats.quality_filter.dropped += 1
                self.stats.dropped_chunks += 1
                results.append(CleaningResult(dropped=True, drop_reason=drop_reason, metrics=metrics))
                continue
            msg = messages_out[0] if messages_out else None
            chunk_id = f'{doc_id}#{kept_idx}'
            results.append(
                CleaningResult(
                    text=chunk,
                    messages=msg,
                    metrics=metrics,
                    document_id=doc_id,
                    chunk_id=chunk_id,
                    source=source,
                    chunk_index=kept_idx,
                    analysis_scan=analysis_scan,
                    analysis_full_len=analysis_full_len,
                    analysis_bundle=analysis_bundle,
                )
            )
            kept_idx += 1
            self.stats.output_chunks += 1
            self.stats.quality_filter.out_docs += 1

        if not results:
            self.stats.dropped_chunks += 1
        return results

    def snapshot(self) -> dict[str, Any]:
        return self.stats.to_dict()


_TEXT_KEYS = ('text', 'content', 'body', 'markdown')


def row_text_key(row: dict[str, Any]) -> str | None:
    for key in _TEXT_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return key
    return None


def extract_row_text(row: dict[str, Any]) -> str:
    for key in _TEXT_KEYS:
        val = row.get(key, '')
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ''


def process_jsonl_row(
    row: dict[str, Any],
    pipeline: CorpusCleaningPipeline,
    *,
    source: str = '',
) -> dict[str, Any] | None:
    key = row_text_key(row)
    if key is None:
        return None
    text = row[key].strip()
    results = pipeline.process(text, source=source, row=row)
    kept = [r for r in results if not r.dropped and r.text]
    if not kept:
        return None
    out = dict(row)
    out[key] = kept[0].text
    return out


def final_pass_jsonl_row(
    row: dict[str, Any],
    *,
    min_chars_after_clean: int | None = None,
    preserve_code_fences: bool = True,
    metadata_cleaning: bool = True,
    strip_code_comments: bool = True,
    document_gate: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    key = row_text_key(row)
    if key is None:
        return None, 'empty_text'
    text = row[key].strip()
    cleaned, _ = clean_document_artifacts(text, preserve_code_fences=preserve_code_fences)
    if not cleaned.strip():
        return None, 'empty_after_clean'
    if metadata_cleaning:
        cleaned, _ = clean_pretraining_metadata(
            cleaned,
            preserve_code_fences=preserve_code_fences,
            strip_code_comments=strip_code_comments,
        )
        if not cleaned.strip():
            return None, 'empty_after_metadata_clean'
        if is_metadata_only_document(cleaned):
            return None, 'metadata_only'
    if document_gate:
        gate = evaluate_document_gate(cleaned)
        if not gate.keep:
            return None, gate.reason or 'document_gate'
    from indw.config.defaults import MIN_CHARS_FINAL
    min_chars = min_chars_after_clean
    if min_chars is None:
        min_chars = MIN_CHARS_FINAL
    if meaningful_char_count(cleaned) < min_chars:
        return None, 'too_short'
    out = dict(row)
    out[key] = cleaned
    return out, ''

