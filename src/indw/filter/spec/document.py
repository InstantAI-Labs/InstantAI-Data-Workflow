from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal

PipelineAction = Literal['KEEP', 'REWRITE', 'DROP']

KEEP_ACTIONS = frozenset({'KEEP'})
EXPORT_ACTIONS = frozenset({'KEEP', 'REWRITE'})

@dataclass(frozen=True)
class Provenance:
    source: str = ''
    url: str = ''
    domain: str = ''
    license: str = 'Unknown'
    row_meta: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ContentClassification:
    category: str = 'unknown'
    document_type: str = 'unknown'
    content_type: str = 'text'
    language: str = 'unknown'
    flags: tuple[str, ...] = ()

@dataclass(frozen=True)
class CuratorDecision:
    action: PipelineAction = 'DROP'
    reason: str = ''
    detail: str = ''
    corpus_partition: str = 'main'
    sample_weight: float = 1.0
    route_scores: dict[str, float] = field(default_factory=dict)

    @property
    def keep(self) -> bool:
        return self.action in EXPORT_ACTIONS

    @property
    def route(self) -> str:
        return self.action

    def to_dict(self) -> dict[str, Any]:
        return {
            'action': self.action,
            'route': self.action,
            'reason': self.reason,
            'detail': self.detail,
            'corpus_partition': self.corpus_partition,
            'sample_weight': round(self.sample_weight, 4),
            'route_scores': {k: round(v, 4) for k, v in self.route_scores.items()},
        }

    def to_training_dict(self) -> dict[str, str]:
        return {
            'action': self.action,
            'route': self.action,
            'reason': self.reason,
            'corpus_partition': self.corpus_partition,
        }

@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = 'error'

@dataclass(frozen=True)
class ValidationResult:
    valid: bool = True
    issues: tuple[ValidationIssue, ...] = ()

    @staticmethod
    def ok() -> ValidationResult:
        return ValidationResult(valid=True, issues=())

@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    raw_text: str
    text: str
    provenance: Provenance = field(default_factory=Provenance)
    classification: ContentClassification | None = None
    scores: Any | None = None
    decision: CuratorDecision | None = None
    validation: ValidationResult | None = None
    stage_trace: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    text_modified: bool = False
    exportable: bool = False

    @staticmethod
    def from_text(
        text: str,
        *,
        doc_id: str | None = None,
        source: str = '',
        url: str = '',
        domain: str = '',
        license: str = 'Unknown',
        row_meta: dict[str, Any] | None = None,
    ) -> CorpusDocument:
        body = (text or '').strip()
        stable = doc_id or hashlib.sha256(body.encode('utf-8', 'ignore')).hexdigest()[:16]
        if not stable:
            stable = uuid.uuid4().hex[:16]
        prov = Provenance(
            source=source,
            url=url,
            domain=domain,
            license=license,
            row_meta=dict(row_meta or {}),
        )
        return CorpusDocument(
            doc_id=stable,
            raw_text=body,
            text=body,
            provenance=prov,
        )

    @staticmethod
    def from_row(row: dict[str, Any], *, text_key: str) -> CorpusDocument:
        meta = row.get('meta') if isinstance(row.get('meta'), dict) else {}
        source = str(row.get('source') or meta.get('source') or '')
        url = str(row.get('url') or meta.get('url') or '')
        domain = str(row.get('domain') or meta.get('domain') or '')
        license_name = str(row.get('license') or meta.get('license') or 'Unknown')
        doc_id = str(row.get('id') or meta.get('id') or '')
        return CorpusDocument.from_text(
            str(row.get(text_key) or ''),
            doc_id=doc_id or None,
            source=source,
            url=url,
            domain=domain,
            license=license_name,
            row_meta=dict(row),
        )

    def with_text(self, text: str, *, modified: bool = True) -> CorpusDocument:
        return replace(self, text=text.strip(), text_modified=self.text_modified or modified)

    def with_stage(self, stage: str, **updates: Any) -> CorpusDocument:
        trace = self.stage_trace + (stage,)
        return replace(self, stage_trace=trace, **updates)

    def with_classification(self, classification: ContentClassification) -> CorpusDocument:
        return self.with_stage('classification', classification=classification)

    def with_scores(self, scores: Any) -> CorpusDocument:
        return self.with_stage('quality_scoring', scores=scores)

    def with_decision(self, decision: CuratorDecision) -> CorpusDocument:
        return self.with_stage('curator', decision=decision)

    def with_validation(self, validation: ValidationResult) -> CorpusDocument:
        exportable = validation.valid and self.decision is not None and self.decision.action != 'DROP'
        return self.with_stage('validation', validation=validation, exportable=exportable)

    def with_flags(self, flags: tuple[str, ...]) -> CorpusDocument:
        return replace(self, flags=flags)
