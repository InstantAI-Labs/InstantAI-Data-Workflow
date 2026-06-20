from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from typing import TYPE_CHECKING

from indw.filter.spec.pipeline import PipelinePolicy

if TYPE_CHECKING:
    from indw.config.resolve import PipelineConfigContext
from indw.filter.decide.engine import DecisionEngine
from indw.filter.spec.document import CorpusDocument
from indw.filter.spec.export import export_row
from indw.filter.refine.rewrite import RewriteEngine
from indw.filter.score.engine import ScoreEngine
from indw.schedule.stages import (
    classify_document,
    clean_artifacts,
    decide_document,
    extract_knowledge,
    normalize_document,
    repair_structure,
    rewrite_document,
    score_document,
    validate_document,
)
from indw.filter.spec.validate import ValidationEngine

@dataclass
class PipelineStats:
    processed: int = 0
    kept: int = 0
    rewritten: int = 0
    dropped: int = 0
    invalid: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)

    def observe(self, doc: CorpusDocument) -> None:
        self.processed += 1
        action = doc.decision.action if doc.decision else 'DROP'
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        if action == 'KEEP':
            self.kept += 1
        elif action == 'REWRITE':
            self.rewritten += 1
        else:
            self.dropped += 1
        if doc.validation is not None and not doc.validation.valid:
            self.invalid += 1

class PipelineRunner:
    def __init__(self, ctx: PipelineConfigContext | None = None) -> None:
        from indw.config.resolve import PipelineConfigContext as _Ctx

        self.ctx = ctx or _Ctx.resolve()
        self.policy = self.ctx.pipeline
        self.scorer = ScoreEngine(self.ctx)
        self.decider = DecisionEngine(self.ctx)
        self.rewriter = RewriteEngine(self.policy)
        self.validator = ValidationEngine(self.policy)

    def run(self, doc: CorpusDocument) -> CorpusDocument:
        doc = normalize_document(doc, self.policy)
        doc = clean_artifacts(doc, self.policy)
        doc = repair_structure(doc, self.policy)
        if not doc.text:
            doc = score_document(doc, self.policy, engine=self.scorer)
            doc = decide_document(doc, self.policy, engine=self.decider)
            return validate_document(doc, self.policy, engine=self.validator)

        doc = extract_knowledge(doc, self.policy)
        doc = classify_document(doc, self.policy)
        doc = score_document(doc, self.policy, engine=self.scorer)
        doc = decide_document(doc, self.policy, engine=self.decider)

        if doc.decision is not None and doc.decision.action == 'REWRITE':
            doc = rewrite_document(doc, self.policy, engine=self.rewriter)
            doc = classify_document(doc, self.policy)
            doc = score_document(doc, self.policy, engine=self.scorer)
            doc = decide_document(doc, self.policy, engine=self.decider)

        return validate_document(doc, self.policy, engine=self.validator)

    def run_text(self, text: str, **provenance: Any) -> CorpusDocument:
        doc = CorpusDocument.from_text(text, **provenance)
        return self.run(doc)

    def run_row(self, row: dict[str, Any], *, text_key: str | None = None) -> tuple[dict[str, Any] | None, CorpusDocument]:
        keys = self.policy.export.get('text_keys') or ['text', 'content', 'body', 'markdown']
        key = text_key
        if key is None:
            for candidate in keys:
                if candidate in row:
                    key = candidate
                    break
        if key is None:
            doc = CorpusDocument.from_text('', row_meta=row)
            doc = self.run(doc)
            return None, doc
        doc = CorpusDocument.from_row(row, text_key=key)
        doc = self.run(doc)
        if not doc.exportable or doc.decision is None or doc.decision.action == 'DROP':
            return None, doc
        return export_row(row, doc, text_key=key, policy=self.policy), doc

    def run_jsonl(
        self,
        rows: Iterator[dict[str, Any]],
        *,
        text_key: str | None = None,
    ) -> tuple[Iterator[tuple[dict[str, Any] | None, CorpusDocument]], PipelineStats]:
        stats = PipelineStats()

        def _gen() -> Iterator[tuple[dict[str, Any] | None, CorpusDocument]]:
            for row in rows:
                out, doc = self.run_row(row, text_key=text_key)
                stats.observe(doc)
                yield out, doc

        return _gen(), stats
