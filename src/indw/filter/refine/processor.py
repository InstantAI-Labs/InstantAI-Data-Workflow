from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.config.resolve import PipelineConfigContext
from indw.filter.spec.document import CorpusDocument
from indw.schedule import PipelineRunner
from indw.filter.spec.document import EXPORT_ACTIONS, PipelineAction
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CuratorDecision
from indw.filter.score.types import CanonicalScores
from indw.filter.content.metadata import build_training_meta

@dataclass
class ProcessedDocument:
    text: str
    keep: bool
    reason: str
    action: PipelineAction
    scores: CanonicalScores
    doc: CorpusDocument
    decision: CuratorDecision
    artifact_signals: dict[str, float] | None = None

    def to_meta(self) -> dict[str, Any]:
        return build_training_meta(
            route_dict=self.decision.to_training_dict(),
            scores_dict=self.scores.to_dict(),
            score_composite=self.scores.composite,
            sample_weight=self.decision.sample_weight,
            corpus_partition=self.decision.corpus_partition,
        )

class DocumentProcessor:
    def __init__(
        self,
        ctx: PipelineConfigContext | None = None,
        *,
        quality_spec: str | None = None,
    ) -> None:
        self.ctx = ctx or PipelineConfigContext.resolve(quality_spec=quality_spec)
        self.runner = PipelineRunner(self.ctx)

    @property
    def policy(self) -> PipelinePolicy:
        return self.runner.policy

    def process(
        self,
        text: str,
        *,
        source: str = '',
        url: str = '',
        domain: str = '',
        row: dict[str, Any] | None = None,
    ) -> ProcessedDocument:
        del row
        doc = self.runner.run_text(text, source=source, url=url, domain=domain)
        return self._from_doc(doc)

    def process_row(
        self,
        row: dict[str, Any],
        *,
        text_key: str | None = None,
    ) -> tuple[dict[str, Any] | None, ProcessedDocument]:
        out, doc = self.runner.run_row(row, text_key=text_key)
        processed = self._from_doc(doc)
        if out is None:
            return None, processed
        return out, processed

    def _from_doc(self, doc: CorpusDocument) -> ProcessedDocument:
        base = doc.decision or CuratorDecision()
        action = base.action
        scores = doc.scores if doc.scores is not None else CanonicalScores()
        components = scores.components if scores.components else None
        weight = (
            self.policy.curator.rewrite_sample_weight
            if action == 'REWRITE'
            else 1.0
        )
        decision = CuratorDecision(
            action=action,
            reason=base.reason,
            detail=base.detail,
            corpus_partition=base.corpus_partition,
            sample_weight=weight,
            route_scores={'composite': scores.composite},
        )
        keep = action in EXPORT_ACTIONS
        return ProcessedDocument(
            text=doc.text,
            keep=keep,
            reason=base.reason,
            action=action,
            scores=scores,
            doc=doc,
            decision=decision,
            artifact_signals=components,
        )
