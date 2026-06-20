from __future__ import annotations

from dataclasses import dataclass

from indw.config.resolve import PipelineConfigContext
from indw.config.validation import ConfigResolutionError
from indw.filter.score.canonical import score_document_canonical
from indw.filter.score.types import CanonicalDocumentScore
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CorpusDocument

@dataclass
class ScoreContext:
    artifact_ratio: float = 0.0
    artifact_components: dict[str, float] | None = None

class ScoreEngine:
    def __init__(self, ctx: PipelineConfigContext) -> None:
        if ctx is None:
            raise ConfigResolutionError('ScoreEngine requires PipelineConfigContext')
        self.ctx = ctx
        self.policy: PipelinePolicy = ctx.pipeline

    def build_context(self, doc: CorpusDocument) -> ScoreContext:
        if not doc.text:
            return ScoreContext()
        scored = self.score(doc)
        return ScoreContext(
            artifact_ratio=scored.artifact_ratio,
            artifact_components=scored.artifact_components,
        )

    def score(self, doc: CorpusDocument, ctx: ScoreContext | None = None) -> CanonicalDocumentScore:
        del ctx
        if not doc.text:
            return CanonicalDocumentScore()
        return score_document_canonical(
            doc.text,
            source=doc.provenance.source,
            gate_ctx=self.ctx,
        )
