from __future__ import annotations

from indw.config.validation import ConfigResolutionError
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CorpusDocument
from indw.filter.score.engine import ScoreEngine

def score_document(
    doc: CorpusDocument,
    policy: PipelinePolicy,
    *,
    engine: ScoreEngine | None = None,
) -> CorpusDocument:
    if engine is None:
        raise ConfigResolutionError('score_document requires ScoreEngine from PipelineRunner')
    scorer = engine
    if not doc.text:
        return doc.with_scores(scorer.score(doc)).with_stage('quality_scoring')
    scores = scorer.score(doc)
    return doc.with_scores(scores).with_stage('quality_scoring')
