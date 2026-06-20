from __future__ import annotations

from indw.config.validation import ConfigResolutionError
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.decide.engine import DecisionEngine
from indw.filter.spec.document import CorpusDocument

def decide_document(
    doc: CorpusDocument,
    policy: PipelinePolicy,
    *,
    engine: DecisionEngine | None = None,
) -> CorpusDocument:
    if engine is None:
        raise ConfigResolutionError('decide_document requires DecisionEngine from PipelineRunner')
    decider = engine
    if doc.scores is None:
        return doc.with_decision(decider._curator.decide(doc))
    pipeline_decision = decider.decide(doc.scores, doc.text, doc=doc)
    return doc.with_decision(pipeline_decision.to_curator_decision())
