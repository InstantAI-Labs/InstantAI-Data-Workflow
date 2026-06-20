from __future__ import annotations

from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CorpusDocument
from indw.filter.refine.rewrite import RewriteEngine

def rewrite_document(
    doc: CorpusDocument,
    policy: PipelinePolicy,
    *,
    engine: RewriteEngine | None = None,
) -> CorpusDocument:
    if doc.decision is None or doc.decision.action != 'REWRITE':
        return doc.with_stage('rewrite')
    rewriter = engine or RewriteEngine(policy)
    updated = rewriter.apply(doc)
    return updated.with_stage('rewrite')
