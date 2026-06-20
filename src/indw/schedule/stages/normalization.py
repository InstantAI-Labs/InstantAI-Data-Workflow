from __future__ import annotations

from indw.clean.document.normalize import normalize_text
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CorpusDocument

def normalize_document(doc: CorpusDocument, policy: PipelinePolicy) -> CorpusDocument:
    if not policy.normalization.enabled or not doc.text:
        return doc.with_stage('normalization')
    text = normalize_text(doc.text)
    return doc.with_text(text, modified=text != doc.text).with_stage('normalization')
