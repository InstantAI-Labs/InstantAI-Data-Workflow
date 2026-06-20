from __future__ import annotations

from indw.clean.document.clean import clean_document_artifact_layer
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CorpusDocument

def clean_artifacts(doc: CorpusDocument, policy: PipelinePolicy) -> CorpusDocument:
    if not doc.text:
        return doc.with_stage('artifact_cleaning')
    working = clean_document_artifact_layer(doc.text, policy)
    return doc.with_text(working, modified=working != doc.text).with_stage('artifact_cleaning')
