from __future__ import annotations

from indw.clean.document.value import analyze_content_value, build_analysis_bundle
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CorpusDocument

def extract_knowledge(doc: CorpusDocument, policy: PipelinePolicy) -> CorpusDocument:
    del policy
    if not doc.text:
        return doc.with_stage('knowledge_extraction')
    bundle = build_analysis_bundle(doc.text)
    cv = analyze_content_value(doc.text, source=doc.provenance.source, bundle=bundle)
    flags = list(doc.flags)
    if cv.evidence is not None and cv.evidence.preserve:
        flags.append('high_value_evidence')
    return doc.with_flags(tuple(flags)).with_stage('knowledge_extraction')
