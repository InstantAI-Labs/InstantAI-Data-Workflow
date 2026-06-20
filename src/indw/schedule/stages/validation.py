from __future__ import annotations

from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CorpusDocument
from indw.filter.spec.validate import ValidationEngine

def validate_document(
    doc: CorpusDocument,
    policy: PipelinePolicy,
    *,
    engine: ValidationEngine | None = None,
) -> CorpusDocument:
    validator = engine or ValidationEngine(policy)
    result = validator.validate(doc)
    return doc.with_validation(result)
