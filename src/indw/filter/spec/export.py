from __future__ import annotations

from typing import Any

from indw.filter.spec.document import EXPORT_ACTIONS
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CorpusDocument

def export_row(
    row: dict[str, Any],
    doc: CorpusDocument,
    *,
    text_key: str,
    policy: PipelinePolicy,
) -> dict[str, Any] | None:
    if doc.decision is None or doc.decision.action not in EXPORT_ACTIONS:
        return None
    if not doc.validation or not doc.validation.valid:
        return None
    if not doc.text:
        return None

    out = dict(row)
    out[text_key] = doc.text
    meta = dict(out.get('meta') or {}) if isinstance(out.get('meta'), dict) else {}

    if policy.export.get('attach_decision', True) and doc.decision is not None:
        meta['pipeline_action'] = doc.decision.action
        meta['pipeline_reason'] = doc.decision.reason
        meta['pipeline_detail'] = doc.decision.detail

    if policy.export.get('attach_scores', True) and doc.scores is not None:
        meta['scores'] = doc.scores.to_dict()
        meta['score_composite'] = doc.scores.composite

    if doc.classification is not None:
        meta['category'] = doc.classification.category
        meta['document_type'] = doc.classification.document_type
        meta['content_type'] = doc.classification.content_type

    meta['pipeline_stages'] = list(doc.stage_trace)
    meta['doc_id'] = doc.doc_id
    out['meta'] = meta
    return out
