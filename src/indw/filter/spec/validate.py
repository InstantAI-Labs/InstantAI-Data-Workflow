from __future__ import annotations

from indw.filter.spec.document import EXPORT_ACTIONS
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CorpusDocument, ValidationIssue, ValidationResult

class ValidationEngine:
    def __init__(self, policy: PipelinePolicy | None = None) -> None:
        if policy is None:
            raise ValueError('ValidationEngine requires PipelinePolicy')
        self.policy = policy

    def validate(self, doc: CorpusDocument) -> ValidationResult:
        issues: list[ValidationIssue] = []
        lo, hi = self.policy.validation.score_bounds

        if self.policy.validation.require_doc_id and not doc.doc_id:
            issues.append(ValidationIssue('missing_doc_id', 'document id required'))

        if doc.scores is not None and not doc.scores.in_bounds(lo, hi):
            issues.append(ValidationIssue('score_out_of_bounds', 'canonical scores outside policy bounds'))

        if doc.decision is None:
            issues.append(ValidationIssue('missing_decision', 'curator decision required'))
        elif doc.decision.action == 'KEEP' and doc.decision.reason == 'repairable':
            issues.append(ValidationIssue('conflicting_decision', 'KEEP cannot use repairable reason'))

        if self.policy.validation.reject_conflicting_action and doc.decision is not None:
            action = doc.decision.action
            if action == 'DROP' and doc.exportable:
                issues.append(ValidationIssue('drop_export_conflict', 'dropped document marked exportable'))
            if action not in EXPORT_ACTIONS and doc.text and doc.exportable:
                issues.append(ValidationIssue('invalid_export', 'non-export action marked exportable'))

        if not doc.text and doc.decision is not None and doc.decision.action in EXPORT_ACTIONS:
            issues.append(ValidationIssue('empty_export', 'export action on empty text'))

        valid = not any(i.severity == 'error' for i in issues) and len(issues) == 0
        return ValidationResult(valid=valid, issues=tuple(issues))
