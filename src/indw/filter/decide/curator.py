from __future__ import annotations

from indw.filter.spec.document import PipelineAction
from indw.filter.spec.pipeline import CuratorBand, CuratorPolicy, PipelinePolicy
from indw.filter.spec.document import CorpusDocument, CuratorDecision
from indw.filter.score.types import CanonicalScores

def _word_count(text: str) -> int:
    return len(text.split())

def _band_match(scores: CanonicalScores, band: CuratorBand, words: int) -> bool:
    return (
        scores.composite >= band.min_composite
        and scores.knowledge >= band.min_knowledge
        and scores.artifact_contamination <= band.max_artifact_contamination
        and words >= band.min_words
    )

class CuratorEngine:
    def __init__(self, policy: PipelinePolicy | None = None) -> None:
        if policy is None:
            raise ValueError('CuratorEngine requires PipelinePolicy')
        self.policy = policy

    @property
    def curator(self) -> CuratorPolicy:
        return self.policy.curator

    def decide(self, doc: CorpusDocument) -> CuratorDecision:
        if not doc.text:
            return CuratorDecision(action='DROP', reason='empty', detail='no_text')

        flags = set(doc.flags)
        if doc.classification is not None:
            flags.update(doc.classification.flags)
        scores = doc.scores
        hard = self.curator.hard_reject_flags & flags
        salvage = self.curator.salvage
        if scores is not None and hard:
            if (
                scores.technical_value >= salvage.technical_value_floor
                or scores.educational_value >= salvage.educational_value_floor
            ):
                hard = hard - salvage.salvage_flags
        if hard:
            return CuratorDecision(
                action='DROP',
                reason='hard_reject',
                detail=','.join(sorted(hard)[:3]),
            )

        if scores is None:
            return CuratorDecision(action='DROP', reason='missing_scores', detail='unscored')

        words = _word_count(doc.text)
        keep = self.curator.keep
        rewrite = self.curator.rewrite

        if _band_match(scores, keep, words):
            return CuratorDecision(
                action='KEEP',
                reason='repaired_quality' if doc.text_modified else 'high_quality',
                detail='meets_keep_band',
            )

        artifact_repairable = (
            scores.knowledge >= rewrite.min_knowledge
            and scores.artifact_contamination > keep.max_artifact_contamination
            and scores.artifact_contamination <= rewrite.max_artifact_contamination
        )
        if artifact_repairable and words >= rewrite.min_words:
            return CuratorDecision(action='REWRITE', reason='repairable_artifacts', detail='artifact_band')

        code_rewrite = self.curator.code_rewrite
        if (
            scores.structural_integrity >= code_rewrite.min_structural_integrity
            and scores.technical_value >= code_rewrite.min_technical_value
            and words >= rewrite.min_words
            and doc.classification is not None
            and doc.classification.content_type in ('code', 'mixed')
        ):
            return CuratorDecision(action='REWRITE', reason='structured_code', detail='algorithmic')

        if _band_match(scores, rewrite, words) or doc.text_modified:
            if scores.artifact_contamination <= rewrite.max_artifact_contamination:
                return CuratorDecision(action='REWRITE', reason='repairable', detail='rewrite_band')
            return CuratorDecision(action='DROP', reason='artifact_dominated', detail='above_rewrite_artifact')

        if words < rewrite.min_words:
            return CuratorDecision(action='DROP', reason='too_short', detail=str(words))

        return CuratorDecision(action='DROP', reason='low_value', detail='below_rewrite_band')
