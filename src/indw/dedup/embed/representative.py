from __future__ import annotations

from indw.dedup.embed.contracts import ClusterMatch, DedupDocumentMeta

class QualityRepresentativeSelector:
    def __init__(self, quality_margin: float = 0.05):
        self.quality_margin = quality_margin

    def should_keep(
        self,
        meta: DedupDocumentMeta,
        match: ClusterMatch | None,
    ) -> bool:
        if match is None:
            return True
        incoming = meta.representative_rank()
        return incoming > match.representative_rank + self.quality_margin

    def should_replace(self, meta: DedupDocumentMeta, match: ClusterMatch) -> bool:
        return meta.representative_rank() > match.representative_rank + self.quality_margin
