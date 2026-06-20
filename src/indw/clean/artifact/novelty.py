from __future__ import annotations

from indw.filter.score.signals import shannon_entropy

class NoveltyScorer:
    def local_uniqueness(self, count_in_doc: int) -> float:
        if count_in_doc <= 1:
            return 1.0
        return 1.0 / count_in_doc

    def global_uniqueness(self, doc_frequency: int) -> float:
        if doc_frequency <= 0:
            return 1.0
        return 1.0 / doc_frequency

    def information_gain(self, text: str, doc_frequency: int) -> float:
        ent = shannon_entropy(text)
        gu = self.global_uniqueness(doc_frequency)
        return ent * gu

    def novelty_confidence(self, text: str, doc_frequency: int, count_in_doc: int) -> float:
        ig = self.information_gain(text, doc_frequency)
        lu = self.local_uniqueness(count_in_doc)
        max_ent = 8.0
        ig_norm = min(1.0, ig / max_ent)
        return min(1.0, ig_norm * 0.6 + lu * 0.4)

    def artifact_novelty_signal(
        self,
        text: str,
        doc_frequency: int,
        count_in_doc: int,
    ) -> float:
        return 1.0 - self.novelty_confidence(text, doc_frequency, count_in_doc)
