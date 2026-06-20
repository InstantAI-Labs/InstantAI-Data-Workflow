from __future__ import annotations

from dataclasses import dataclass, field

from indw.clean.artifact.discovery_config import DiscoveryConfig
from indw.clean.artifact.discovery_corpus import CorpusStatsAccumulator, FragmentStats
from indw.clean.artifact.decompose import LayoutVector, compute_layout
from indw.clean.artifact.novelty import NoveltyScorer
from indw.clean.artifact.positional import PositionalLearner
from indw.clean.artifact.discovery_structural import StructuralLearner

def _structural_category(layout: LayoutVector, pos_conf: float, struct_conf: float) -> str:
    if layout.url_ratio > 0.06 and struct_conf >= 0.45:
        return 'navigation'
    if layout.punct_ratio > 0.2 and layout.alpha_ratio < 0.35:
        return 'separator'
    if layout.digit_ratio > 0.15 and layout.avg_len < 80:
        return 'metadata'
    if pos_conf >= 0.7 and layout.line_count <= 2:
        return 'wrapper'
    if layout.fence_ratio > 0.1:
        return 'code_adjacent'
    return 'learned'

@dataclass
class ArtifactEntry:
    key: str
    text_sample: str = ''
    doc_frequency: int = 0
    weight: float = 0.0
    frequency_confidence: float = 0.0
    position_confidence: float = 0.0
    structural_confidence: float = 0.0
    novelty_confidence: float = 0.0
    artifact_confidence: float = 0.0
    promoted: bool = False
    category: str = 'learned'

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'text_sample': self.text_sample[:120],
            'doc_frequency': self.doc_frequency,
            'weight': round(self.weight, 4),
            'frequency_confidence': round(self.frequency_confidence, 4),
            'position_confidence': round(self.position_confidence, 4),
            'structural_confidence': round(self.structural_confidence, 4),
            'novelty_confidence': round(self.novelty_confidence, 4),
            'artifact_confidence': round(self.artifact_confidence, 4),
            'promoted': self.promoted,
            'category': self.category,
        }

@dataclass
class DynamicArtifactRegistry:
    config: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    _entries: dict[str, ArtifactEntry] = field(default_factory=dict)
    _positional: PositionalLearner = field(default_factory=PositionalLearner)
    _structural: StructuralLearner = field(default_factory=StructuralLearner)
    _novelty: NoveltyScorer = field(default_factory=NoveltyScorer)

    def lookup(
        self,
        text: str,
        accumulator: CorpusStatsAccumulator,
        *,
        layout: LayoutVector | None = None,
        count_in_doc: int = 1,
    ) -> ArtifactEntry | None:
        lay = layout or compute_layout(text)
        frag = accumulator.fragment_for_text(text, lay)
        if frag is None:
            return None
        return self._entry_from_fragment(frag, text, accumulator, count_in_doc=count_in_doc, layout=lay)

    def _entry_from_fragment(
        self,
        frag: FragmentStats,
        text: str,
        accumulator: CorpusStatsAccumulator,
        *,
        count_in_doc: int = 1,
        layout: LayoutVector | None = None,
    ) -> ArtifactEntry:
        docs = accumulator.docs_seen
        freq_conf = frag.wilson_low(docs)
        pos_conf = self._positional.score(frag.position_histogram, frag.doc_frequency, docs)
        lay = layout or compute_layout(text)
        struct_conf = self._structural.score(lay, frag.doc_frequency, docs, text=text)
        nov_conf = self._novelty.novelty_confidence(text, frag.doc_frequency, count_in_doc)
        artifact_conf = min(
            1.0,
            freq_conf * 0.30
            + pos_conf * 0.25
            + struct_conf * 0.25
            + (1.0 - nov_conf) * 0.20,
        ) * frag.weight
        baseline = accumulator.baseline_doc_rate()
        promote_thr = accumulator.adaptive_promote_threshold(self.config.promote_doc_freq)
        if frag.doc_frequency < 2:
            artifact_conf *= 0.15
        promoted = (
            frag.doc_frequency >= promote_thr
            and pos_conf >= 0.6
            and freq_conf > baseline
            and artifact_conf >= 0.5
        )
        entry = ArtifactEntry(
            key=frag.key,
            text_sample=frag.text_sample or text[:200],
            doc_frequency=frag.doc_frequency,
            weight=frag.weight,
            frequency_confidence=freq_conf,
            position_confidence=pos_conf,
            structural_confidence=struct_conf,
            novelty_confidence=nov_conf,
            artifact_confidence=artifact_conf,
            promoted=promoted,
            category=_structural_category(lay, pos_conf, struct_conf),
        )
        self._entries[frag.key] = entry
        return entry

    def calibrate(self, accumulator: CorpusStatsAccumulator) -> dict[str, int]:
        promoted = 0
        demoted = 0
        baseline = accumulator.baseline_doc_rate()
        promote_thr = accumulator.adaptive_promote_threshold(self.config.promote_doc_freq)
        for frag in accumulator._fragments.values():
            entry = self._entry_from_fragment(frag, frag.text_sample, accumulator)
            was = entry.promoted
            entry.promoted = (
                frag.doc_frequency >= promote_thr
                and entry.position_confidence >= 0.6
                and entry.frequency_confidence > baseline
                and entry.artifact_confidence >= 0.5
                and frag.weight >= self.config.demote_weight
            )
            if entry.promoted and not was:
                promoted += 1
            elif was and not entry.promoted:
                demoted += 1
            self._entries[frag.key] = entry
        return {'promoted': promoted, 'demoted': demoted, 'total': len(self._entries)}

    def promoted_entries(self) -> list[ArtifactEntry]:
        return [e for e in self._entries.values() if e.promoted]

    def audit_flags(self, text: str, accumulator: CorpusStatsAccumulator) -> list[str]:
        flags: list[str] = []
        lay = compute_layout(text)
        entry = self.lookup(text, accumulator, layout=lay)
        if entry and entry.promoted and entry.artifact_confidence >= 0.55:
            if entry.structural_confidence >= 0.5 and lay.url_ratio > 0.05:
                flags.append('website_artifact')
            if entry.position_confidence >= 0.65 and lay.punct_ratio > 0.15:
                flags.append('forum_junk')
            if entry.position_confidence >= 0.7 and entry.frequency_confidence >= 0.4:
                flags.append('copyright_notice')
            if not flags:
                flags.append('learned_artifact')
        return flags

    def scan_text(self, text: str, accumulator: CorpusStatsAccumulator) -> dict[str, int]:
        counts: dict[str, int] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if len(stripped) < 4:
                continue
            entry = self.lookup(stripped, accumulator)
            if entry and entry.promoted:
                counts[entry.key[:12]] = counts.get(entry.key[:12], 0) + 1
        return counts

    def artifact_ratio(self, text: str, accumulator: CorpusStatsAccumulator) -> float:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return 0.0
        hits = 0.0
        for ln in lines:
            entry = self.lookup(ln.strip(), accumulator)
            if not entry:
                continue
            if entry.promoted:
                hits += entry.artifact_confidence
            elif (
                entry.doc_frequency >= accumulator.adaptive_promote_threshold(self.config.promote_doc_freq)
                and entry.artifact_confidence >= 0.55
            ):
                hits += entry.artifact_confidence * 0.6
        return min(1.0, hits / max(len(lines), 1))
