from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from indw.clean.artifact.discovery_corpus import CorpusStatsAccumulator, fragment_key
from indw.clean.artifact.decompose import compute_layout, normalize_ws

@dataclass
class TemplateProfile:
    prefix_template_score: float = 0.0
    suffix_template_score: float = 0.0
    middle_template_score: float = 0.0
    corpus_template_score: float = 0.0
    repeated_block_count: int = 0
    dominant_fingerprints: list[str] = field(default_factory=list)

    @property
    def template_density(self) -> float:
        return min(
            1.0,
            max(
                self.prefix_template_score,
                self.suffix_template_score,
                self.middle_template_score,
                self.corpus_template_score,
            ),
        )

def _edge_repeat_score(lines: list[str], *, edge: int = 3) -> float:
    if len(lines) < edge * 2:
        return 0.0
    head = [normalize_ws(ln).lower() for ln in lines[:edge] if ln.strip()]
    tail = [normalize_ws(ln).lower() for ln in lines[-edge:] if ln.strip()]
    if not head or not tail:
        return 0.0
    head_rep = 1.0 - len(set(head)) / len(head)
    tail_rep = 1.0 - len(set(tail)) / len(tail)
    short = sum(1 for ln in head + tail if len(ln.split()) <= 5) / max(len(head) + len(tail), 1)
    return min(1.0, (head_rep + tail_rep) * 0.45 + short * 0.35)

def _middle_repeat_score(paragraphs: list[str]) -> float:
    if len(paragraphs) < 3:
        return 0.0
    keys = [normalize_ws(p)[:160].lower() for p in paragraphs]
    counts: dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return min(1.0, repeated / max(len(keys), 1))

class TemplateMiner:
    def __init__(self, accumulator: CorpusStatsAccumulator | None = None) -> None:
        self.accumulator = accumulator

    def analyze(self, text: str) -> TemplateProfile:
        prof = TemplateProfile()
        if not text or not text.strip():
            return prof

        lines = [ln for ln in text.splitlines() if ln.strip()]
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        prof.prefix_template_score = _edge_repeat_score(lines, edge=3)
        prof.suffix_template_score = _edge_repeat_score(lines, edge=3)
        prof.middle_template_score = _middle_repeat_score(paras)
        prof.repeated_block_count = sum(
            1 for c in Counter(normalize_ws(p)[:120].lower() for p in paras).values() if c > 1
        )

        if self.accumulator is not None and self.accumulator.docs_seen > 0:
            scores: list[float] = []
            fps: list[str] = []
            for unit_text in (lines[:5] + lines[-5:] + [p[:200] for p in paras[:3]]):
                lay = compute_layout(unit_text)
                frag = self.accumulator.fragment_for_text(unit_text, lay)
                if frag and frag.doc_frequency > 1:
                    scores.append(frag.doc_rate(self.accumulator.docs_seen))
                    if frag.fingerprint:
                        fps.append(frag.fingerprint)
            if scores:
                prof.corpus_template_score = min(1.0, max(scores))
            prof.dominant_fingerprints = fps[:6]
        return prof
