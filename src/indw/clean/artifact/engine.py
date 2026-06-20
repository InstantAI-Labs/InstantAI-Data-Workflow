from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from indw.clean.artifact.registry import ArtifactPatternRegistry, get_artifact_registry
from indw.clean.document.patterns import _UI_LINE
from indw.extract.assess.engine import (
    CleanStats,
    DocumentUnderstandingEngine,
    get_understanding_engine,
)

if TYPE_CHECKING:
    from indw.clean.artifact.discovery_engine import ArtifactDiscoveryEngine

_BOILERPLATE_HINT = re.compile(
    r'(?i)\b(?:cookie|privacy\s+policy|sign\s+up|advertisement|skip\s+to\s+content)\b'
)

@dataclass
class ArtifactSignals:
    line_hits: int = 0
    line_count: int = 0
    span_hits: int = 0
    span_chars: int = 0
    text_chars: int = 0
    registry_hits: dict[str, int] = field(default_factory=dict)

    @property
    def line_ratio(self) -> float:
        if self.line_count <= 0:
            return 0.0
        return min(1.0, self.line_hits / self.line_count)

    @property
    def span_density(self) -> float:
        if self.text_chars <= 0:
            return 0.0
        return min(1.0, self.span_chars / self.text_chars)

    @property
    def combined_density(self) -> float:
        return min(1.0, self.line_ratio * 0.45 + self.span_density * 0.55)

    @property
    def ui_noise_ratio(self) -> float:
        return self.combined_density

@dataclass
class InlineStripStats:
    spans_removed: int = 0
    chars_removed: int = 0
    lines_removed: int = 0

class ArtifactDetectionEngine:
    def __init__(
        self,
        registry: ArtifactPatternRegistry | None = None,
        *,
        understanding: DocumentUnderstandingEngine | None = None,
    ):
        self.registry = registry or get_artifact_registry()
        self._understanding = understanding or get_understanding_engine()

    def bind_discovery(
        self,
        discovery: ArtifactDiscoveryEngine | None,
        *,
        corpus_dir: str = '',
    ) -> None:
        if discovery is None:
            return
        self._understanding = get_understanding_engine(discovery=discovery, corpus_dir=corpus_dir)
        self._understanding.discovery = discovery
        if corpus_dir:
            self._understanding.corpus_dir = corpus_dir

    def analyze(self, text: str) -> ArtifactSignals:
        if not text or not text.strip():
            return ArtifactSignals()

        report = self._understanding.analyze(text)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        line_hits = sum(
            1 for ln in lines
            if _UI_LINE.search(ln) or _BOILERPLATE_HINT.search(ln) or self.registry.match_line(ln)
        )
        prose_chars = max(len(text), 1)
        span_chars = int(report.artifact_ratio * prose_chars)
        span_hits = max(1, int(report.artifact_ratio * len(lines))) if report.artifact_ratio > 0.02 else 0
        return ArtifactSignals(
            line_hits=line_hits,
            line_count=len(lines),
            span_hits=span_hits,
            span_chars=span_chars,
            text_chars=prose_chars,
            registry_hits=self.registry.scan_text(text),
        )

    def ui_noise_ratio(self, text: str) -> float:
        return self._understanding.ui_noise_ratio(text)

    def artifact_ratio(self, text: str) -> float:
        return self._understanding.artifact_ratio(text)

    def strip_inline(
        self,
        text: str,
        *,
        preserve_code_fences: bool = True,
        doc_id: str = '',
    ) -> tuple[str, InlineStripStats]:
        cleaned, stats = self._understanding.clean(
            text,
            preserve_code_fences=preserve_code_fences,
            doc_id=doc_id,
        )
        return cleaned, _clean_stats(stats)

    def strip_lines(self, text: str) -> tuple[str, int]:
        removed = 0
        kept: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                kept.append('')
                continue
            if (
                _UI_LINE.search(stripped)
                or _BOILERPLATE_HINT.search(stripped)
                or self.registry.match_line(stripped)
            ):
                removed += 1
                continue
            kept.append(line)
        out = '\n'.join(kept)
        out = re.sub(r'\n{3,}', '\n\n', out).strip()
        return out, removed

def _clean_stats(stats: CleanStats) -> InlineStripStats:
    return InlineStripStats(
        spans_removed=stats.spans_removed,
        chars_removed=stats.chars_removed,
        lines_removed=stats.units_removed,
    )

_ENGINE: ArtifactDetectionEngine | None = None

def get_artifact_engine() -> ArtifactDetectionEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ArtifactDetectionEngine()
    return _ENGINE

def reset_artifact_engine() -> None:
    global _ENGINE
    _ENGINE = None
    from indw.extract.assess.engine import reset_understanding_engines
    reset_understanding_engines()
