from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.clean.artifact.engine import get_artifact_engine

_LEAK_MARKERS = (
    'share|improve',
    'add comment',
    'up vote',
    'down vote',
    'you are here:',
    'load more stories',
    'subscribe now',
    'take the 2-minute tour',
)

@dataclass
class ArtifactLeakageReport:
    documents: int = 0
    marker_hits: int = 0
    density_hits: int = 0
    mean_ui_ratio: float = 0.0
    max_ui_ratio: float = 0.0
    samples: list[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.samples is None:
            self.samples = []

    @property
    def marker_rate_pct(self) -> float:
        if self.documents <= 0:
            return 0.0
        return 100.0 * self.marker_hits / self.documents

    @property
    def density_rate_pct(self) -> float:
        if self.documents <= 0:
            return 0.0
        return 100.0 * self.density_hits / self.documents

    def to_dict(self) -> dict[str, Any]:
        return {
            'documents': self.documents,
            'marker_hits': self.marker_hits,
            'density_hits': self.density_hits,
            'marker_rate_pct': round(self.marker_rate_pct, 3),
            'density_rate_pct': round(self.density_rate_pct, 3),
            'mean_ui_ratio': round(self.mean_ui_ratio, 4),
            'max_ui_ratio': round(self.max_ui_ratio, 4),
            'samples': self.samples[:10],
        }

def measure_text_leakage(text: str, *, density_threshold: float = 0.018) -> tuple[bool, bool, float]:
    engine = get_artifact_engine()
    sig = engine.analyze(text)
    low = text.lower()
    marker = any(m in low for m in _LEAK_MARKERS)
    dense = sig.ui_noise_ratio >= density_threshold or sig.span_hits > 0
    return marker, dense, sig.ui_noise_ratio

def measure_corpus_leakage(
    texts: list[str],
    *,
    density_threshold: float = 0.018,
    max_marker_rate_pct: float = 1.5,
    max_density_rate_pct: float = 2.5,
) -> tuple[ArtifactLeakageReport, list[str]]:
    report = ArtifactLeakageReport()
    issues: list[str] = []
    ratios: list[float] = []

    for text in texts:
        if not text or not text.strip():
            continue
        report.documents += 1
        marker, dense, ratio = measure_text_leakage(text, density_threshold=density_threshold)
        ratios.append(ratio)
        if marker:
            report.marker_hits += 1
            if len(report.samples) < 10:
                report.samples.append({
                    'kind': 'marker',
                    'ratio': round(ratio, 4),
                    'preview': text[:240].replace('\n', ' '),
                })
        if dense:
            report.density_hits += 1

    if ratios:
        report.mean_ui_ratio = sum(ratios) / len(ratios)
        report.max_ui_ratio = max(ratios)

    if report.documents > 0:
        if report.marker_rate_pct > max_marker_rate_pct:
            issues.append(
                f'Artifact marker leakage {report.marker_rate_pct:.2f}% exceeds {max_marker_rate_pct}%'
            )
        if report.density_rate_pct > max_density_rate_pct:
            issues.append(
                f'Artifact density leakage {report.density_rate_pct:.2f}% exceeds {max_density_rate_pct}%'
            )
    return report, issues
