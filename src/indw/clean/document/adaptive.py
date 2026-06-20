from __future__ import annotations

from typing import Any

from indw.clean.document.patterns import _WORD
from indw.clean.document.value import (
    ContentValueSignals,
    analyze_content_value,
    build_analysis_bundle,
    is_information_rich,
)
from indw.clean.artifact.evidence import (
    AdaptiveBaselineEstimator,
    DocumentFeatureExtractor,
    PopulationAdaptiveScaler,
)

def bundle_raw(bundle: Any | None, text: str) -> Any:
    sem = getattr(bundle, '_bundle', None) if bundle is not None else None
    if sem is not None:
        return sem.raw
    return DocumentFeatureExtractor().extract(text)

def aggregate_component_noise(
    components: dict[str, float],
    *,
    evidence: Any | None = None,
) -> float:
    baseline = AdaptiveBaselineEstimator()
    vals = list(components.values())
    if not vals:
        return 0.0
    artifact = baseline.baseline(vals)
    if evidence is None:
        return min(1.0, artifact)
    neg = list(evidence.negative.values()) if evidence.negative else []
    if not neg:
        return min(1.0, artifact)
    noise = baseline.baseline(neg)
    substance = max(evidence.utility, evidence.semantic_strength)
    if substance > artifact:
        return min(1.0, artifact * (1.0 - substance * 0.5))
    return min(1.0, baseline.baseline([artifact, noise]))

def adaptive_artifact_threshold(
    *,
    evidence: Any | None = None,
    utility: float = 0.0,
) -> tuple[float, float]:
    baseline = AdaptiveBaselineEstimator()
    if evidence is not None:
        remove = evidence.threshold * (1.0 - evidence.uncertainty * 0.35)
        clean = baseline.baseline([remove * 0.4, evidence.uncertainty, 1.0 - evidence.utility])
        return min(1.0, max(remove, 0.08)), min(remove, max(clean, 0.04))
    spread = baseline.spread([utility, 0.15, 0.28])
    return baseline.baseline([spread, 0.22]), baseline.baseline([spread * 0.4, 0.08])

def metadata_only_drop(
    text: str,
    noise_ratio: float,
    *,
    bundle: Any | None = None,
    min_words: int = 30,
) -> bool:
    if not text or not text.strip():
        return True
    words = _WORD.findall(text)
    if not words:
        return True
    ctx = bundle or build_analysis_bundle(text)
    cv = analyze_content_value(text, bundle=ctx)
    if is_information_rich(cv, text=text):
        return False
    evidence = cv.evidence or ctx.evidence(text)
    if evidence.preserve:
        return False
    baseline = AdaptiveBaselineEstimator()
    wc = len(words)
    substance = baseline.baseline([
        evidence.utility,
        evidence.semantic_strength,
        1.0 - noise_ratio,
        cv.overall_value_score,
    ])
    noise_peer = baseline.baseline([noise_ratio, 1.0 - substance])
    short = PopulationAdaptiveScaler.short_doc_boundary(bundle_raw(ctx, text))
    if wc < min_words and noise_ratio > noise_peer:
        return True
    if wc < short and noise_ratio > baseline.spread([noise_ratio, substance]):
        return True
    meaningful = sum(1 for w in words if len(w) > 2)
    raw = bundle_raw(ctx, text)
    vocab = PopulationAdaptiveScaler.rate(meaningful, wc, raw.char_count)
    if vocab < substance and noise_ratio > baseline.baseline([noise_ratio, vocab]):
        return True
    return noise_ratio > baseline.baseline([noise_ratio, 1.0 - evidence.utility]) and substance < noise_peer

def chunk_noise_ceiling(
    metrics: Any,
    *,
    cfg: Any,
    cv: ContentValueSignals | None = None,
    code_heavy: bool = False,
) -> dict[str, float]:
    baseline = AdaptiveBaselineEstimator()
    evidence = cv.evidence if cv else None
    utility = evidence.utility if evidence else metrics.quality_score
    if evidence and evidence.preserve:
        return {'ui': 1.0, 'boiler': 1.0, 'dup': 1.0}
    cfg_ui = getattr(cfg, 'max_ui_noise_ratio', 0.45)
    cfg_boiler = getattr(cfg, 'max_boilerplate_ratio', 0.55)
    cfg_dup = getattr(cfg, 'max_duplicate_ratio', 0.35)
    peers = [
        metrics.ui_noise_ratio,
        metrics.boilerplate_ratio,
        metrics.spam_probability,
        1.0 - utility,
    ]
    spread = baseline.spread(peers)
    ui = baseline.baseline([cfg_ui, spread, metrics.ui_noise_ratio]) if not code_heavy else 1.0
    boiler = baseline.baseline([cfg_boiler, spread, metrics.boilerplate_ratio]) if not code_heavy else 1.0
    dup = baseline.baseline([cfg_dup, metrics.duplicate_ratio, 1.0 - utility])
    return {'ui': ui, 'boiler': boiler, 'dup': dup}

def adaptive_chunk_keep(
    text: str,
    metrics: Any,
    cfg: Any,
    *,
    cv: ContentValueSignals | None = None,
    code_heavy: bool = False,
) -> tuple[bool, str]:
    if cv is None:
        cv = analyze_content_value(text)
    if cv.evidence and cv.evidence.preserve:
        return True, ''
    ceiling = chunk_noise_ceiling(metrics, cfg=cfg, cv=cv, code_heavy=code_heavy)
    if metrics.ui_noise_ratio > ceiling['ui']:
        return False, 'ui_noise'
    if metrics.boilerplate_ratio > ceiling['boiler']:
        return False, 'boilerplate'
    if metrics.duplicate_ratio > ceiling['dup']:
        return False, 'duplicate'
    return True, ''

def adaptive_quality_score(
    *,
    cv: ContentValueSignals,
    sem_density: float,
    ui: float,
    duplicate_ratio: float,
) -> float:
    evidence = cv.evidence
    if evidence is not None:
        baseline = AdaptiveBaselineEstimator()
        penalty = baseline.baseline([ui, duplicate_ratio, 1.0 - evidence.semantic_strength])
        return max(0.0, min(1.0, evidence.utility - penalty * evidence.uncertainty))
    baseline = AdaptiveBaselineEstimator()
    signal = baseline.baseline([cv.overall_value_score, sem_density, cv.educational_score])
    penalty = baseline.baseline([ui, duplicate_ratio])
    return max(0.0, min(1.0, signal - penalty))
