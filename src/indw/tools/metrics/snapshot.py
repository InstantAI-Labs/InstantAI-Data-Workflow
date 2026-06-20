from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from indw.filter.gate.quality import QualityGate
from indw.filter.gate.reports import CorpusQualityReport, length_histogram

@dataclass
class CorpusSnapshot:
    version: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_documents: int = 0
    accepted_documents: int = 0
    rejected_documents: int = 0
    duplicate_rate: float = 0.0
    quality_score_mean: float = 0.0
    quality_score_p10: float = 0.0
    quality_score_distribution: dict[str, float] = field(default_factory=dict)
    toxicity_rate: float = 0.0
    pii_rate: float = 0.0
    language_distribution: dict[str, float] = field(default_factory=dict)
    average_document_length: float = 0.0
    source_distribution: dict[str, float] = field(default_factory=dict)
    reject_reasons: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'version': self.version,
            'timestamp': self.timestamp,
            'documents': self.accepted_documents,
            'total_documents': self.total_documents,
            'accepted_documents': self.accepted_documents,
            'rejected_documents': self.rejected_documents,
            'duplicate_rate': round(self.duplicate_rate, 4),
            'quality_score_mean': round(self.quality_score_mean, 4),
            'quality_score_p10': round(self.quality_score_p10, 4),
            'quality_score_distribution': self.quality_score_distribution,
            'toxicity_rate': round(self.toxicity_rate, 4),
            'pii_rate': round(self.pii_rate, 4),
            'language_distribution': self.language_distribution,
            'average_document_length': round(self.average_document_length, 2),
            'source_distribution': self.source_distribution,
            'reject_reasons': self.reject_reasons,
            'metadata': self.metadata,
        }

def _score_histogram(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {}
    buckets = {'0.0-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
    for s in scores:
        if s < 0.4:
            buckets['0.0-0.4'] += 1
        elif s < 0.6:
            buckets['0.4-0.6'] += 1
        elif s < 0.8:
            buckets['0.6-0.8'] += 1
        else:
            buckets['0.8-1.0'] += 1
    total = len(scores)
    return {k: round(v / total, 4) for k, v in buckets.items()}

def build_snapshot(
    gate: QualityGate,
    report: CorpusQualityReport,
    *,
    version: str,
    dedup_stats: Optional[dict[str, Any]] = None,
    merge_stats: Optional[dict[str, Any]] = None,
    corpus_manifest_version: Optional[int] = None,
) -> CorpusSnapshot:
    qs = gate.stats.to_dict()
    if merge_stats:
        if merge_stats.get('kept') is not None:
            kept = int(merge_stats['kept'])
            qs['kept'] = kept
        else:
            kept = int(qs.get('kept', 0))
        if merge_stats.get('rejected') is not None:
            rejected = int(merge_stats['rejected'])
            qs['rejected'] = rejected
        else:
            rejected = int(qs.get('rejected', 0))
        if merge_stats.get('scanned') is not None:
            scanned = int(merge_stats['scanned'])
        else:
            scanned = kept + rejected
    else:
        kept = int(qs.get('kept', 0))
        rejected = int(qs.get('rejected', 0))
        scanned = kept + rejected
    dedup = dedup_stats or {}
    exact_dup = int(dedup.get('exact_duplicates', 0))
    fuzzy_dup = int(dedup.get('fuzzy_duplicates', 0) or dedup.get('duplicates', 0))
    semantic_dup = int(dedup.get('semantic_duplicates', 0))
    dup_total = exact_dup + fuzzy_dup + semantic_dup
    duplicate_rate = dup_total / max(scanned + dup_total, 1)
    token_chars = int(qs.get('token_chars_kept', 0))
    avg_len = token_chars / max(kept, 1)
    lang_total = sum(gate.stats.language_kept.values()) or 1
    lang_dist = {k: v / lang_total for k, v in gate.stats.language_kept.items()}
    if not lang_dist:
        lang_dist = gate.lang_balancer.distribution()
    source_dist = gate.stats.source_distribution()
    tox_scanned = max(gate.toxicity_stats.documents_scanned, 1)
    tox_rate = gate.toxicity_stats.rejected / tox_scanned
    pii_scanned = max(gate.pii_stats.documents_scanned, 1)
    pii_rate = gate.pii_stats.rejected / pii_scanned
    scores = list(gate.stats.score_samples)
    domain_total = sum(gate.stats.domain_kept.values()) or 1
    domain_dist = {k: round(v / domain_total, 4) for k, v in gate.stats.domain_kept.items()}
    code_frac = domain_dist.get('code', 0.0)
    factual = float(qs.get('factual_density_mean', 0.0))
    educational = float(qs.get('educational_value_mean', 0.0))
    reasoning = float(qs.get('reasoning_density_mean', 0.0))
    synthetic = float(qs.get('synthetic_score_mean', 0.0))
    info = 0.30 * factual + 0.28 * educational + 0.27 * reasoning + 0.15 * code_frac
    knowledge_density = max(0.0, min(1.0, info * (1.0 - min(1.0, synthetic * 0.85) * 0.5)))
    avg_len = token_chars / max(kept, 1)
    length_dist = length_histogram(list(gate.stats.length_samples))
    return CorpusSnapshot(
        version=version,
        total_documents=scanned,
        accepted_documents=kept,
        rejected_documents=rejected,
        duplicate_rate=duplicate_rate,
        quality_score_mean=float(qs.get('score_mean', 0.0)),
        quality_score_p10=float(qs.get('score_p10', 0.0)),
        quality_score_distribution=_score_histogram(scores),
        toxicity_rate=tox_rate,
        pii_rate=pii_rate,
        language_distribution={k: round(v, 4) for k, v in lang_dist.items()},
        average_document_length=avg_len,
        source_distribution={k: round(v, 4) for k, v in source_dist.items()},
        reject_reasons=dict(qs.get('reject_reasons') or {}),
        metadata={
            'corpus_manifest_version': corpus_manifest_version,
            'merge_stats': merge_stats or {},
            'dedup': dedup,
            'report_created_at': report.created_at,
            'quality_signals': {
                'knowledge_density': round(knowledge_density, 4),
                'domain_distribution': domain_dist,
                'document_length_distribution': length_dist,
                'reasoning_density_mean': float(qs.get('reasoning_density_mean', 0.0)),
                'factual_density_mean': float(qs.get('factual_density_mean', 0.0)),
                'educational_value_mean': float(qs.get('educational_value_mean', 0.0)),
            },
        },
    )
