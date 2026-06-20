from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.tools.metrics.snapshot import CorpusSnapshot
from indw.filter.gate.quality import QualityGate
from indw.filter.gate.reports import length_histogram

@dataclass
class CorpusMetrics:
    quality_score: float = 0.0
    duplicate_rate: float = 0.0
    toxicity_rate: float = 0.0
    pii_rate: float = 0.0
    language_distribution: dict[str, float] = field(default_factory=dict)
    source_distribution: dict[str, float] = field(default_factory=dict)
    domain_distribution: dict[str, float] = field(default_factory=dict)
    document_length_distribution: dict[str, float] = field(default_factory=dict)
    knowledge_density: float = 0.0
    reasoning_density: float = 0.0
    educational_value: float = 0.0
    factual_density: float = 0.0
    technical_content: float = 0.0
    repetitive_content: float = 0.0
    accepted_documents: int = 0
    version: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'quality_score': round(self.quality_score, 4),
            'duplicate_rate': round(self.duplicate_rate, 4),
            'toxicity_rate': round(self.toxicity_rate, 4),
            'pii_rate': round(self.pii_rate, 4),
            'language_distribution': self.language_distribution,
            'source_distribution': self.source_distribution,
            'domain_distribution': self.domain_distribution,
            'document_length_distribution': self.document_length_distribution,
            'knowledge_density': round(self.knowledge_density, 4),
            'reasoning_density': round(self.reasoning_density, 4),
            'educational_value': round(self.educational_value, 4),
            'factual_density': round(self.factual_density, 4),
            'technical_content': round(self.technical_content, 4),
            'repetitive_content': round(self.repetitive_content, 4),
            'accepted_documents': self.accepted_documents,
            'version': self.version,
        }

def collect_corpus_metrics(gate: QualityGate, snapshot: CorpusSnapshot) -> CorpusMetrics:
    qs = gate.stats.to_dict()
    domain_total = sum(gate.stats.domain_kept.values()) or 1
    domain_dist = {k: round(v / domain_total, 4) for k, v in gate.stats.domain_kept.items()}
    code_frac = domain_dist.get('code', 0.0)
    from indw.store.eval.knowledge import compute_evaluation_knowledge_density

    return CorpusMetrics(
        quality_score=snapshot.quality_score_mean,
        duplicate_rate=snapshot.duplicate_rate,
        toxicity_rate=snapshot.toxicity_rate,
        pii_rate=snapshot.pii_rate,
        language_distribution=dict(snapshot.language_distribution),
        source_distribution=dict(snapshot.source_distribution),
        domain_distribution=domain_dist,
        document_length_distribution=length_histogram(list(gate.stats.length_samples)),
        knowledge_density=compute_evaluation_knowledge_density(
            factual_density=float(qs.get('factual_density_mean', 0.0)),
            educational_value=float(qs.get('educational_value_mean', 0.0)),
            reasoning_density=float(qs.get('reasoning_density_mean', 0.0)),
            synthetic_score=float(qs.get('synthetic_score_mean', 0.0)),
            technical_fraction=code_frac,
        ),
        reasoning_density=float(qs.get('reasoning_density_mean', 0.0)),
        educational_value=float(qs.get('educational_value_mean', 0.0)),
        factual_density=float(qs.get('factual_density_mean', 0.0)),
        technical_content=code_frac,
        repetitive_content=float(qs.get('synthetic_score_mean', 0.0)),
        accepted_documents=snapshot.accepted_documents,
        version=snapshot.version,
    )
