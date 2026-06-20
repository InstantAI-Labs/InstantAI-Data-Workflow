from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from indw.store.eval.compare import compare_versions
from indw.store.eval.config import CorpusEvaluationConfig
from indw.store.eval.decision import decide_acceptance
from indw.store.eval.diversity import compute_diversity
from indw.store.eval.metrics import CorpusMetrics
from indw.store.eval.scoring import compute_corpus_score

@dataclass
class EvaluationValidationCase:
    name: str
    current: CorpusMetrics
    previous: Optional[CorpusMetrics]
    expect_decision: str

def _base() -> CorpusMetrics:
    return CorpusMetrics(
        version='v1',
        quality_score=0.81,
        duplicate_rate=0.04,
        toxicity_rate=0.01,
        pii_rate=0.002,
        knowledge_density=0.78,
        language_distribution={'en': 0.72, 'hi': 0.12, 'zh': 0.16},
        source_distribution={'web': 0.5, 'wiki': 0.3, 'code': 0.2},
        domain_distribution={'web': 0.5, 'wiki': 0.3, 'code': 0.2},
        accepted_documents=8000,
    )

def generate_validation_cases() -> list[EvaluationValidationCase]:
    base = _base()
    return [
        EvaluationValidationCase(
            'improved_corpus',
            CorpusMetrics(
                version='v2',
                quality_score=0.90,
                duplicate_rate=0.02,
                toxicity_rate=0.008,
                pii_rate=0.001,
                knowledge_density=0.86,
                language_distribution=base.language_distribution,
                source_distribution=base.source_distribution,
                domain_distribution=base.domain_distribution,
            ),
            base,
            'ACCEPT',
        ),
        EvaluationValidationCase(
            'degraded_corpus',
            CorpusMetrics(
                version='v3',
                quality_score=0.68,
                duplicate_rate=0.04,
                toxicity_rate=0.01,
                pii_rate=0.002,
                knowledge_density=0.55,
                language_distribution=base.language_distribution,
                source_distribution=base.source_distribution,
                domain_distribution=base.domain_distribution,
            ),
            base,
            'REJECT',
        ),
        EvaluationValidationCase(
            'duplicate_heavy',
            CorpusMetrics(
                version='v4',
                quality_score=0.79,
                duplicate_rate=0.18,
                toxicity_rate=0.01,
                pii_rate=0.002,
                knowledge_density=0.75,
                language_distribution=base.language_distribution,
                source_distribution=base.source_distribution,
                domain_distribution=base.domain_distribution,
            ),
            base,
            'REJECT',
        ),
        EvaluationValidationCase(
            'low_quality',
            CorpusMetrics(
                version='v5',
                quality_score=0.62,
                duplicate_rate=0.05,
                toxicity_rate=0.02,
                pii_rate=0.004,
                knowledge_density=0.48,
                language_distribution=base.language_distribution,
                source_distribution=base.source_distribution,
                domain_distribution=base.domain_distribution,
            ),
            base,
            'REJECT',
        ),
        EvaluationValidationCase(
            'marginal_improvement',
            CorpusMetrics(
                version='v6',
                quality_score=0.815,
                duplicate_rate=0.039,
                toxicity_rate=0.01,
                pii_rate=0.002,
                knowledge_density=0.79,
                language_distribution=base.language_distribution,
                source_distribution=base.source_distribution,
                domain_distribution=base.domain_distribution,
            ),
            base,
            'REVIEW',
        ),
        EvaluationValidationCase(
            'first_version',
            base,
            None,
            'ACCEPT',
        ),
    ]

def run_corpus_evaluation_validation(
    *,
    policy: Optional[CorpusEvaluationConfig] = None,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    pol = policy or CorpusEvaluationConfig.resolve()
    tp = tn = fp = fn = 0
    rows: list[dict[str, Any]] = []
    for case in generate_validation_cases():
        diversity = compute_diversity(
            language_distribution=case.current.language_distribution,
            source_distribution=case.current.source_distribution,
            domain_distribution=case.current.domain_distribution,
        )
        score = compute_corpus_score(case.current, diversity, config=pol)
        comparison = compare_versions(case.current, case.previous, config=pol)
        decision = decide_acceptance(
            score,
            comparison,
            config=pol,
            has_previous=case.previous is not None,
        )
        ok = decision.decision == case.expect_decision
        if case.expect_decision == 'ACCEPT':
            if ok:
                tp += 1
                outcome = 'tp'
            else:
                fn += 1
                outcome = 'fn'
        elif case.expect_decision == 'REJECT':
            if ok:
                tn += 1
                outcome = 'tn'
            else:
                fp += 1
                outcome = 'fp'
        else:
            if ok:
                tn += 1
                outcome = 'tn'
            else:
                if decision.decision == 'ACCEPT':
                    fp += 1
                else:
                    fn += 1
                outcome = 'fp' if decision.decision == 'ACCEPT' else 'fn'
        rows.append(
            {
                'name': case.name,
                'expect': case.expect_decision,
                'actual': decision.decision,
                'corpus_score': decision.corpus_score,
                'outcome': outcome,
                'reasons': decision.reasons,
            }
        )
    promote_checks = tp + fn
    reject_checks = tn + fp
    promotion_accuracy = tp / max(promote_checks, 1) if promote_checks else 1.0
    false_promotion_rate = fp / max(reject_checks + fp, 1)
    overall = (tp + tn) / max(len(rows), 1)
    passed = (
        overall >= pol.validation_min_promotion_accuracy
        and false_promotion_rate <= pol.validation_max_false_promotion_rate
    )
    report = {
        'CORPUS_EVALUATION_STATUS': 'PASS' if passed else 'FAIL',
        'metrics': {
            'promotion_accuracy': round(promotion_accuracy, 4),
            'overall_accuracy': round(overall, 4),
            'false_promotion_rate': round(false_promotion_rate, 4),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
        },
        'thresholds': {
            'min_promotion_accuracy': pol.validation_min_promotion_accuracy,
            'max_false_promotion_rate': pol.validation_max_false_promotion_rate,
        },
        'cases': rows,
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
