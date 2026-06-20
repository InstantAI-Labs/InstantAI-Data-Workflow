from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from indw.tools.metrics.alerts import alerts_from_regression
from indw.tools.metrics.config import ObservabilityPolicyConfig
from indw.tools.metrics.regression import analyze_regression
from indw.tools.metrics.snapshot import CorpusSnapshot

@dataclass
class ObservabilityValidationCase:
    name: str
    previous: CorpusSnapshot
    current: CorpusSnapshot
    expect_regression: bool

def _baseline() -> CorpusSnapshot:
    return CorpusSnapshot(
        version='v1',
        total_documents=10000,
        accepted_documents=8200,
        rejected_documents=1800,
        duplicate_rate=0.04,
        quality_score_mean=0.81,
        quality_score_p10=0.62,
        toxicity_rate=0.01,
        pii_rate=0.002,
        language_distribution={'en': 0.72, 'hi': 0.1, 'ar': 0.06, 'zh': 0.12},
        source_distribution={'web': 0.5, 'wiki': 0.3, 'code': 0.2},
        average_document_length=2400.0,
    )

def generate_validation_cases() -> list[ObservabilityValidationCase]:
    base = _baseline()
    return [
        ObservabilityValidationCase(
            'stable_rerun',
            base,
            CorpusSnapshot(
                version='v2',
                total_documents=10050,
                accepted_documents=8250,
                rejected_documents=1800,
                duplicate_rate=0.041,
                quality_score_mean=0.809,
                toxicity_rate=0.0105,
                pii_rate=0.0021,
                language_distribution={'en': 0.71, 'hi': 0.1, 'ar': 0.06, 'zh': 0.13},
                source_distribution={'web': 0.49, 'wiki': 0.31, 'code': 0.2},
                average_document_length=2410.0,
            ),
            expect_regression=False,
        ),
        ObservabilityValidationCase(
            'degraded_quality',
            base,
            CorpusSnapshot(
                version='v3',
                accepted_documents=7000,
                rejected_documents=3000,
                duplicate_rate=0.04,
                quality_score_mean=0.72,
                toxicity_rate=0.01,
                pii_rate=0.002,
                language_distribution=base.language_distribution,
                source_distribution=base.source_distribution,
            ),
            expect_regression=True,
        ),
        ObservabilityValidationCase(
            'duplicate_heavy',
            base,
            CorpusSnapshot(
                version='v4',
                accepted_documents=6000,
                duplicate_rate=0.14,
                quality_score_mean=0.8,
                toxicity_rate=0.01,
                pii_rate=0.002,
                language_distribution=base.language_distribution,
                source_distribution=base.source_distribution,
            ),
            expect_regression=True,
        ),
        ObservabilityValidationCase(
            'spam_toxic',
            base,
            CorpusSnapshot(
                version='v5',
                accepted_documents=7500,
                duplicate_rate=0.04,
                quality_score_mean=0.79,
                toxicity_rate=0.05,
                pii_rate=0.002,
                language_distribution=base.language_distribution,
                source_distribution=base.source_distribution,
            ),
            expect_regression=True,
        ),
        ObservabilityValidationCase(
            'language_drift',
            base,
            CorpusSnapshot(
                version='v6',
                accepted_documents=8000,
                duplicate_rate=0.04,
                quality_score_mean=0.8,
                toxicity_rate=0.01,
                pii_rate=0.002,
                language_distribution={'en': 0.35, 'hi': 0.35, 'ar': 0.15, 'zh': 0.15},
                source_distribution=base.source_distribution,
            ),
            expect_regression=True,
        ),
        ObservabilityValidationCase(
            'pii_spike',
            base,
            CorpusSnapshot(
                version='v7',
                accepted_documents=7800,
                duplicate_rate=0.04,
                quality_score_mean=0.8,
                toxicity_rate=0.01,
                pii_rate=0.025,
                language_distribution=base.language_distribution,
                source_distribution=base.source_distribution,
            ),
            expect_regression=True,
        ),
    ]

def run_observability_validation(
    *,
    policy: Optional[ObservabilityPolicyConfig] = None,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    pol = policy or ObservabilityPolicyConfig.resolve()
    cases = generate_validation_cases()
    tp = tn = fp = fn = 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = analyze_regression(case.current, case.previous, policy=pol)
        detected = result.regression_detected
        if case.expect_regression:
            if detected:
                tp += 1
                outcome = 'tp'
            else:
                fn += 1
                outcome = 'fn'
        else:
            if detected:
                fp += 1
                outcome = 'fp'
            else:
                tn += 1
                outcome = 'tn'
        rows.append(
            {
                'name': case.name,
                'outcome': outcome,
                'expect_regression': case.expect_regression,
                'regression_detected': detected,
                'reason': result.reason,
                'alerts': len(alerts_from_regression(result)),
            }
        )
    total_pos = max(tp + fn, 1)
    total_neg = max(tn + fp, 1)
    detection_accuracy = tp / total_pos
    false_alert_rate = fp / total_neg
    passed = (
        detection_accuracy >= pol.validation_min_regression_detection_accuracy
        and false_alert_rate <= pol.validation_max_false_alert_rate
    )
    report = {
        'OBSERVABILITY_SYSTEM_STATUS': 'PASS' if passed else 'FAIL',
        'metrics': {
            'regression_detection_accuracy': round(detection_accuracy, 4),
            'false_alert_rate': round(false_alert_rate, 4),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
        },
        'thresholds': {
            'min_regression_detection_accuracy': pol.validation_min_regression_detection_accuracy,
            'max_false_alert_rate': pol.validation_max_false_alert_rate,
        },
        'cases': rows,
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
