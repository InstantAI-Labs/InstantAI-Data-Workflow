from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from indw.filter.toxicity.config import ToxicityPolicyConfig
from indw.filter.toxicity.detect import ToxicityDetector

@dataclass
class ValidationCase:
    text: str
    label: str
    category: str

def default_validation_cases() -> list[ValidationCase]:
    return [
        ValidationCase(
            'You are worthless trash and I hope you suffer. Nobody wants you here.',
            'toxic',
            'harassment',
        ),
        ValidationCase("I will kill you if you show up again.", 'toxic', 'threats'),
        ValidationCase('Go kill yourself, nobody wants you here.', 'toxic', 'self_harm'),
        ValidationCase(
            'Join our cell today for the martyr operation and fight for the cause.',
            'toxic',
            'propaganda',
        ),
        ValidationCase('Kill all of them. Exterminate the entire group now.', 'toxic', 'hate'),
        ValidationCase(
            'According to the peer-reviewed study, hate crime legislation evolved after civil rights reforms. '
            'Historically, courts analyzed harassment statutes with academic discussion of constitutional limits.',
            'educational',
            'history',
        ),
        ValidationCase(
            'News report: investigators revealed the suspect threatened violence; journalists documented events '
            'without directing abuse toward victims.',
            'educational',
            'news',
        ),
        ValidationCase(
            'This paper presents methodology for detecting online harassment in datasets using labeled '
            'examples from research corpora and legal review of statutes.',
            'educational',
            'academic',
        ),
        ValidationCase(
            'In 1945 the tribunal documented war crimes and genocide for the historical record, citing '
            'primary sources and archival evidence in the legal judgment.',
            'educational',
            'historical',
        ),
        ValidationCase(
            'The transformer architecture maps queries to keys for parallelizable sequence modeling.',
            'neutral',
            'technical',
        ),
    ]

def run_toxicity_validation(
    *,
    policy: Optional[ToxicityPolicyConfig] = None,
    cases: Optional[list[ValidationCase]] = None,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    pol = policy or ToxicityPolicyConfig.resolve()
    detector = ToxicityDetector(pol)
    cases = cases or default_validation_cases()
    toxic_expected = {'toxic'}
    educational_expected = {'educational', 'neutral'}
    tp = tn = fp = fn = 0
    rows: list[dict[str, Any]] = []
    classifier_backend = 'none'
    for case in cases:
        edu_kw = (
            {'factual_density': 0.45, 'educational_value': 0.45}
            if case.label == 'educational'
            else {}
        )
        assessment = detector.assess(case.text, **edu_kw)
        classifier_backend = assessment.ml.backend
        final = assessment.final
        predicted_toxic = final.should_reject or final.band in ('reject', 'hard_reject')
        if case.label in toxic_expected:
            if predicted_toxic:
                tp += 1
                outcome = 'tp'
            else:
                fn += 1
                outcome = 'fn'
        else:
            if predicted_toxic:
                fp += 1
                outcome = 'fp'
            else:
                tn += 1
                outcome = 'tn'
        rows.append(
            {
                'category': case.category,
                'label': case.label,
                'outcome': outcome,
                'band': final.band,
                'final_toxicity_score': round(final.final_toxicity_score, 4),
                'classifier': assessment.ml.to_dict(),
                'context': assessment.context.context,
                'toxicity_reason': final.toxicity_reason,
            }
        )
    total_toxic = max(tp + fn, 1)
    total_benign = max(tn + fp, 1)
    detection_rate = tp / total_toxic
    false_positive_rate = fp / total_benign
    false_negative_rate = fn / total_toxic
    passed = (
        detection_rate >= pol.validation_min_detection_rate
        and false_positive_rate <= pol.validation_max_false_positive_rate
        and false_negative_rate <= pol.validation_max_false_negative_rate
    )
    report = {
        'TOXICITY_SYSTEM_STATUS': 'PASS' if passed else 'FAIL',
        'metrics': {
            'detection_rate': round(detection_rate, 4),
            'false_positive_rate': round(false_positive_rate, 4),
            'false_negative_rate': round(false_negative_rate, 4),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
        },
        'thresholds': {
            'min_detection_rate': pol.validation_min_detection_rate,
            'max_false_positive_rate': pol.validation_max_false_positive_rate,
            'max_false_negative_rate': pol.validation_max_false_negative_rate,
        },
        'classifier_backend': classifier_backend,
        'cases': rows,
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
