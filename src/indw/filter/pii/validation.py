from __future__ import annotations

import json
import random
import secrets
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from indw.filter.pii.config import PiiPolicyConfig
from indw.filter.pii.detect import PiiDetector
from indw.filter.pii.redaction import redact_text

@dataclass
class PiiValidationCase:
    text: str
    label: str
    category: str
    expect_redact: bool = False

def _rand_email() -> str:
    user = ''.join(secrets.choice(string.ascii_lowercase) for _ in range(10))
    dom = ''.join(secrets.choice(string.ascii_lowercase) for _ in range(8))
    tld = secrets.choice(['io', 'tech', 'cloud', 'ai'])
    return f'{user}@{dom}.{tld}'

def _rand_api_key() -> str:
    prefix = secrets.choice(['zx', 'vk', 'qm', 'lp'])
    body = ''.join(secrets.choice(string.ascii_letters + string.digits + '_-') for _ in range(40))
    return f'{prefix}live_{body}'

def _rand_jwt() -> str:
    def part(n: int) -> str:
        return ''.join(secrets.choice(string.ascii_letters + string.digits + '_-') for _ in range(n))

    return f'eyJ{part(12)}.{part(24)}.{part(16)}'

def generate_dynamic_cases(seed: int = 42) -> list[PiiValidationCase]:
    rng = random.Random(seed)
    email = _rand_email()
    key = _rand_api_key()
    jwt = _rand_jwt()
    phone = f'+{rng.randint(40,99)}{rng.randint(100,999)}{rng.randint(1000000,9999999)}'
    return [
        PiiValidationCase(
            f'Contact {email} or call {phone} for billing. Account holder Maria Chen.',
            'pii',
            'contact_leak',
        ),
        PiiValidationCase(
            f'Production deployment failed. API credential: {key}',
            'pii',
            'production_secret',
        ),
        PiiValidationCase(
            f'Session token={jwt} stored in plaintext log.',
            'pii',
            'jwt_leak',
        ),
        PiiValidationCase(
            'Example API key:\nxxxxxxxx\nUse placeholder values only in tutorials.',
            'benign',
            'example_doc',
            expect_redact=False,
        ),
        PiiValidationCase(
            'Documentation: the email field accepts any user@domain.tld formatted address for validation testing.',
            'benign',
            'documentation',
        ),
        PiiValidationCase(
            'Research paper discusses GDPR impacts on pseudonymized patient records without listing real identifiers.',
            'benign',
            'educational',
        ),
        PiiValidationCase(
            f'राजेश शर्मा का पंजीकृत ईमेल { _rand_email() } है और फोन {phone} है।',
            'pii',
            'multilingual_hi',
        ),
        PiiValidationCase(
            'The transformer architecture uses multi-head attention for sequence modeling tasks.',
            'benign',
            'technical',
        ),
    ]

def run_pii_validation(
    *,
    policy: Optional[PiiPolicyConfig] = None,
    cases: Optional[list[PiiValidationCase]] = None,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    pol = policy or PiiPolicyConfig.resolve()
    detector = PiiDetector(pol)
    cases = cases or generate_dynamic_cases()
    pii_labels = {'pii'}
    benign_labels = {'benign'}
    tp = tn = fp = fn = 0
    redact_ok = 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        assessment = detector.assess(case.text)
        detected = (
            assessment.risk.should_reject
            or assessment.risk.should_redact
            or assessment.risk.band in ('reject', 'redact', 'hard_reject')
        )
        redacted = redact_text(
            case.text,
            entities=assessment.entities.entities,
            secrets=assessment.secrets.spans,
        )
        has_redaction = redacted != case.text and '<' in redacted
        if case.label in pii_labels:
            if detected:
                tp += 1
                outcome = 'tp'
            else:
                fn += 1
                outcome = 'fn'
            if case.expect_redact or assessment.risk.should_redact:
                if has_redaction:
                    redact_ok += 1
        else:
            if detected:
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
                'band': assessment.risk.band,
                'pii_score': round(assessment.risk.pii_score, 4),
                'entities': len(assessment.entities.entities),
                'secrets': len(assessment.secrets.spans),
                'context': assessment.context.context,
                'redacted': has_redaction,
                'reason': assessment.risk.reason,
            }
        )
    total_pii = max(tp + fn, 1)
    total_benign = max(tn + fp, 1)
    detection_rate = tp / total_pii
    false_positive_rate = fp / total_benign
    false_negative_rate = fn / total_pii
    passed = (
        detection_rate >= pol.validation_min_detection_rate
        and false_positive_rate <= pol.validation_max_false_positive_rate
        and false_negative_rate <= pol.validation_max_false_negative_rate
    )
    report = {
        'PII_SYSTEM_STATUS': 'PASS' if passed else 'FAIL',
        'metrics': {
            'detection_rate': round(detection_rate, 4),
            'false_positive_rate': round(false_positive_rate, 4),
            'false_negative_rate': round(false_negative_rate, 4),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'redaction_checks_passed': redact_ok,
        },
        'thresholds': {
            'min_detection_rate': pol.validation_min_detection_rate,
            'max_false_positive_rate': pol.validation_max_false_positive_rate,
            'max_false_negative_rate': pol.validation_max_false_negative_rate,
        },
        'cases': rows,
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
