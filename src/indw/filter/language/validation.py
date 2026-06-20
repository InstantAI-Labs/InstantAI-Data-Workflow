from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from indw.filter.language.config import LanguagePolicyConfig
from indw.filter.language.detect import LanguageIdentifier

@dataclass
class LanguageValidationCase:
    text: str
    expect_primary: str
    category: str
    expect_mixed: bool = False
    expect_unknown: bool = False
    min_confidence: float = 0.0

def generate_validation_cases(seed: int = 42) -> list[LanguageValidationCase]:
    rng = random.Random(seed)
    noisy_en = 'Th1s 1s a n01sy OCR smple w1th minor character subst1tutions for testing.'
    return [
        LanguageValidationCase(
            'The transformer architecture uses multi-head attention for sequence modeling.',
            'en',
            'monolingual_en',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            'यह एक हिंदी परीक्षण वाक्य है जो भाषा पहचान के लिए उपयोग किया जाता है।',
            'hi',
            'monolingual_hi',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            'مرحبا كيف حالك هذا نص عربي لاختبار كشف اللغة.',
            'ar',
            'monolingual_ar',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            '你好，这是一个用于语言识别测试的中文句子。',
            'zh',
            'monolingual_zh',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            'こんにちは、これは言語識別のテスト用の日本語文です。',
            'ja',
            'monolingual_ja',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            '안녕하세요, 이것은 언어 식별 테스트를 위한 한국어 문장입니다.',
            'ko',
            'monolingual_ko',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            'Hola, esta es una oración en español para pruebas de detección.',
            'es',
            'monolingual_es',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            'Bonjour, ceci est une phrase française pour les tests de détection.',
            'fr',
            'monolingual_fr',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            'Hallo, dies ist ein deutscher Satz für Spracherkennungstests.',
            'de',
            'monolingual_de',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            'Olá, esta é uma frase em português para testes de detecção.',
            'pt',
            'monolingual_pt',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            'Привет, это русское предложение для проверки определения языка.',
            'ru',
            'monolingual_ru',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            'Merhaba, bu dil algılama testi için Türkçe bir cümledir. '
            'Türkiye Cumhuriyeti vatandaşları için örnek metin.',
            'tr',
            'monolingual_tr',
            min_confidence=0.55,
        ),
        LanguageValidationCase(
            'Xin chào, đây là câu tiếng Việt dùng để kiểm tra nhận dạng ngôn ngữ.',
            'vi',
            'monolingual_vi',
            min_confidence=0.7,
        ),
        LanguageValidationCase(
            'Halo, ini adalah kalimat bahasa Indonesia untuk uji deteksi bahasa. '
            'Jakarta adalah ibu kota Indonesia.',
            'id',
            'monolingual_id',
            min_confidence=0.55,
        ),
        LanguageValidationCase(
            'Hello दुनिया',
            'en',
            'code_switch_en_hi',
            expect_mixed=True,
            min_confidence=0.35,
        ),
        LanguageValidationCase(
            'Welcome bienvenue 欢迎',
            'en',
            'multilingual_tri',
            expect_mixed=True,
            min_confidence=0.25,
        ),
        LanguageValidationCase(noisy_en, 'en', 'noisy_ocr', min_confidence=0.45),
        LanguageValidationCase(
            ''.join(rng.choice('⌘◇▣▤▥▦▧▨▧▦') for _ in range(80)),
            'und',
            'unknown_symbols',
            expect_unknown=True,
        ),
    ]

def run_language_validation(
    *,
    policy: Optional[LanguagePolicyConfig] = None,
    cases: Optional[list[LanguageValidationCase]] = None,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    pol = policy or LanguagePolicyConfig.resolve()
    identifier = LanguageIdentifier(pol)
    cases = cases or generate_validation_cases()
    mono_correct = mono_total = 0
    mixed_correct = mixed_total = 0
    unknown_correct = unknown_total = 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        assessment = identifier.assess(case.text)
        primary_ok = assessment.primary_language == case.expect_primary
        mixed_ok = assessment.mixed_language == case.expect_mixed
        unknown_ok = (
            assessment.primary_language in ('und', 'unknown')
            if case.expect_unknown
            else assessment.primary_language not in ('und', 'unknown')
        )
        conf_ok = assessment.confidence >= case.min_confidence or case.expect_unknown
        if case.expect_mixed:
            mixed_total += 1
            mixed_hit = mixed_ok
            if mixed_hit:
                mixed_correct += 1
            outcome = 'tp' if mixed_hit else 'fn'
        elif case.expect_unknown:
            unknown_total += 1
            low_conf = assessment.confidence < 0.25
            if unknown_ok or (low_conf and assessment.fragmentation >= 0.8):
                unknown_correct += 1
            outcome = 'tp' if unknown_ok or (low_conf and assessment.fragmentation >= 0.8) else 'fn'
        else:
            mono_total += 1
            if primary_ok and conf_ok:
                mono_correct += 1
            outcome = 'tp' if primary_ok and conf_ok else 'fn'
        rows.append(
            {
                'category': case.category,
                'expect_primary': case.expect_primary,
                'outcome': outcome,
                'primary_language': assessment.primary_language,
                'confidence': round(assessment.confidence, 4),
                'mixed_language': assessment.mixed_language,
                'languages': assessment.languages,
                'fragmentation': round(assessment.fragmentation, 4),
            }
        )
    accuracy = mono_correct / max(mono_total, 1)
    mixed_accuracy = mixed_correct / max(mixed_total, 1)
    unknown_accuracy = unknown_correct / max(unknown_total, 1)
    passed = (
        accuracy >= pol.validation_min_accuracy
        and mixed_accuracy >= pol.validation_min_mixed_accuracy
        and unknown_accuracy >= pol.validation_min_unknown_accuracy
    )
    report = {
        'LANGUAGE_SYSTEM_STATUS': 'PASS' if passed else 'FAIL',
        'metrics': {
            'accuracy': round(accuracy, 4),
            'mixed_language_accuracy': round(mixed_accuracy, 4),
            'unknown_detection_accuracy': round(unknown_accuracy, 4),
            'monolingual_correct': mono_correct,
            'monolingual_total': mono_total,
            'mixed_correct': mixed_correct,
            'mixed_total': mixed_total,
            'unknown_correct': unknown_correct,
            'unknown_total': unknown_total,
        },
        'thresholds': {
            'min_accuracy': pol.validation_min_accuracy,
            'min_mixed_accuracy': pol.validation_min_mixed_accuracy,
            'min_unknown_accuracy': pol.validation_min_unknown_accuracy,
        },
        'cases': rows,
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
