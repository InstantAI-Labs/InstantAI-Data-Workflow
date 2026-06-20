from __future__ import annotations

from typing import Any

from indw.store.io.json_codec import dumps_line
from indw.clean.corpus import CleaningResult
from indw.filter.gate.scorer import DocumentScore


def mixture_index_row(
    *,
    src_name: str,
    doc: DocumentScore,
    clean_result: CleaningResult,
) -> str:
    cm = clean_result.metrics
    return dumps_line(
        {
            'source': src_name,
            'domain': doc.domain,
            'language': doc.language,
            'language_confidence': round(doc.language_confidence, 4),
            'mixed_language': doc.mixed_language,
            'dominant_script': getattr(doc.script_profile, 'dominant_script', 'other'),
            'mixed_script_score': round(getattr(doc.script_profile, 'mixed_script_score', 0.0), 4),
            'multilingual_quality': round(doc.multilingual_quality, 4),
            'chars_per_token': round(doc.chars_per_token, 4),
            'token_inflation_risk': round(doc.token_inflation_risk, 4),
            'score': round(doc.score, 4),
            'synthetic_score': round(doc.signals.synthetic_score, 4),
            'reasoning_density': round(doc.signals.reasoning_density, 4),
            'factual_density': round(doc.signals.factual_density, 4),
            'educational_value': round(doc.signals.educational_value, 4),
            'token_spam_score': round(doc.signals.token_spam_score, 4),
            'truncation_score': round(doc.signals.truncation_score, 4),
            'boilerplate_score': round(doc.signals.boilerplate_score, 4),
            'commercial_score': round(doc.signals.commercial_score, 4),
            'seo_spam_score': round(doc.signals.seo_spam_score, 4),
            'low_information_score': round(doc.signals.low_information_score, 4),
            'software_piracy_score': round(doc.signals.software_piracy_score, 4),
            'quality_score_10': round(doc.quality_score_10, 2),
            'quality_score_100': round(doc.quality_score_10 * 10.0, 1),
            'filter_decision': doc.filter_decision,
            'content_type': doc.content_type,
            'downrank_weight': round(doc.downrank_weight, 3),
            'toxicity_score': round(doc.toxicity_score, 4),
            'toxicity_reason': doc.toxicity_reason,
            'pii_score': round(doc.pii_score, 4),
            'pii_entities': doc.pii_entities,
            'pii_secrets': doc.pii_secrets,
            'pii_reason': doc.pii_reason,
            'context_len': doc.signals.length,
            'document_id': clean_result.document_id,
            'chunk_id': clean_result.chunk_id,
            'chunk_index': clean_result.chunk_index,
            'token_count': int(cm.token_estimate or max(1, doc.signals.length // 4)),
            'clean_word_count': cm.word_count,
            'clean_token_estimate': cm.token_estimate,
            'clean_code_ratio': round(cm.code_ratio, 4),
            'clean_ui_noise_ratio': round(cm.ui_noise_ratio, 4),
            'clean_duplicate_ratio': round(cm.duplicate_ratio, 4),
            'clean_quality_score': round(cm.quality_score, 4),
            'clean_educational_score': round(cm.educational_score, 4),
            'clean_technical_score': round(cm.technical_score, 4),
            'clean_semantic_density': round(cm.semantic_density, 4),
            'clean_boilerplate_score': round(cm.boilerplate_score, 4),
            'clean_spam_probability': round(cm.spam_probability, 4),
            'clean_commercial_probability': round(cm.commercial_probability, 4),
            'clean_domain': cm.domain,
            'content_category': doc.content_category,
            'information_density': round(
                doc.content_value.information_density, 4
            ) if doc.content_value else 0.0,
            'entertainment_score': round(
                doc.content_value.entertainment_score, 4
            ) if doc.content_value else 0.0,
            'storytelling_score': round(
                doc.content_value.storytelling_score, 4
            ) if doc.content_value else 0.0,
            'overall_value_score': round(
                doc.content_value.overall_value_score, 4
            ) if doc.content_value else 0.0,
            'reference_score': round(
                doc.content_value.reference_score, 4
            ) if doc.content_value else 0.0,
            'information_density_per_token': round(
                doc.content_value.information_density_per_token, 4
            ) if doc.content_value else 0.0,
            'narrative_filler_score': round(
                doc.content_value.narrative_filler_score, 4
            ) if doc.content_value else 0.0,
            'clean_information_density': round(cm.information_density, 4),
            'clean_entertainment_score': round(cm.entertainment_score, 4),
            'clean_storytelling_score': round(cm.storytelling_score, 4),
            'clean_overall_score': round(cm.overall_score, 4),
            'clean_category': cm.category,
            'utility_score': round(
                doc.training_utility.utility_score, 4
            ) if doc.training_utility else 0.0,
            'utility_confidence': round(
                doc.training_utility.confidence, 4
            ) if doc.training_utility else 0.0,
            'utility_novelty': round(
                doc.training_utility.novelty, 4
            ) if doc.training_utility else 0.0,
            'synthetic_penalty': round(
                doc.training_utility.synthetic_penalty, 4
            ) if doc.training_utility else 0.0,
            'hallucination_risk': round(
                doc.training_utility.hallucination_risk, 4
            ) if doc.training_utility else 0.0,
            'decision_confidence': round(
                getattr(doc, 'filter_confidence', doc.downrank_weight), 4
            ),
            'license': getattr(doc, 'license', 'Unknown'),
            'license_confidence': round(getattr(doc, 'license_confidence', 0.0), 4),
            'copyright_status': getattr(doc, 'copyright_status', 'unknown'),
            'attribution_required': bool(getattr(doc, 'attribution_required', False)),
            'document_type': getattr(doc, 'document_type', 'unknown'),
            'license_filter_action': (
                doc.license_assessment.filter_action if doc.license_assessment else 'KEEP'
            ),
            'license_filter_reason': (
                doc.license_assessment.filter_reason if doc.license_assessment else ''
            ),
        },
    )
