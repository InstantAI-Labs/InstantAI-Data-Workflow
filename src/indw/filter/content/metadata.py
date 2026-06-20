from __future__ import annotations

from typing import Any

_STRIP_FROM_INPUT = frozenset({

    'ml_labels',

    'domain_labels',

    'domain_label',

    'topic',

    'topics',

    'categories',

    'category_tags',

    'semantic_tags',

    'predicted_domain',

    'predicted_topics',

    'classifier_output',

    'weak_labels',

    'label_probs',

    'aux_labels',

    'soft_domains',

    'label_confidence',

    'processing_temp',

    'temp_fields',

    '_internal',

    'shadow_ml',

    'route_scores',

})

_STRIP_PREFIXES = ('tmp_', 'debug_', 'x_', 'internal_')

_TRAINING_SCORE_KEYS = frozenset({

    'knowledge', 'educational', 'coherence', 'technical', 'artifact',

})

def strip_noisy_input_meta(meta: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:

    if not meta:

        return {}, []

    clean: dict[str, Any] = {}

    removed: list[str] = []

    for k, v in meta.items():

        if k in _STRIP_FROM_INPUT:

            removed.append(k)

            continue

        if any(k.startswith(p) for p in _STRIP_PREFIXES):

            removed.append(k)

            continue

        if k.endswith('_temp') or k.endswith('_debug'):

            removed.append(k)

            continue

        clean[k] = v

    return clean, removed

def _slim_route(route_dict: dict[str, Any]) -> dict[str, Any]:

    return {

        'route': route_dict.get('route', 'KEEP'),

        'reason': route_dict.get('reason', ''),

        'corpus_partition': route_dict.get('corpus_partition', 'main'),

    }

def _slim_scores(scores_dict: dict[str, float]) -> dict[str, float]:

    return {k: scores_dict[k] for k in _TRAINING_SCORE_KEYS if k in scores_dict}

def build_training_meta(

    *,

    route_dict: dict[str, Any],

    scores_dict: dict[str, float],

    score_composite: float,

    sample_weight: float,

    corpus_partition: str,

    preserved_meta: dict[str, Any] | None = None,

) -> dict[str, Any]:

    meta: dict[str, Any] = {}

    if preserved_meta:

        meta.update(preserved_meta)

    meta['route'] = _slim_route(route_dict)

    meta['scores'] = _slim_scores(scores_dict)

    meta['score_composite'] = round(score_composite, 2)

    meta['sample_weight'] = round(sample_weight, 4)

    meta['corpus_partition'] = corpus_partition

    return meta
