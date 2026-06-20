from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from orchestration.resolver.immutable import thaw
from orchestration.resolver.refs import ConfigRef
from orchestration.resolver.resolver import Resolver

DEFAULT_DOCUMENT_GATE_SPEC = 'data/filtering/document_gate'

def _require_float(block: dict[str, Any], key: str) -> float:
    if key not in block:
        raise ValueError(f'document_gate policy missing {key}')
    return float(block[key])

def _require_int(block: dict[str, Any], key: str) -> int:
    if key not in block:
        raise ValueError(f'document_gate policy missing {key}')
    return int(block[key])

@dataclass(frozen=True)
class DocumentGatePolicy:
    max_replacement_chars: int
    min_keyboard_smash_hits: int
    alpha_floor: float
    repl_char_divisor: int
    ctrl_char_divisor: int
    smash_hit_weight: float
    word_count_smash_divisor: int
    html_tag_weight: int
    html_dom_weight: int
    html_min_tags_floor: int
    nav_meaningful_short_multiplier: float
    disambig_min_list_lines: int
    disambig_short_bound_line_multiplier: float
    seo_keyword_stuffing_multiplier: float
    seo_transaction_ratio_multiplier: float
    seo_link_density_word_divisor: int
    seo_meaningful_short_multiplier: float
    ai_discourse_template_multiplier: float
    ai_low_information_multiplier: float
    ai_min_filler_density: float
    ai_meaningful_short_multiplier: float
    short_junk_max_lines: int
    artifact_advertisement_commercial_multiplier: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DocumentGatePolicy:
        corruption = raw.get('corruption') or {}
        html = raw.get('html') or {}
        navigation = raw.get('navigation') or {}
        disambiguation = raw.get('disambiguation') or {}
        seo = raw.get('seo') or {}
        ai_filler = raw.get('ai_filler') or {}
        short_junk = raw.get('short_junk') or {}
        artifact = raw.get('artifact') or {}
        return cls(
            max_replacement_chars=_require_int(corruption, 'max_replacement_chars'),
            min_keyboard_smash_hits=_require_int(corruption, 'min_keyboard_smash_hits'),
            alpha_floor=_require_float(corruption, 'alpha_floor'),
            repl_char_divisor=_require_int(corruption, 'repl_char_divisor'),
            ctrl_char_divisor=_require_int(corruption, 'ctrl_char_divisor'),
            smash_hit_weight=_require_float(corruption, 'smash_hit_weight'),
            word_count_smash_divisor=_require_int(corruption, 'word_count_smash_divisor'),
            html_tag_weight=_require_int(html, 'tag_weight'),
            html_dom_weight=_require_int(html, 'dom_hit_weight'),
            html_min_tags_floor=_require_int(html, 'min_tags_floor'),
            nav_meaningful_short_multiplier=_require_float(navigation, 'meaningful_short_multiplier'),
            disambig_min_list_lines=_require_int(disambiguation, 'min_list_lines'),
            disambig_short_bound_line_multiplier=_require_float(disambiguation, 'short_bound_line_multiplier'),
            seo_keyword_stuffing_multiplier=_require_float(seo, 'keyword_stuffing_multiplier'),
            seo_transaction_ratio_multiplier=_require_float(seo, 'transaction_ratio_multiplier'),
            seo_link_density_word_divisor=_require_int(seo, 'link_density_word_divisor'),
            seo_meaningful_short_multiplier=_require_float(seo, 'meaningful_short_multiplier'),
            ai_discourse_template_multiplier=_require_float(ai_filler, 'discourse_template_multiplier'),
            ai_low_information_multiplier=_require_float(ai_filler, 'low_information_multiplier'),
            ai_min_filler_density=_require_float(ai_filler, 'min_filler_density'),
            ai_meaningful_short_multiplier=_require_float(ai_filler, 'meaningful_short_multiplier'),
            short_junk_max_lines=_require_int(short_junk, 'max_lines'),
            artifact_advertisement_commercial_multiplier=_require_float(
                artifact, 'advertisement_commercial_multiplier',
            ),
        )

@lru_cache(maxsize=1)
def resolve_document_gate_policy() -> DocumentGatePolicy:
    resolved = Resolver.default().resolve(ConfigRef(kind='quality', id=DEFAULT_DOCUMENT_GATE_SPEC))
    return DocumentGatePolicy.from_dict(thaw(resolved.raw))
