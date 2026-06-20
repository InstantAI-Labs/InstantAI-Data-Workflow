from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from indw.config.defaults import DEFAULT_REFINER_SPEC, MIN_CHARS_AFTER_REPAIR
from orchestration.resolver.refs import ConfigRef
from orchestration.resolver.resolver import Resolver

DIVERSITY_PROTECTED_AXES = frozenset({
    'educational', 'factual', 'reasoning', 'referential', 'technical', 'procedural',
})

@dataclass
class RefinerConfig:
    enabled: bool = True
    min_knowledge_density: float = 38.0
    adaptive_density_only: bool = True
    diversity_density_discount: float = 8.0
    remove_heavy_truncation: bool = True
    repair_slight_truncation: bool = True
    remove_raw_code_dumps: bool = True
    strip_dominant_code_blocks: bool = True
    document_curation: bool = True
    min_chars_after_refine: int = MIN_CHARS_AFTER_REPAIR
    metadata_cleaning: bool = True
    strip_code_license_headers: bool = True
    corpus_dedup: bool = True
    chars_per_token: float = 3.8
    artifact_discovery: bool = True
    artifact_discovery_shadow: bool = False
    artifact_discovery_trim: bool = True
    artifact_discovery_primary: bool = False
    artifact_discovery_legacy_fallback: bool = True
    artifact_discovery_corpus_dir: str = ''
    semantic_cleaning: bool = True
    legacy_regex_cleaning: bool = False

    @classmethod
    def resolve(cls, spec: Optional[str] = None) -> RefinerConfig:
        refiner_spec = spec or DEFAULT_REFINER_SPEC
        try:
            resolved = Resolver.default().resolve(ConfigRef(kind='refining', id=refiner_spec))
            return cls.from_dict(dict(resolved.raw))
        except (FileNotFoundError, KeyError):
            return cls()

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> RefinerConfig:
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get('enabled', True)),
            min_knowledge_density=float(raw.get('min_knowledge_density', 38.0)),
            adaptive_density_only=bool(raw.get('adaptive_density_only', True)),
            diversity_density_discount=float(raw.get('diversity_density_discount', 8.0)),
            remove_heavy_truncation=bool(raw.get('remove_heavy_truncation', True)),
            repair_slight_truncation=bool(raw.get('repair_slight_truncation', True)),
            remove_raw_code_dumps=bool(raw.get('remove_raw_code_dumps', True)),
            strip_dominant_code_blocks=bool(raw.get('strip_dominant_code_blocks', True)),
            document_curation=bool(raw.get('document_curation', True)),
            min_chars_after_refine=int(raw.get('min_chars_after_refine', 60)),
            metadata_cleaning=bool(raw.get('metadata_cleaning', True)),
            strip_code_license_headers=bool(raw.get('strip_code_license_headers', True)),
            corpus_dedup=bool(raw.get('corpus_dedup', True)),
            chars_per_token=float(raw.get('chars_per_token', 3.8)),
            artifact_discovery=bool(raw.get('artifact_discovery', True)),
            artifact_discovery_shadow=bool(raw.get('artifact_discovery_shadow', False)),
            artifact_discovery_trim=bool(raw.get('artifact_discovery_trim', True)),
            artifact_discovery_primary=bool(raw.get('artifact_discovery_primary', False)),
            artifact_discovery_legacy_fallback=bool(raw.get('artifact_discovery_legacy_fallback', True)),
            artifact_discovery_corpus_dir=str(raw.get('artifact_discovery_corpus_dir', '') or ''),
            semantic_cleaning=bool(raw.get('semantic_cleaning', True)),
            legacy_regex_cleaning=bool(raw.get('legacy_regex_cleaning', False)),
        )
