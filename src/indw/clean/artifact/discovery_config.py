from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class DiscoveryConfig:
    enabled: bool = True
    shadow: bool = True
    trim: bool = False
    primary: bool = False
    legacy_fallback: bool = True
    corpus_dir: str = ''
    min_trim_confidence: float = 0.92
    medium_trim_confidence: float = 0.72
    promote_doc_freq: int = 8
    demote_weight: float = 0.08
    decay: float = 0.95
    max_trim_ratio: float = 0.40
    min_doc_chars: int = 200

    @classmethod
    def from_cleaning(cls, raw: Any) -> DiscoveryConfig:
        if raw is None:
            return cls()
        if isinstance(raw, DiscoveryConfig):
            return raw
        if isinstance(raw, dict):
            return cls.from_dict(raw)
        return cls(
            enabled=bool(getattr(raw, 'artifact_discovery', True)),
            shadow=bool(getattr(raw, 'artifact_discovery_shadow', True)),
            trim=bool(getattr(raw, 'artifact_discovery_trim', False)),
            primary=bool(getattr(raw, 'artifact_discovery_primary', False)),
            legacy_fallback=bool(getattr(raw, 'artifact_discovery_legacy_fallback', True)),
            corpus_dir=str(getattr(raw, 'artifact_discovery_corpus_dir', '') or ''),
            min_trim_confidence=float(getattr(raw, 'artifact_discovery_min_trim_confidence', 0.92)),
            medium_trim_confidence=float(getattr(raw, 'artifact_discovery_medium_trim_confidence', 0.72)),
            promote_doc_freq=int(getattr(raw, 'artifact_discovery_promote_doc_freq', 8)),
            demote_weight=float(getattr(raw, 'artifact_discovery_demote_weight', 0.08)),
            decay=float(getattr(raw, 'artifact_discovery_decay', 0.95)),
            max_trim_ratio=float(getattr(raw, 'artifact_discovery_max_trim_ratio', 0.40)),
            min_doc_chars=int(getattr(raw, 'artifact_discovery_min_doc_chars', 200)),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> DiscoveryConfig:
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get('enabled', raw.get('artifact_discovery', True))),
            shadow=bool(raw.get('shadow', raw.get('artifact_discovery_shadow', True))),
            trim=bool(raw.get('trim', raw.get('artifact_discovery_trim', False))),
            primary=bool(raw.get('primary', raw.get('artifact_discovery_primary', False))),
            legacy_fallback=bool(raw.get('legacy_fallback', raw.get('artifact_discovery_legacy_fallback', True))),
            corpus_dir=str(raw.get('corpus_dir', raw.get('artifact_discovery_corpus_dir', '')) or ''),
            min_trim_confidence=float(
                raw.get('min_trim_confidence', raw.get('artifact_discovery_min_trim_confidence', 0.92))
            ),
            medium_trim_confidence=float(
                raw.get('medium_trim_confidence', raw.get('artifact_discovery_medium_trim_confidence', 0.72))
            ),
            promote_doc_freq=int(
                raw.get('promote_doc_freq', raw.get('artifact_discovery_promote_doc_freq', 8))
            ),
            demote_weight=float(
                raw.get('demote_weight', raw.get('artifact_discovery_demote_weight', 0.08))
            ),
            decay=float(raw.get('decay', raw.get('artifact_discovery_decay', 0.95))),
            max_trim_ratio=float(
                raw.get('max_trim_ratio', raw.get('artifact_discovery_max_trim_ratio', 0.40))
            ),
            min_doc_chars=int(raw.get('min_doc_chars', raw.get('artifact_discovery_min_doc_chars', 200))),
        )
