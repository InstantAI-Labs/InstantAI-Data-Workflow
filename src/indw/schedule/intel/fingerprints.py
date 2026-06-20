from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from indw.schedule.intel.scores import (
    char_entropy_norm,
    family_confidence,
    layout_complexity,
    novelty_score,
    structural_complexity,
)
from indw.schedule.intel.pci import FingerprintBundle, _hash, build_fingerprint_bundle, fingerprint_from_raw


def family_key(fp: FingerprintBundle) -> str:
    return f'{fp.structural}:{fp.wrapper}:{fp.section_shape}'


def semantic_fingerprint(text: str, fp: FingerprintBundle, *, scan: Any | None = None) -> str:
    if scan is not None and getattr(scan, 'semantic', ''):
        return scan.semantic
    sample = text[:2048]
    words = sample.lower().split()
    vocab = len(set(words))
    q_ratio = sample.count('?') / max(len(sample), 1)
    return _hash((
        fp.line_shape,
        str(vocab),
        f'{q_ratio:.4f}',
        sample[:64].lower(),
        sample[-64:].lower() if len(sample) > 64 else '',
    ))


@dataclass(frozen=True)
class IntelligenceBundle:
    fp: FingerprintBundle
    semantic: str
    layout: str
    entropy: float
    complexity: float
    layout_complexity: float
    novelty: float
    family_id: str
    family_key: str
    family_confidence: float
    observation_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'fp': self.fp.to_dict(),
            'semantic': self.semantic,
            'layout': self.layout,
            'entropy': round(self.entropy, 4),
            'complexity': round(self.complexity, 4),
            'layout_complexity': round(self.layout_complexity, 4),
            'novelty': round(self.novelty, 4),
            'family_id': self.family_id,
            'family_key': self.family_key,
            'family_confidence': round(self.family_confidence, 4),
            'observation_count': self.observation_count,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> IntelligenceBundle | None:
        fp = fingerprint_from_raw(raw.get('fp') if isinstance(raw.get('fp'), dict) else None)
        if fp is None:
            return None
        return IntelligenceBundle(
            fp=fp,
            semantic=str(raw.get('semantic') or ''),
            layout=str(raw.get('layout') or ''),
            entropy=float(raw.get('entropy') or 0.0),
            complexity=float(raw.get('complexity') or 0.0),
            layout_complexity=float(raw.get('layout_complexity') or 0.0),
            novelty=float(raw.get('novelty') or 1.0),
            family_id=str(raw.get('family_id') or ''),
            family_key=str(raw.get('family_key') or family_key(fp)),
            family_confidence=float(raw.get('family_confidence') or 0.0),
            observation_count=int(raw.get('observation_count') or 0),
        )

    def with_family(
        self,
        *,
        family_id: str,
        observation_count: int | None = None,
        verified_count: int = 0,
    ) -> IntelligenceBundle:
        obs = self.observation_count if observation_count is None else observation_count
        return IntelligenceBundle(
            fp=self.fp,
            semantic=self.semantic,
            layout=self.layout,
            entropy=self.entropy,
            complexity=self.complexity,
            layout_complexity=self.layout_complexity,
            novelty=novelty_score(
                observation_count=obs,
                verified=verified_count > 0,
            ),
            family_id=family_id,
            family_key=self.family_key,
            family_confidence=family_confidence(
                observation_count=obs,
                verified_count=verified_count,
            ),
            observation_count=obs,
        )

    def with_store_record(self, rec: Any) -> IntelligenceBundle:
        verified = int(getattr(rec, 'verified_count', 0) or 0)
        obs = int(getattr(rec, 'observation_count', self.observation_count) or 0)
        return IntelligenceBundle(
            fp=self.fp,
            semantic=self.semantic,
            layout=self.layout,
            entropy=self.entropy,
            complexity=self.complexity,
            layout_complexity=self.layout_complexity,
            novelty=novelty_score(
                observation_count=obs,
                verified=verified > 0,
            ),
            family_id=str(getattr(rec, 'family_id', self.family_id) or self.family_id),
            family_key=self.family_key,
            family_confidence=family_confidence(
                observation_count=obs,
                verified_count=verified,
            ),
            observation_count=obs,
        )


def build_intelligence_bundle(
    text: str,
    *,
    family_id: str = '',
    observation_count: int = 0,
    verified_count: int = 0,
    fp: FingerprintBundle | None = None,
    scan: Any | None = None,
) -> IntelligenceBundle:
    fp = fp or build_fingerprint_bundle(text)
    fkey = family_key(fp)
    fid = family_id or f'fam_{_hash((fkey,))[:12]}'
    ent = scan.entropy if scan is not None else char_entropy_norm(text)
    comp = structural_complexity(text, fp, scan=scan)
    lay = layout_complexity(fp)
    conf = family_confidence(
        observation_count=observation_count,
        verified_count=verified_count,
    )
    nov = novelty_score(
        observation_count=observation_count,
        verified=verified_count > 0,
    )
    return IntelligenceBundle(
        fp=fp,
        semantic=semantic_fingerprint(text, fp, scan=scan),
        layout=_hash((fp.wrapper, fp.line_shape)),
        entropy=ent,
        complexity=comp,
        layout_complexity=lay,
        novelty=nov,
        family_id=fid,
        family_key=fkey,
        family_confidence=conf,
        observation_count=observation_count,
    )
