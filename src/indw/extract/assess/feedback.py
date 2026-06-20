from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from indw.clean.artifact.discovery_corpus import fragment_key
from indw.clean.artifact.decompose import compute_layout
from indw.clean.artifact.discovery_registry import DynamicArtifactRegistry
from indw.extract.nav.context import NavigationLearner
from indw.extract.roles.publication import PublicationLearner
from indw.extract.roles.education import EducationalLearner

@dataclass
class LeakageSample:
    text: str
    reason: str = 'output_leakage'
    fragment_keys: list[str] = field(default_factory=list)

class SelfLearningFeedback:
    def __init__(self, *, corpus_dir: str = '') -> None:
        self.corpus_dir = corpus_dir
        self._samples: list[LeakageSample] = []
        self._demote_weights: dict[str, float] = {}
        self.nav_learner = NavigationLearner()
        self.pub_learner = PublicationLearner()
        self.edu_learner = EducationalLearner()

    def record_leakage(self, text: str, *, reason: str = 'output_leakage') -> None:
        if not text or not text.strip():
            return
        keys: list[str] = []
        total = max(len(text), 1)
        for i, line in enumerate(text.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            lay = compute_layout(stripped)
            keys.append(fragment_key(stripped, lay))
            pos = i / max(len(text.splitlines()), 1)
            if reason in ('output_leakage', 'navigation_leakage'):
                self.nav_learner.record_surviving_nav(stripped, position_ratio=pos)
            if reason in ('output_leakage', 'publication_leakage'):
                self.pub_learner.record_surviving_scaffold(stripped, position_ratio=pos)
            if reason in ('output_leakage', 'instruction_leakage'):
                self.edu_learner.record_surviving_instruction(stripped, position_ratio=pos)
        self._samples.append(LeakageSample(text=text[:2000], reason=reason, fragment_keys=keys[:40]))
        if len(self._samples) > 500:
            self._samples.pop(0)

    def apply_to_registry(self, registry: DynamicArtifactRegistry) -> int:
        updated = 0
        for sample in self._samples:
            for key in sample.fragment_keys:
                entry = registry._entries.get(key)
                if entry is None:
                    continue
                entry.weight = max(0.05, entry.weight * 0.65)
                if entry.artifact_confidence > 0.4:
                    entry.artifact_confidence *= 0.8
                if entry.promoted and entry.artifact_confidence < 0.45:
                    entry.promoted = False
                self._demote_weights[key] = entry.weight
                updated += 1
        return updated

    def save(self) -> Path | None:
        if not self.corpus_dir or not self._samples:
            return None
        path = Path(self.corpus_dir) / 'understanding_feedback.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as fh:
            for sample in self._samples[-50:]:
                fh.write(json.dumps({
                    'reason': sample.reason,
                    'fragment_keys': sample.fragment_keys,
                    'preview': sample.text[:240],
                }, ensure_ascii=False) + '\n')
        return path
