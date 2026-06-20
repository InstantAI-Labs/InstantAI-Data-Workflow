from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import yaml

from indw.clean.document.config import CleaningConfig
from indw.filter.spec.quality import (
    BalanceConfig,
    DedupConfig,
    QualityPipelineConfig,
    SyntheticDefenseConfig,
)

MERGE_PASSAGE_A = (
    'Binary search locates a target value in a sorted array in O(log n) time. '
    'Each comparison eliminates half of the remaining search space until the target '
    'is found or the interval is empty. The algorithm requires sorted input and uses '
    'constant auxiliary space. Iterative and recursive formulations are equivalent '
    'in asymptotic complexity for typical implementations.'
)

MERGE_PASSAGE_B = (
    'Mitochondria generate ATP through oxidative phosphorylation in eukaryotic cells. '
    'The electron transport chain pumps protons across the inner membrane to drive '
    'ATP synthase rotation and phosphorylation of ADP. Cellular respiration couples '
    'nutrient oxidation to usable chemical energy for biosynthesis and transport.'
)

MERGE_PASSAGE_C = (
    'Neural networks learn hierarchical representations through backpropagation and '
    'gradient descent. Deep learning models stack nonlinear transformations to '
    'approximate complex functions from labeled or unlabeled data across vision, '
    'language, and multimodal domains.'
)

MERGE_CORPUS: tuple[str, ...] = (MERGE_PASSAGE_A, MERGE_PASSAGE_B, MERGE_PASSAGE_C)

def lenient_merge_config() -> QualityPipelineConfig:
    cfg = QualityPipelineConfig()
    cfg.cleaning = CleaningConfig(
        enabled=False,
        min_chars_after_clean=100,
        artifact_discovery=False,
    )
    cfg.balance = BalanceConfig(enabled=False)
    cfg.dedup = DedupConfig(exact=True, fuzzy=False, semantic=False)
    cfg.synthetic_defense = SyntheticDefenseConfig(enabled=False)
    cfg.curriculum.enabled = False
    cfg.orchestration = {'enabled': False}
    cfg.thresholds.min_chars = 100
    return cfg

def write_resolved_quality(work_dir: Path, cfg: QualityPipelineConfig) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / '_resolved_quality.yaml'
    payload = {
        'enabled': cfg.enabled,
        'thresholds': asdict(cfg.thresholds),
        'dedup': asdict(cfg.dedup),
        'balance': asdict(cfg.balance),
        'cleaning': asdict(cfg.cleaning),
        'synthetic_defense': asdict(cfg.synthetic_defense),
        'curriculum': asdict(cfg.curriculum),
        'orchestration': cfg.orchestration or {'enabled': False},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding='utf-8')
    return path

def write_raw_sources(
    raw_dir: Path,
    sources: Iterable[str],
    *,
    texts: Iterable[str] = MERGE_CORPUS,
) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for source in sources:
        src_dir = raw_dir / source
        src_dir.mkdir(parents=True, exist_ok=True)
        out = src_dir / 'data.jsonl'
        with out.open('w', encoding='utf-8') as fh:
            for text in texts:
                fh.write(json.dumps({'text': text, 'source': source}, ensure_ascii=False) + '\n')
        paths.append(out)
    return paths

def write_jsonl(path: Path, rows: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    return path
