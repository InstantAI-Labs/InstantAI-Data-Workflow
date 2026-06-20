from __future__ import annotations

from pathlib import Path

from indw.store.corpus.manifest import file_sha256
from indw.schedule.state.checkpoint import MergeCheckpoint
from indw.schedule.config.resolve import env_yes


def resolve_config_fingerprints(work_dir: Path) -> tuple[str, str]:
    root = Path(work_dir)
    if (root / 'merge').is_dir():
        root = root.parent if (root.parent / '_resolved_quality.yaml').exists() else root
    quality = root / '_resolved_quality.yaml'
    sources = root / '_resolved_sources.yaml'
    return (
        file_sha256(quality) if quality.is_file() else '',
        file_sha256(sources) if sources.is_file() else '',
    )


def bind_checkpoint_config(
    checkpoint: MergeCheckpoint,
    work_dir: Path,
    *,
    fresh: bool,
) -> None:
    quality_fp, sources_fp = resolve_config_fingerprints(work_dir)
    if fresh or not checkpoint.quality_config_fingerprint:
        checkpoint.quality_config_fingerprint = quality_fp
        checkpoint.sources_config_fingerprint = sources_fp
        return
    allow = env_yes('INSTANT_ALLOW_CONFIG_DRIFT')
    if quality_fp and checkpoint.quality_config_fingerprint and quality_fp != checkpoint.quality_config_fingerprint:
        msg = (
            'Quality config fingerprint changed since checkpoint; '
            'use --fresh-merge or set INSTANT_ALLOW_CONFIG_DRIFT=1'
        )
        if not allow:
            raise RuntimeError(msg)
    if sources_fp and checkpoint.sources_config_fingerprint and sources_fp != checkpoint.sources_config_fingerprint:
        msg = (
            'Sources config fingerprint changed since checkpoint; '
            'use --fresh-merge or set INSTANT_ALLOW_CONFIG_DRIFT=1'
        )
        if not allow:
            raise RuntimeError(msg)
