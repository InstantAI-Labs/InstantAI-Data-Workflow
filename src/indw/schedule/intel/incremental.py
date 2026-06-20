from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from indw.config.defaults import LCI_INCREMENTAL_OVERLAP_MIN
from indw.schedule.intel.genome import GenomeProfile
from indw.schedule.intel.pci import _hash


@dataclass(frozen=True)
class IncrementalContext:
    doc_fingerprint: str
    genome_key: str
    overlap_ratio: float
    regions_unchanged: int
    regions_novel: int
    reuse_eligible: bool
    prior_observations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'doc_fingerprint': self.doc_fingerprint,
            'genome_key': self.genome_key,
            'overlap_ratio': round(self.overlap_ratio, 4),
            'regions_unchanged': self.regions_unchanged,
            'regions_novel': self.regions_novel,
            'reuse_eligible': self.reuse_eligible,
            'prior_observations': self.prior_observations,
        }


def content_fingerprint(line: dict[str, Any], genome: GenomeProfile) -> str:
    hashes: list[str] = []
    for chunk in line.get('chunks') or []:
        h = str(chunk.get('content_hash') or '')
        if h:
            hashes.append(h)
    if not hashes:
        raw = str(line.get('raw_text') or '')
        if raw:
            hashes.append(_hash((raw[:4096],)))
    return _hash((genome.genome_key, '|'.join(sorted(hashes)) or '0'))


def assess_incremental_overlap(
    *,
    doc_fp: str,
    genome_key: str,
    content_hash: str,
    known_gene_keys: tuple[str, ...],
    novel_gene_keys: tuple[str, ...],
    registry_row: tuple[int, str] | None,
) -> IncrementalContext:
    unchanged = len(known_gene_keys)
    novel = len(novel_gene_keys)
    total = max(unchanged + novel, 1)
    gene_overlap = unchanged / total
    prior_obs = 0
    hash_match = False
    if registry_row is not None:
        prior_obs, prior_hash = registry_row
        hash_match = prior_hash == content_hash
    overlap = gene_overlap
    if hash_match and prior_obs > 0:
        overlap = max(overlap, min(1.0, prior_obs / (prior_obs + 1)))
    reuse = (
        hash_match
        and gene_overlap >= LCI_INCREMENTAL_OVERLAP_MIN
        and prior_obs >= 1
    )
    return IncrementalContext(
        doc_fingerprint=doc_fp,
        genome_key=genome_key,
        overlap_ratio=overlap,
        regions_unchanged=unchanged,
        regions_novel=novel,
        reuse_eligible=reuse,
        prior_observations=prior_obs,
    )


class IncrementalRegistry:
    def __init__(self, conn, lock) -> None:
        self._conn = conn
        self._lock = lock
        self._hits = 0
        self._assessed = 0
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock():
            self._conn().executescript('''
                CREATE TABLE IF NOT EXISTS corpus_doc_registry (
                    doc_fp TEXT PRIMARY KEY,
                    genome_key TEXT NOT NULL,
                    family_key TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT '',
                    observation_count INTEGER NOT NULL DEFAULT 0,
                    last_seen REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_corpus_doc_genome ON corpus_doc_registry(genome_key);
                CREATE TABLE IF NOT EXISTS corpus_versions (
                    version_id TEXT PRIMARY KEY,
                    parent_id TEXT NOT NULL DEFAULT '',
                    doc_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL DEFAULT 0
                );
            ''')
            self._conn().commit()

    def lookup(self, doc_fp: str) -> tuple[int, str] | None:
        row = self._conn().execute(
            'SELECT observation_count, content_hash FROM corpus_doc_registry WHERE doc_fp=?',
            (doc_fp,),
        ).fetchone()
        if row is None:
            return None
        return int(row[0]), str(row[1])

    def record(
        self,
        *,
        doc_fp: str,
        genome_key: str,
        family_key: str,
        content_hash: str,
    ) -> None:
        now = time.time()
        with self._lock():
            row = self._conn().execute(
                'SELECT observation_count FROM corpus_doc_registry WHERE doc_fp=?',
                (doc_fp,),
            ).fetchone()
            if row is None:
                self._conn().execute(
                    'INSERT INTO corpus_doc_registry(doc_fp, genome_key, family_key, '
                    'content_hash, observation_count, last_seen) VALUES(?,?,?,?,1,?)',
                    (doc_fp, genome_key, family_key, content_hash, now),
                )
            else:
                self._conn().execute(
                    'UPDATE corpus_doc_registry SET observation_count=observation_count+1, '
                    'last_seen=?, content_hash=?, genome_key=?, family_key=? WHERE doc_fp=?',
                    (now, content_hash, genome_key, family_key, doc_fp),
                )
            self._conn().commit()

    def assess(
        self,
        line: dict[str, Any],
        genome: GenomeProfile,
        *,
        family_key: str,
        known_gene_keys: tuple[str, ...],
        novel_gene_keys: tuple[str, ...],
    ) -> IncrementalContext:
        doc_fp = content_fingerprint(line, genome)
        content_hash = _hash((doc_fp, genome.genome_key))
        row = self.lookup(doc_fp)
        ctx = assess_incremental_overlap(
            doc_fp=doc_fp,
            genome_key=genome.genome_key,
            content_hash=content_hash,
            known_gene_keys=known_gene_keys,
            novel_gene_keys=novel_gene_keys,
            registry_row=row,
        )
        self._assessed += 1
        if ctx.reuse_eligible:
            self._hits += 1
        self.record(
            doc_fp=doc_fp,
            genome_key=genome.genome_key,
            family_key=family_key,
            content_hash=content_hash,
        )
        return ctx

    def stats(self) -> dict[str, Any]:
        count = int(self._conn().execute(
            'SELECT COUNT(*) FROM corpus_doc_registry',
        ).fetchone()[0])
        return {
            'doc_registry': count,
            'incremental_assessed': self._assessed,
            'incremental_reuse_hits': self._hits,
            'incremental_reuse_rate': round(
                self._hits / self._assessed, 4,
            ) if self._assessed else 0.0,
        }
