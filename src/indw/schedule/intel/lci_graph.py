from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

from indw.config.defaults import (
    LCI_GENE_DECAY_HALF_LIFE_SEC,
    LCI_GENE_MAX,
    LCI_GENE_PROMOTE_MATCH_STREAK,
    LCI_GENE_PROMOTE_MIN_OBS,
    LCI_GENE_RETIRE_UNVERIFIED_DAYS,
)
from indw.schedule.intel.genome import GenomeProfile, StructuralGene
from indw.schedule.intel.incremental import IncrementalContext, IncrementalRegistry
from indw.schedule.intel.inheritance import domain_lineage, inherited_confidence
from indw.schedule.intel.promotion import (
    PromotionResult,
    domain_promotion_eligible,
    family_promotion_eligible,
    gene_shape_stable,
    genes_structurally_aligned,
    kinds_for_promotion,
)
from indw.schedule.intel.store import IntelligenceStore
from indw.schedule.intel.pci import _hash


def shard_for_key(key: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    return int(_hash((key,))[:4], 16) % num_shards


def aged_gene_confidence(
    *,
    confidence: float,
    verified_count: int,
    last_validated: float,
    now: float | None = None,
) -> float:
    now = now or time.time()
    if verified_count <= 0:
        return confidence * 0.45
    elapsed = max(0.0, now - last_validated)
    half = max(86400.0, float(LCI_GENE_DECAY_HALF_LIFE_SEC))
    decay = 0.5 ** (elapsed / half)
    boost = min(1.0, math.log1p(verified_count) / math.log1p(24))
    return min(1.0, confidence * (0.35 + 0.65 * boost) * (0.4 + 0.6 * decay))


@dataclass(frozen=True)
class GeneRecord:
    gene_key: str
    kind: str
    observation_count: int
    verified_count: int
    confidence: float
    aged_confidence: float


@dataclass(frozen=True)
class LCIContext:
    genome: GenomeProfile
    gene_confidence: float
    domain_confidence: float
    inherited_confidence: float
    novel_gene_ratio: float
    overlap_ratio: float
    known_gene_count: int
    shard: int = 0
    incremental: IncrementalContext | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            **self.genome.to_dict(),
            'gene_confidence': round(self.gene_confidence, 4),
            'domain_confidence': round(self.domain_confidence, 4),
            'inherited_confidence': round(self.inherited_confidence, 4),
            'novel_gene_ratio': round(self.novel_gene_ratio, 4),
            'overlap_ratio': round(self.overlap_ratio, 4),
            'known_gene_count': self.known_gene_count,
            'shard': self.shard,
        }
        if self.incremental is not None:
            out['incremental'] = self.incremental.to_dict()
        return out


class LivingCorpusGraph:
    def __init__(self, store: IntelligenceStore, *, num_shards: int = 1) -> None:
        self._store = store
        self._num_shards = max(1, int(num_shards))
        self._promoted_genes = 0
        self._promotion_rejects = 0
        self._family_promotions = 0
        self._init_schema()
        self._incremental = IncrementalRegistry(self._conn, self._lock)

    def _conn(self):
        return self._store._conn

    def _lock(self):
        return self._store._lock

    def _init_schema(self) -> None:
        with self._lock():
            self._conn().executescript('''
                CREATE TABLE IF NOT EXISTS genes (
                    gene_key TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    observation_count INTEGER NOT NULL DEFAULT 0,
                    verified_count INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0,
                    last_validated REAL NOT NULL DEFAULT 0,
                    shape_sig TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_genes_kind ON genes(kind);
                CREATE TABLE IF NOT EXISTS domains (
                    domain_id TEXT PRIMARY KEY,
                    parent_id TEXT NOT NULL DEFAULT '',
                    observation_count INTEGER NOT NULL DEFAULT 0,
                    verified_count INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS family_genes (
                    family_id TEXT NOT NULL,
                    gene_key TEXT NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (family_id, gene_key)
                );
                CREATE TABLE IF NOT EXISTS corpus_overlap (
                    genome_key TEXT PRIMARY KEY,
                    family_key TEXT NOT NULL,
                    observation_count INTEGER NOT NULL DEFAULT 0,
                    last_seen REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS gene_promotions (
                    gene_key TEXT PRIMARY KEY,
                    streak INTEGER NOT NULL DEFAULT 0,
                    last_shape_sig TEXT NOT NULL DEFAULT '',
                    last_structural_hash TEXT NOT NULL DEFAULT '',
                    promoted_at REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS family_promotions (
                    family_id TEXT PRIMARY KEY,
                    streak INTEGER NOT NULL DEFAULT 0,
                    last_structural_hash TEXT NOT NULL DEFAULT '',
                    promoted_at REAL NOT NULL DEFAULT 0
                );
            ''')
            self._conn().commit()

    def lookup_gene(self, gene_key: str) -> GeneRecord | None:
        row = self._conn().execute(
            'SELECT gene_key, kind, observation_count, verified_count, confidence, last_validated '
            'FROM genes WHERE gene_key=?',
            (gene_key,),
        ).fetchone()
        if row is None:
            return None
        gkey, kind, obs, ver, conf, last = row
        aged = aged_gene_confidence(
            confidence=conf, verified_count=ver, last_validated=last,
        )
        return GeneRecord(gkey, kind, obs, ver, conf, aged)

    def observe_genome(
        self,
        genome: GenomeProfile,
        *,
        family_id: str,
        family_key: str,
    ) -> LCIContext:
        now = time.time()
        known: list[str] = []
        novel: list[str] = []
        confs: list[float] = []
        with self._lock():
            for gene in genome.genes:
                rec = self.lookup_gene(gene.gene_key)
                if rec is None:
                    novel.append(gene.gene_key)
                    self._conn().execute(
                        'INSERT INTO genes(gene_key, kind, observation_count, verified_count, '
                        'confidence, last_validated, shape_sig) VALUES(?,?,?,?,?,?,?)',
                        (gene.gene_key, gene.kind.value, 1, 0, 0.15, now, gene.shape_sig),
                    )
                    confs.append(0.15)
                else:
                    known.append(gene.gene_key)
                    obs = rec.observation_count + 1
                    ver = rec.verified_count
                    conf = min(1.0, (rec.confidence * (obs - 1) + rec.aged_confidence) / obs)
                    self._conn().execute(
                        'UPDATE genes SET observation_count=?, confidence=? WHERE gene_key=?',
                        (obs, conf, gene.gene_key),
                    )
                    confs.append(aged_gene_confidence(
                        confidence=conf, verified_count=ver, last_validated=now,
                    ))
                self._conn().execute(
                    'INSERT INTO family_genes(family_id, gene_key, hits) VALUES(?,?,1) '
                    'ON CONFLICT(family_id, gene_key) DO UPDATE SET hits=hits+1',
                    (family_id, gene.gene_key),
                )
            dom_row = self._conn().execute(
                'SELECT observation_count, verified_count, confidence FROM domains WHERE domain_id=?',
                (genome.domain_id,),
            ).fetchone()
            if dom_row is None:
                parent = ''
                from indw.schedule.intel.inheritance import DOMAIN_PARENTS
                parent = DOMAIN_PARENTS.get(genome.domain_id, '')
                self._conn().execute(
                    'INSERT INTO domains(domain_id, parent_id, observation_count, verified_count, '
                    'confidence, updated_at) VALUES(?,?,?,?,?,?)',
                    (genome.domain_id, parent, 1, 0, 0.2, now),
                )
                dom_conf = 0.2
            else:
                obs, ver, conf = dom_row
                obs += 1
                dom_conf = min(1.0, conf + 0.02)
                self._conn().execute(
                    'UPDATE domains SET observation_count=?, confidence=?, updated_at=? '
                    'WHERE domain_id=?',
                    (obs, dom_conf, now, genome.domain_id),
                )
            row = self._conn().execute(
                'SELECT observation_count FROM corpus_overlap WHERE genome_key=?',
                (genome.genome_key,),
            ).fetchone()
            if row is None:
                overlap_ratio = 0.0
                self._conn().execute(
                    'INSERT INTO corpus_overlap(genome_key, family_key, observation_count, last_seen) '
                    'VALUES(?,?,1,?)',
                    (genome.genome_key, family_key, now),
                )
            else:
                overlap_ratio = min(1.0, row[0] / max(row[0] + 1, 1))
                self._conn().execute(
                    'UPDATE corpus_overlap SET observation_count=observation_count+1, '
                    'last_seen=?, family_key=? WHERE genome_key=?',
                    (now, family_key, genome.genome_key),
                )
            self._conn().commit()
            self._evict_genes_if_needed()
        parent_confs = self._domain_confidences(domain_lineage(genome.domain_id))
        inh = inherited_confidence(
            domain_confidence=dom_conf,
            parent_confidences=parent_confs,
            domain_id=genome.domain_id,
        )
        gene_conf = sum(confs) / max(len(confs), 1)
        novel_ratio = len(novel) / max(len(genome.genes), 1)
        overlap = 1.0 - novel_ratio if genome.genes else 0.0
        if row is not None:
            overlap = max(overlap, overlap_ratio)
        genome_known = GenomeProfile(
            genes=genome.genes,
            domain_id=genome.domain_id,
            genome_key=genome.genome_key,
            gene_kinds=genome.gene_kinds,
            novel_gene_keys=tuple(novel),
            known_gene_keys=tuple(known),
        )
        return LCIContext(
            genome=genome_known,
            gene_confidence=gene_conf,
            domain_confidence=dom_conf,
            inherited_confidence=inh,
            novel_gene_ratio=novel_ratio,
            overlap_ratio=overlap,
            known_gene_count=len(known),
            shard=shard_for_key(family_key, self._num_shards),
        )

    def attach_incremental(
        self,
        lci: LCIContext,
        line: dict[str, Any],
        *,
        family_key: str,
    ) -> LCIContext:
        inc = self._incremental.assess(
            line,
            lci.genome,
            family_key=family_key,
            known_gene_keys=lci.genome.known_gene_keys,
            novel_gene_keys=lci.genome.novel_gene_keys,
        )
        overlap = max(lci.overlap_ratio, inc.overlap_ratio)
        return LCIContext(
            genome=lci.genome,
            gene_confidence=lci.gene_confidence,
            domain_confidence=lci.domain_confidence,
            inherited_confidence=lci.inherited_confidence,
            novel_gene_ratio=lci.novel_gene_ratio,
            overlap_ratio=overlap,
            known_gene_count=lci.known_gene_count,
            shard=lci.shard,
            incremental=inc,
        )

    def attempt_promotions(
        self,
        *,
        cleaned: GenomeProfile,
        raw: GenomeProfile | None,
        family_id: str,
        structural_hash: str,
        quality_ok: bool,
        intel_observation_count: int,
        intel_family_confidence: float,
        coordinator: Any = None,
    ) -> PromotionResult:
        if not quality_ok or not genes_structurally_aligned(raw, cleaned):
            self._promotion_rejects += len(cleaned.genes)
            return PromotionResult(
                genes_promoted=0,
                genes_rejected=len(cleaned.genes),
                family_promoted=False,
                domain_promoted=False,
                structural_hash=structural_hash,
                quality_ok=False,
            )
        now = time.time()
        promoted = 0
        rejected = 0
        with self._lock():
            for gene in cleaned.genes:
                if coordinator is not None and not coordinator.owns_key(gene.gene_key):
                    rejected += 1
                    continue
                locked = True
                if coordinator is not None:
                    locked = coordinator.try_promotion_lock(gene.gene_key)
                if not locked:
                    rejected += 1
                    continue
                try:
                    row = self._conn().execute(
                        'SELECT streak, last_shape_sig, last_structural_hash FROM gene_promotions '
                        'WHERE gene_key=?',
                        (gene.gene_key,),
                    ).fetchone()
                    streak = 0
                    last_shape = ''
                    last_hash = ''
                    if row is not None:
                        streak, last_shape, last_hash = int(row[0]), str(row[1]), str(row[2])
                    stable = gene_shape_stable(gene, last_shape)
                    hash_ok = not last_hash or last_hash == structural_hash
                    if stable and hash_ok:
                        streak += 1
                    else:
                        streak = 1
                    gene_row = self._conn().execute(
                        'SELECT observation_count, verified_count FROM genes WHERE gene_key=?',
                        (gene.gene_key,),
                    ).fetchone()
                    obs = int(gene_row[0]) if gene_row else 0
                    if (
                        streak >= LCI_GENE_PROMOTE_MATCH_STREAK
                        and obs >= LCI_GENE_PROMOTE_MIN_OBS
                    ):
                        self._conn().execute(
                            'UPDATE genes SET verified_count=verified_count+1, '
                            'last_validated=?, confidence=MIN(1.0, confidence+0.08) '
                            'WHERE gene_key=?',
                            (now, gene.gene_key),
                        )
                        self._conn().execute(
                            'INSERT INTO gene_promotions(gene_key, streak, last_shape_sig, '
                            'last_structural_hash, promoted_at) VALUES(?,?,?,?,?) '
                            'ON CONFLICT(gene_key) DO UPDATE SET streak=?, last_shape_sig=?, '
                            'last_structural_hash=?, promoted_at=?',
                            (
                                gene.gene_key, streak, gene.shape_sig, structural_hash, now,
                                streak, gene.shape_sig, structural_hash, now,
                            ),
                        )
                        promoted += 1
                        self._promoted_genes += 1
                    else:
                        self._conn().execute(
                            'INSERT INTO gene_promotions(gene_key, streak, last_shape_sig, '
                            'last_structural_hash, promoted_at) VALUES(?,?,?,?,0) '
                            'ON CONFLICT(gene_key) DO UPDATE SET streak=?, last_shape_sig=?, '
                            'last_structural_hash=?',
                            (
                                gene.gene_key, streak, gene.shape_sig, structural_hash,
                                streak, gene.shape_sig, structural_hash,
                            ),
                        )
                        rejected += 1
                finally:
                    if coordinator is not None:
                        coordinator.release_promotion_lock(gene.gene_key)
            fam_row = self._conn().execute(
                'SELECT streak, last_structural_hash FROM family_promotions WHERE family_id=?',
                (family_id,),
            ).fetchone()
            fam_streak = 0
            fam_last_hash = ''
            if fam_row is not None:
                fam_streak, fam_last_hash = int(fam_row[0]), str(fam_row[1])
            if fam_last_hash == structural_hash or not fam_last_hash:
                fam_streak += 1
            else:
                fam_streak = 1
            family_promoted = False
            if family_promotion_eligible(
                observation_count=intel_observation_count,
                family_confidence=intel_family_confidence,
                genes_promoted=promoted,
                streak=fam_streak,
            ):
                for kind in kinds_for_promotion(cleaned):
                    self._store.put_graph_node(
                        family_id=family_id,
                        kind=kind,
                        fp_hash=structural_hash,
                        payload={'genome_key': cleaned.genome_key, 'domain_id': cleaned.domain_id},
                        verified=True,
                    )
                self._conn().execute(
                    'INSERT INTO family_promotions(family_id, streak, last_structural_hash, promoted_at) '
                    'VALUES(?,?,?,?) ON CONFLICT(family_id) DO UPDATE SET streak=?, '
                    'last_structural_hash=?, promoted_at=?',
                    (family_id, fam_streak, structural_hash, now, fam_streak, structural_hash, now),
                )
                self._store.mark_verified(family_id)
                family_promoted = True
                self._family_promotions += 1
            else:
                self._conn().execute(
                    'INSERT INTO family_promotions(family_id, streak, last_structural_hash, promoted_at) '
                    'VALUES(?,?,?,0) ON CONFLICT(family_id) DO UPDATE SET streak=?, '
                    'last_structural_hash=?',
                    (family_id, fam_streak, structural_hash, fam_streak, structural_hash),
                )
            dom_row = self._conn().execute(
                'SELECT observation_count, verified_count FROM domains WHERE domain_id=?',
                (cleaned.domain_id,),
            ).fetchone()
            domain_promoted = False
            if dom_row is not None:
                dom_obs, dom_ver = int(dom_row[0]), int(dom_row[1])
                if domain_promotion_eligible(dom_obs, dom_ver + promoted):
                    self._conn().execute(
                        'UPDATE domains SET verified_count=verified_count+1, '
                        'confidence=MIN(1.0, confidence+0.05) WHERE domain_id=?',
                        (cleaned.domain_id,),
                    )
                    domain_promoted = True
            self._conn().commit()
        return PromotionResult(
            genes_promoted=promoted,
            genes_rejected=rejected,
            family_promoted=family_promoted,
            domain_promoted=domain_promoted,
            structural_hash=structural_hash,
            quality_ok=True,
        )

    def retire_stale_genes(self) -> int:
        cutoff = time.time() - float(LCI_GENE_RETIRE_UNVERIFIED_DAYS) * 86400
        with self._lock():
            victims = self._conn().execute(
                'SELECT gene_key FROM genes WHERE verified_count=0 AND last_validated < ? '
                'ORDER BY observation_count ASC LIMIT 500',
                (cutoff,),
            ).fetchall()
            for (gkey,) in victims:
                self._conn().execute('DELETE FROM family_genes WHERE gene_key=?', (gkey,))
                self._conn().execute('DELETE FROM gene_promotions WHERE gene_key=?', (gkey,))
                self._conn().execute('DELETE FROM genes WHERE gene_key=?', (gkey,))
            self._conn().commit()
        return len(victims)

    def preview_genome(
        self,
        genome: GenomeProfile,
        *,
        family_id: str,
        family_key: str,
    ) -> LCIContext:
        known: list[str] = []
        novel: list[str] = []
        confs: list[float] = []
        for gene in genome.genes:
            rec = self.lookup_gene(gene.gene_key)
            if rec is None:
                novel.append(gene.gene_key)
                confs.append(0.15)
            else:
                known.append(gene.gene_key)
                confs.append(rec.aged_confidence)
        dom_row = self._conn().execute(
            'SELECT confidence FROM domains WHERE domain_id=?',
            (genome.domain_id,),
        ).fetchone()
        dom_conf = float(dom_row[0]) if dom_row else 0.2
        parent_confs = self._domain_confidences(domain_lineage(genome.domain_id))
        inh = inherited_confidence(
            domain_confidence=dom_conf,
            parent_confidences=parent_confs,
            domain_id=genome.domain_id,
        )
        row = self._conn().execute(
            'SELECT observation_count FROM corpus_overlap WHERE genome_key=?',
            (genome.genome_key,),
        ).fetchone()
        overlap_ratio = 0.0
        if row is not None:
            overlap_ratio = min(1.0, row[0] / max(row[0] + 1, 1))
        novel_ratio = len(novel) / max(len(genome.genes), 1)
        overlap = max(1.0 - novel_ratio, overlap_ratio) if genome.genes else 0.0
        genome_out = GenomeProfile(
            genes=genome.genes,
            domain_id=genome.domain_id,
            genome_key=genome.genome_key,
            gene_kinds=genome.gene_kinds,
            novel_gene_keys=tuple(novel),
            known_gene_keys=tuple(known),
        )
        return LCIContext(
            genome=genome_out,
            gene_confidence=sum(confs) / max(len(confs), 1),
            domain_confidence=dom_conf,
            inherited_confidence=inh,
            novel_gene_ratio=novel_ratio,
            overlap_ratio=overlap,
            known_gene_count=len(known),
            shard=shard_for_key(family_key, self._num_shards),
        )

    def _domain_confidences(self, domains: tuple[str, ...]) -> dict[str, float]:
        if not domains:
            return {}
        placeholders = ','.join('?' for _ in domains)
        rows = self._conn().execute(
            f'SELECT domain_id, confidence FROM domains WHERE domain_id IN ({placeholders})',
            domains,
        ).fetchall()
        return {r[0]: float(r[1]) for r in rows}

    def _evict_genes_if_needed(self) -> None:
        count = self._conn().execute('SELECT COUNT(*) FROM genes').fetchone()[0]
        if count <= LCI_GENE_MAX:
            return
        excess = count - LCI_GENE_MAX
        victims = self._conn().execute(
            'SELECT gene_key FROM genes ORDER BY verified_count ASC, observation_count ASC LIMIT ?',
            (excess,),
        ).fetchall()
        for (gkey,) in victims:
            self._conn().execute('DELETE FROM family_genes WHERE gene_key=?', (gkey,))
            self._conn().execute('DELETE FROM genes WHERE gene_key=?', (gkey,))
        self._conn().commit()

    def stats(self) -> dict[str, Any]:
        genes = int(self._conn().execute('SELECT COUNT(*) FROM genes').fetchone()[0])
        domains = int(self._conn().execute('SELECT COUNT(*) FROM domains').fetchone()[0])
        overlap = int(self._conn().execute('SELECT COUNT(*) FROM corpus_overlap').fetchone()[0])
        top_genes = self._conn().execute(
            'SELECT kind, COUNT(*) FROM genes GROUP BY kind ORDER BY COUNT(*) DESC LIMIT 8',
        ).fetchall()
        top_domains = self._conn().execute(
            'SELECT domain_id, observation_count, confidence FROM domains '
            'ORDER BY observation_count DESC LIMIT 8',
        ).fetchall()
        verified_genes = int(self._conn().execute(
            'SELECT COUNT(*) FROM genes WHERE verified_count > 0',
        ).fetchone()[0])
        return {
            'genes': genes,
            'genes_verified': verified_genes,
            'domains': domains,
            'genome_overlaps': overlap,
            'genes_by_kind': {k: int(n) for k, n in top_genes},
            'top_domains': [
                {'domain_id': r[0], 'observations': r[1], 'confidence': round(r[2], 4)}
                for r in top_domains
            ],
            'num_shards': self._num_shards,
            'promotions': {
                'genes_promoted': self._promoted_genes,
                'genes_rejected': self._promotion_rejects,
                'families_promoted': self._family_promotions,
            },
            'incremental': self._incremental.stats(),
        }
