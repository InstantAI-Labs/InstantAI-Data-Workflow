from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indw.config.defaults import ACIM_MAX_FAMILIES, ACIM_STORE_VERSION
from indw.dedup.storage import connect_sqlite, run_locked
from indw.schedule.intel.fingerprints import IntelligenceBundle
from indw.schedule.intel.pci import FingerprintBundle, _hash


@dataclass(frozen=True)
class FamilyRecord:
    family_id: str
    family_key: str
    label: str
    observation_count: int
    verified_count: int
    confidence: float
    layout: str
    semantic: str


class IntelligenceStore:
    def __init__(
        self,
        path: Path,
        *,
        version: str = ACIM_STORE_VERSION,
        readonly: bool = False,
    ) -> None:
        self.path = Path(path)
        self.version = version
        self.readonly = readonly
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if readonly:
            uri = f'file:{self.path.as_posix()}?mode=ro'
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self._conn = connect_sqlite(self.path, check_same_thread=False)
        self._init_schema()

    def _commit(self) -> None:
        run_locked(self._conn.commit, journal_dir=self.path.parent)

    @classmethod
    def open_readonly(cls, path: Path, *, version: str = ACIM_STORE_VERSION) -> IntelligenceStore:
        return cls(path, version=version, readonly=True)

    def _init_schema(self) -> None:
        if self.readonly:
            return
        self._conn.executescript('''
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS families (
                family_id TEXT PRIMARY KEY,
                family_key TEXT UNIQUE NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                observation_count INTEGER NOT NULL DEFAULT 0,
                verified_count INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0,
                layout TEXT NOT NULL DEFAULT '',
                semantic TEXT NOT NULL DEFAULT '',
                entropy_mean REAL NOT NULL DEFAULT 0,
                complexity_mean REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_families_key ON families(family_key);
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                family_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                fp_hash TEXT NOT NULL,
                payload TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                hits INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_graph_family ON graph_nodes(family_id, kind);
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                family_id TEXT NOT NULL,
                seq INTEGER,
                source TEXT,
                depth TEXT NOT NULL,
                reused INTEGER NOT NULL DEFAULT 0,
                ts REAL NOT NULL
            );
        ''')
        self._conn.execute(
            'INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)',
            ('version', self.version),
        )
        self._commit()

    def lookup_family(self, key: str) -> FamilyRecord | None:
        with self._lock:
            row = self._conn.execute(
                'SELECT family_id, family_key, label, observation_count, verified_count, '
                'confidence, layout, semantic FROM families WHERE family_key=?',
                (key,),
            ).fetchone()
        if row is None:
            return None
        return FamilyRecord(*row)

    def observe(self, intel: IntelligenceBundle, *, label: str = '') -> FamilyRecord:
        if self.readonly:
            rec = self.lookup_family(intel.family_key)
            if rec is None:
                return FamilyRecord(
                    intel.family_id, intel.family_key, label or intel.family_id,
                    0, 0, intel.family_confidence, intel.layout, intel.semantic,
                )
            return rec
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                'SELECT observation_count, verified_count, entropy_mean, complexity_mean '
                'FROM families WHERE family_key=?',
                (intel.family_key,),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    'INSERT INTO families(family_id, family_key, label, observation_count, '
                    'verified_count, confidence, layout, semantic, entropy_mean, complexity_mean, updated_at) '
                    'VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (
                        intel.family_id, intel.family_key, label or intel.family_id,
                        1, 0, intel.family_confidence,
                        intel.layout, intel.semantic,
                        intel.entropy, intel.complexity, now,
                    ),
                )
                obs_count, verified = 1, 0
            else:
                obs_count, verified, ent_m, comp_m = row
                obs_count += 1
                ent_m = (ent_m * (obs_count - 1) + intel.entropy) / obs_count
                comp_m = (comp_m * (obs_count - 1) + intel.complexity) / obs_count
                from indw.schedule.intel.scores import family_confidence
                conf = family_confidence(
                    observation_count=obs_count,
                    verified_count=verified,
                )
                self._conn.execute(
                    'UPDATE families SET observation_count=?, confidence=?, layout=?, semantic=?, '
                    'entropy_mean=?, complexity_mean=?, updated_at=? WHERE family_key=?',
                    (obs_count, conf, intel.layout, intel.semantic, ent_m, comp_m, now, intel.family_key),
                )
            self._commit()
            self._evict_if_needed()
        rec = self.lookup_family(intel.family_key)
        assert rec is not None
        return rec

    def record_observation_event(
        self,
        *,
        family_id: str,
        seq: int,
        source: str,
        depth: str,
        reused: bool,
    ) -> None:
        if self.readonly:
            return
        with self._lock:
            self._conn.execute(
                'INSERT INTO observations(family_id, seq, source, depth, reused, ts) '
                'VALUES(?,?,?,?,?,?)',
                (family_id, seq, source, depth, int(reused), time.time()),
            )
            self._commit()

    def get_graph_node(self, family_id: str, kind: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                'SELECT payload, fp_hash, verified, hits FROM graph_nodes '
                'WHERE family_id=? AND kind=?',
                (family_id, kind),
            ).fetchone()
        if row is None:
            return None
        payload_raw, fp_hash, verified, hits = row
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            return None
        return {
            'payload': payload,
            'fp_hash': fp_hash,
            'verified': bool(verified),
            'hits': hits,
        }

    def put_graph_node(
        self,
        *,
        family_id: str,
        kind: str,
        fp_hash: str,
        payload: dict[str, Any],
        verified: bool = False,
    ) -> None:
        if self.readonly:
            return
        node_id = _hash((family_id, kind))
        now = time.time()
        with self._lock:
            self._conn.execute(
                'INSERT INTO graph_nodes(node_id, family_id, kind, fp_hash, payload, verified, hits, updated_at) '
                'VALUES(?,?,?,?,?,?,0,?) '
                'ON CONFLICT(node_id) DO UPDATE SET fp_hash=excluded.fp_hash, payload=excluded.payload, '
                'verified=excluded.verified, updated_at=excluded.updated_at',
                (node_id, family_id, kind, fp_hash, json.dumps(payload, separators=(',', ':')), int(verified), now),
            )
            self._commit()

    def mark_verified(self, family_id: str) -> None:
        if self.readonly:
            return
        with self._lock:
            self._conn.execute(
                'UPDATE families SET verified_count=verified_count+1 WHERE family_id=?',
                (family_id,),
            )
            self._commit()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            fam_count = self._conn.execute('SELECT COUNT(*) FROM families').fetchone()[0]
            node_count = self._conn.execute('SELECT COUNT(*) FROM graph_nodes').fetchone()[0]
            obs_count = self._conn.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
            top = self._conn.execute(
                'SELECT family_id, observation_count, confidence FROM families '
                'ORDER BY observation_count DESC LIMIT 12',
            ).fetchall()
        return {
            'version': self.version,
            'families': int(fam_count),
            'graph_nodes': int(node_count),
            'observations': int(obs_count),
            'top_families': [
                {'family_id': r[0], 'observations': r[1], 'confidence': round(r[2], 4)}
                for r in top
            ],
        }

    def export_pci_snapshot(self) -> dict[str, Any]:
        rows = self._conn.execute(
            'SELECT family_key, label FROM families WHERE confidence >= 0.5 ORDER BY observation_count DESC',
        ).fetchall()
        return {
            'snapshot_id': self.version,
            'entries': {key: label for key, label in rows},
        }

    def _evict_if_needed(self) -> None:
        count = self._conn.execute('SELECT COUNT(*) FROM families').fetchone()[0]
        if count <= ACIM_MAX_FAMILIES:
            return
        excess = count - ACIM_MAX_FAMILIES
        victims = self._conn.execute(
            'SELECT family_id FROM families ORDER BY verified_count ASC, observation_count ASC LIMIT ?',
            (excess,),
        ).fetchall()
        for (fid,) in victims:
            self._conn.execute('DELETE FROM graph_nodes WHERE family_id=?', (fid,))
            self._conn.execute('DELETE FROM observations WHERE family_id=?', (fid,))
            self._conn.execute('DELETE FROM families WHERE family_id=?', (fid,))
        self._commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def structural_verify_hash(fp: FingerprintBundle) -> str:
    return _hash((fp.structural, fp.section_shape, fp.line_shape))
