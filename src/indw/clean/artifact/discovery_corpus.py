from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from indw.clean.artifact.decompose import (
    DecomposedDocument,
    LayoutVector,
    compute_layout,
    normalize_ws,
    position_bin,
)
from indw.dedup.storage import connect_sqlite, run_locked
from indw.config.defaults import DISCOVERY_FRAGMENT_CACHE_MAX
from indw.util.stats import wilson_ci
from indw.filter.score.signals import shannon_entropy

import re

_DIGIT_COLLAPSE = re.compile(r'\d+')

def fuzzy_normalize(text: str) -> str:
    t = unicodedata.normalize('NFC', text)
    t = normalize_ws(t).lower()
    t = _DIGIT_COLLAPSE.sub('#', t)
    t = re.sub(r'[^\w\s|/\\>:.+#\-]', '', t)
    return t

def fragment_fingerprint(text: str, layout: LayoutVector) -> str:
    payload = f'{fuzzy_normalize(text)}|{layout.bucket()}'
    return hashlib.sha256(payload.encode('utf-8', errors='ignore')).hexdigest()[:24]

def fragment_key(text: str, layout: LayoutVector) -> str:
    norm = normalize_ws(text)
    payload = f'{norm}|{layout.bucket()}'
    return hashlib.sha256(payload.encode('utf-8', errors='ignore')).hexdigest()

@dataclass
class FragmentStats:
    key: str
    fingerprint: str = ''
    text_sample: str = ''
    doc_frequency: int = 0
    line_frequency: int = 0
    char_frequency: int = 0
    position_histogram: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    layout_bucket: str = ''
    char_entropy: float = 0.0
    last_seen_batch: int = 0
    first_seen_batch: int = 0
    weight: float = 1.0
    seen_in_batch: bool = False

    def doc_rate(self, docs_seen: int) -> float:
        return self.doc_frequency / max(docs_seen, 1)

    def wilson_low(self, docs_seen: int) -> float:
        return wilson_ci(self.doc_frequency, max(docs_seen, 1))['low']

class CorpusStatsStore:
    def __init__(self, db_path: Path | str | None) -> None:
        self.db_path = Path(db_path) if db_path else None
        self._conn: sqlite3.Connection | None = None
        if self.db_path:
            self._conn = connect_sqlite(self.db_path, check_same_thread=False)
            self._init_schema()

    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS fragments (
                key TEXT PRIMARY KEY,
                text_sample TEXT,
                doc_frequency INTEGER,
                line_frequency INTEGER,
                char_frequency INTEGER,
                position_histogram TEXT,
                layout_bucket TEXT,
                char_entropy REAL,
                last_seen_batch INTEGER,
                first_seen_batch INTEGER,
                weight REAL
            )
            '''
        )
        self._conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS corpus_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            '''
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_meta(self, key: str, default: str = '') -> str:
        if not self._conn:
            return default
        row = self._conn.execute(
            'SELECT value FROM corpus_meta WHERE key = ?', (key,)
        ).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        if not self._conn:
            return

        def _write() -> None:
            self._conn.execute(
                'INSERT OR REPLACE INTO corpus_meta (key, value) VALUES (?, ?)',
                (key, value),
            )
            self._conn.commit()

        run_locked(_write)

    def load_fragment(self, key: str) -> FragmentStats | None:
        if not self._conn:
            return None
        row = self._conn.execute('SELECT * FROM fragments WHERE key = ?', (key,)).fetchone()
        if not row:
            return None
        hist = json.loads(row[5]) if row[5] else [0, 0, 0, 0, 0]
        return FragmentStats(
            key=row[0],
            text_sample=row[1] or '',
            doc_frequency=int(row[2]),
            line_frequency=int(row[3]),
            char_frequency=int(row[4]),
            position_histogram=hist,
            layout_bucket=row[6] or '',
            char_entropy=float(row[7] or 0),
            last_seen_batch=int(row[8] or 0),
            first_seen_batch=int(row[9] or 0),
            weight=float(row[10] or 1.0),
        )

    def upsert_fragment(self, stats: FragmentStats) -> None:
        if not self._conn:
            return

        def _write() -> None:
            self._conn.execute(
                '''
                INSERT OR REPLACE INTO fragments
                (key, text_sample, doc_frequency, line_frequency, char_frequency,
                 position_histogram, layout_bucket, char_entropy,
                 last_seen_batch, first_seen_batch, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    stats.key,
                    stats.text_sample[:500],
                    stats.doc_frequency,
                    stats.line_frequency,
                    stats.char_frequency,
                    json.dumps(stats.position_histogram),
                    stats.layout_bucket,
                    stats.char_entropy,
                    stats.last_seen_batch,
                    stats.first_seen_batch,
                    stats.weight,
                ),
            )

        run_locked(_write)

    def commit(self) -> None:
        if not self._conn:
            return
        run_locked(self._conn.commit)

    def all_fragments(self) -> list[FragmentStats]:
        if not self._conn:
            return []
        rows = self._conn.execute('SELECT * FROM fragments').fetchall()
        out: list[FragmentStats] = []
        for row in rows:
            hist = json.loads(row[5]) if row[5] else [0, 0, 0, 0, 0]
            out.append(
                FragmentStats(
                    key=row[0],
                    text_sample=row[1] or '',
                    doc_frequency=int(row[2]),
                    line_frequency=int(row[3]),
                    char_frequency=int(row[4]),
                    position_histogram=hist,
                    layout_bucket=row[6] or '',
                    char_entropy=float(row[7] or 0),
                    last_seen_batch=int(row[8] or 0),
                    first_seen_batch=int(row[9] or 0),
                    weight=float(row[10] or 1.0),
                )
            )
        return out

class CorpusStatsAccumulator:
    def __init__(self, store: CorpusStatsStore | None = None) -> None:
        self.store = store or CorpusStatsStore(None)
        self._fragments: dict[str, FragmentStats] = {}
        self._fp_index: dict[str, str] = {}
        self._docs_seen: int = int(self.store.get_meta('docs_seen', '0') or 0)
        self._batch_id: int = int(self.store.get_meta('batch_id', '0') or 0)

    def _load_fragment(self, key: str) -> FragmentStats | None:
        frag = self._fragments.get(key)
        if frag is not None:
            return frag
        loaded = self.store.load_fragment(key)
        if loaded is None:
            return None
        self._fragments[key] = loaded
        if loaded.fingerprint:
            self._fp_index[loaded.fingerprint] = loaded.key
        return loaded

    def _trim_fragment_cache(self, *, max_entries: int | None = None) -> int:
        cap = max_entries if max_entries is not None else DISCOVERY_FRAGMENT_CACHE_MAX
        if len(self._fragments) <= cap:
            return 0
        target = max(1, int(cap * 0.9))
        ranked = sorted(
            self._fragments.items(),
            key=lambda item: (item[1].weight, item[1].last_seen_batch, -item[1].doc_frequency),
        )
        evicted = 0
        for key, frag in ranked:
            if len(self._fragments) <= target:
                break
            if frag.seen_in_batch:
                continue
            self._fragments.pop(key, None)
            if frag.fingerprint and self._fp_index.get(frag.fingerprint) == key:
                self._fp_index.pop(frag.fingerprint, None)
            evicted += 1
        return evicted

    @property
    def docs_seen(self) -> int:
        return self._docs_seen

    @property
    def batch_id(self) -> int:
        return self._batch_id

    def observe_document(self, doc: DecomposedDocument, *, doc_id: str = '') -> None:
        self._docs_seen += 1
        doc_len = max(doc.char_count, 1)
        seen_keys: set[str] = set()
        key_counts: dict[str, int] = {}

        for unit in doc.units:
            if unit.in_fence or unit.kind == 'code':
                continue
            if unit.kind not in ('line', 'paragraph', 'block', 'header', 'footer', 'list', 'table'):
                continue
            if len(unit.text.strip()) < 4:
                continue
            key = fragment_key(unit.text, unit.layout)
            key_counts[key] = key_counts.get(key, 0) + 1

        for unit in doc.units:
            if unit.in_fence or unit.kind == 'code':
                continue
            if unit.kind not in ('line', 'paragraph', 'block', 'header', 'footer', 'list', 'table'):
                continue
            if len(unit.text.strip()) < 4:
                continue
            key = fragment_key(unit.text, unit.layout)
            fp = fragment_fingerprint(unit.text, unit.layout)
            frag = self._fragments.get(key)
            if frag is None and fp in self._fp_index:
                frag = self._fragments.get(self._fp_index[fp])
            if frag is None:
                frag = self.store.load_fragment(key)
            if frag is None and fp in self._fp_index:
                frag = self.store.load_fragment(self._fp_index[fp])
            if frag is not None and frag.key not in self._fragments:
                self._fragments[frag.key] = frag
                if frag.fingerprint:
                    self._fp_index[frag.fingerprint] = frag.key
            if frag is None:
                frag = FragmentStats(
                    key=key,
                    fingerprint=fp,
                    text_sample=unit.text[:200],
                    layout_bucket=unit.layout.bucket(),
                    char_entropy=shannon_entropy(unit.text),
                    first_seen_batch=self._batch_id + 1,
                )
                self._fragments[key] = frag
                self._fp_index[fp] = key
            elif not frag.fingerprint:
                frag.fingerprint = fp
                self._fp_index[fp] = frag.key
            if key not in seen_keys:
                frag.doc_frequency += 1
                seen_keys.add(key)
            frag.line_frequency += 1
            frag.char_frequency += len(unit.text)
            bin_idx = position_bin(unit.start, doc_len)
            hist = list(frag.position_histogram)
            while len(hist) < 5:
                hist.append(0)
            hist[bin_idx] += 1
            frag.position_histogram = hist[:5]
            frag.char_entropy = shannon_entropy(unit.text)
            frag.last_seen_batch = self._batch_id + 1
            frag.seen_in_batch = True
            frag.weight = min(1.0, frag.weight + 0.02)

        self.store.set_meta('docs_seen', str(self._docs_seen))

    def end_batch(self, *, decay: float = 0.95) -> None:
        self._batch_id += 1
        for frag in self._fragments.values():
            if not frag.seen_in_batch:
                frag.weight *= decay
            else:
                frag.seen_in_batch = False
            self.store.upsert_fragment(frag)
        self.store.set_meta('batch_id', str(self._batch_id))
        self.store.commit()
        self._trim_fragment_cache()

    def get(self, key: str) -> FragmentStats | None:
        return self._load_fragment(key)

    def fragment_for_text(self, text: str, layout: LayoutVector | None = None) -> FragmentStats | None:
        lay = layout or compute_layout(text)
        key = fragment_key(text, lay)
        frag = self._load_fragment(key)
        if frag is not None:
            return frag
        fp = fragment_fingerprint(text, lay)
        alt = self._fp_index.get(fp)
        if alt:
            return self._load_fragment(alt)
        return None

    def adaptive_promote_threshold(self, base: int) -> int:
        if self._docs_seen < 50:
            return max(3, base // 2)
        if self._docs_seen < 200:
            return max(4, int(base * 0.75))
        return base

    def baseline_doc_rate(self) -> float:
        if not self._fragments:
            return 0.01
        rates = [f.doc_rate(self._docs_seen) for f in self._fragments.values()]
        rates.sort()
        idx = max(0, int(len(rates) * 0.75) - 1)
        return max(0.01, rates[idx] if rates else 0.01)
