from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from indw.config.defaults import LCI_NUM_SHARDS, LCI_STORE_VERSION
from indw.schedule.intel.lci_graph import shard_for_key
from indw.schedule.config.resolve import env_int


class IntelligenceCoordinator:
    def __init__(
        self,
        merge_work: Path,
        *,
        num_shards: int = LCI_NUM_SHARDS,
        worker_id: str = 'coordinator',
    ) -> None:
        self.merge_work = Path(merge_work)
        self.num_shards = max(1, int(num_shards))
        self.worker_id = worker_id
        self._acim_dir = self.merge_work / 'acim'
        self._acim_dir.mkdir(parents=True, exist_ok=True)
        self._lock_dir = self._acim_dir / 'locks'
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        self._snap_dir = self._acim_dir / 'snapshots'
        self._snap_dir.mkdir(parents=True, exist_ok=True)

    def owns_key(self, key: str) -> bool:
        shard = env_int('INSTANT_LCI_WORKER_SHARD', -1, minimum=-1)
        if shard < 0:
            return True
        return shard_for_key(key, self.num_shards) == shard

    def try_promotion_lock(self, gene_key: str) -> bool:
        if not self.owns_key(gene_key):
            return False
        lock_path = self._lock_dir / f'{gene_key}.lock'
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, self.worker_id.encode('utf-8'))
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def release_promotion_lock(self, gene_key: str) -> None:
        lock_path = self._lock_dir / f'{gene_key}.lock'
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def write_snapshot(self, db_path: Path, *, tag: str = '') -> Path:
        db_path = Path(db_path)
        stamp = int(time.time())
        suffix = f'_{tag}' if tag else ''
        dest = self._snap_dir / f'intelligence_{stamp}{suffix}.sqlite'
        shutil.copy2(db_path, dest)
        meta = {
            'version': LCI_STORE_VERSION,
            'created_at': stamp,
            'worker_id': self.worker_id,
            'num_shards': self.num_shards,
            'path': str(dest),
        }
        (self._snap_dir / 'latest.json').write_text(
            json.dumps(meta, indent=2),
            encoding='utf-8',
        )
        return dest

    def import_prior_snapshot(self, snapshot_path: Path, db_path: Path) -> bool:
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.is_file():
            return False
        db_path = Path(db_path)
        if db_path.is_file():
            backup = db_path.with_suffix('.sqlite.bak')
            shutil.copy2(db_path, backup)
        shutil.copy2(snapshot_path, db_path)
        return True

    def load_latest_snapshot_meta(self) -> dict[str, Any] | None:
        meta_path = self._snap_dir / 'latest.json'
        if not meta_path.is_file():
            return None
        try:
            return json.loads(meta_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return None

    def stats(self) -> dict[str, Any]:
        locks = len(list(self._lock_dir.glob('*.lock')))
        snaps = len(list(self._snap_dir.glob('*.sqlite')))
        return {
            'num_shards': self.num_shards,
            'worker_id': self.worker_id,
            'active_locks': locks,
            'snapshots': snaps,
            'latest': self.load_latest_snapshot_meta(),
        }
