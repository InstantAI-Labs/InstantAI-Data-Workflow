from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from indw.store.io.json_codec import dumps, loads


def sorted_output_hash(path: Path) -> str:
    texts = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        texts.append(str(loads(line).get('text') or ''))
    return hashlib.sha256('\n'.join(sorted(texts)).encode()).hexdigest()


def load_work_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = loads(path.read_bytes())
    return raw if isinstance(raw, dict) else {}
