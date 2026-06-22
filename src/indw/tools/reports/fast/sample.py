from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import random
from indw.tools.reports.fast.stats import DocRecord, SampleCounters, count_lines, detect_lang, norm_dedup

def reservoir_sample_lines(
    path: Path,
    k: int,
    seed: int,
    *,
    stratify_key: str | None = 'source',
) -> tuple[int, list[str]]:
    rng = random.Random(seed)
    total = 0
    reservoir: list[str] = []
    strata_reservoirs: dict[str, list[str]] = defaultdict(list)
    strata_counts: Counter = Counter()
    per_stratum = max(k // 8, 200) if stratify_key else 0

    with path.open(encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            total += 1
            if stratify_key and per_stratum:
                try:
                    row = json.loads(line)
                    key = str(row.get(stratify_key) or row.get('meta', {}).get(stratify_key) or 'unknown')
                except json.JSONDecodeError:
                    key = 'unknown'
                strata_counts[key] += 1
                bucket = strata_reservoirs[key]
                if len(bucket) < per_stratum:
                    bucket.append(line)
                else:
                    j = rng.randint(0, strata_counts[key] - 1)
                    if j < per_stratum:
                        bucket[j] = line
            if len(reservoir) < k:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = line

    if stratify_key and strata_reservoirs:
        merged: list[str] = []
        seen: set[str] = set()
        for lines in strata_reservoirs.values():
            for ln in lines:
                h = hash(ln)
                if h not in seen:
                    seen.add(h)
                    merged.append(ln)
        for ln in reservoir:
            h = hash(ln)
            if h not in seen and len(merged) < k:
                seen.add(h)
                merged.append(ln)
        return total, merged[:k]
    return total, reservoir

def parse_sample_lines(lines: list[str]) -> list[DocRecord]:
    docs: list[DocRecord] = []
    for i, raw in enumerate(lines):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        text = row.get('text', '') or ''
        source = str(row.get('source') or row.get('meta', {}).get('source') or 'unknown')
        docs.append(DocRecord(line_no=i, text=text, source=source, char_len=len(text)))
    return docs
