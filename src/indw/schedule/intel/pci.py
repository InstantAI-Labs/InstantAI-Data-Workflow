from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indw.config.defaults import PCI_ENABLED, PCI_OBSERVE_ONLY
from indw.schedule.config.resolve import env_str
from indw.schedule.config.resolve import env_flag as _env_flag

_PCI_PUNCT = frozenset(',.:;!?()[]{}|/-_')
_PCI_CODE_HINTS = ('def ', 'class ', 'import ', '{', '};', '=>')
_CODE_MARKERS = ('def ', 'class ', 'import ', '```', 'function ', 'const ')


def _hash(parts: tuple[str, ...], *, size: int = 12) -> str:
    payload = '\x1f'.join(parts).encode('utf-8', 'surrogatepass')
    return hashlib.blake2b(payload, digest_size=size).hexdigest()


@dataclass(frozen=True)
class FingerprintScanMetrics:
    entropy: float
    line_count: int
    code_hits: int
    section_count: int
    uniq_section_lens: int
    word_count: int
    semantic: str = ''


@dataclass(frozen=True)
class FingerprintBundle:
    structural: str
    wrapper: str
    section_shape: str
    line_shape: str
    chars: int
    lines: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'structural': self.structural,
            'wrapper': self.wrapper,
            'section_shape': self.section_shape,
            'line_shape': self.line_shape,
            'chars': self.chars,
            'lines': self.lines,
        }


def fingerprint_from_raw(raw: dict[str, Any] | None) -> FingerprintBundle | None:
    if not isinstance(raw, dict):
        return None
    return FingerprintBundle(
        structural=str(raw.get('structural') or ''),
        wrapper=str(raw.get('wrapper') or ''),
        section_shape=str(raw.get('section_shape') or ''),
        line_shape=str(raw.get('line_shape') or ''),
        chars=int(raw.get('chars') or 0),
        lines=int(raw.get('lines') or 0),
    )


def fingerprint_from_line(line: dict[str, Any]) -> FingerprintBundle | None:
    intel_raw = line.get('acim_intel')
    if isinstance(intel_raw, dict):
        fp = fingerprint_from_raw(intel_raw.get('fp'))
        if fp is not None:
            return fp
    return fingerprint_from_raw(line.get('pci_fp'))


def _entropy_norm(char_counts: Counter[str], n: int) -> float:
    if n <= 0:
        return 0.0
    ent = -sum((c / n) * math.log2(c / n) for c in char_counts.values())
    max_ent = math.log2(max(1, min(256, len(char_counts))))
    if max_ent <= 0:
        return 0.0
    return min(1.0, ent / max_ent)


def build_fingerprint_bundle_detail(
    text: str,
    *,
    raw: Any | None = None,
) -> tuple[FingerprintBundle, FingerprintScanMetrics]:
    blob = text or ''
    n = len(blob)
    upper = alpha = digit = punct = 0
    word_count = 0
    in_word = False
    char_counts: Counter[str] = Counter()
    for c in blob:
        char_counts[c] += 1
        oc = ord(c)
        if 48 <= oc <= 57:
            digit += 1
        if c in _PCI_PUNCT:
            punct += 1
        if 65 <= oc <= 90:
            alpha += 1
            upper += 1
        elif 97 <= oc <= 122:
            alpha += 1
        if c.isspace():
            if in_word:
                word_count += 1
                in_word = False
        else:
            in_word = True
    if in_word:
        word_count += 1

    if raw is not None and getattr(raw, 'lines', None):
        lines = [ln.strip() for ln in raw.lines if ln and ln.strip()]
        if getattr(raw, 'word_count', 0):
            word_count = int(raw.word_count)
    else:
        lines = []
        for raw_line in blob.splitlines():
            stripped = raw_line.strip()
            if stripped:
                lines.append(stripped)
    code_hits = 0
    for stripped in lines:
        if any(m in stripped for m in _CODE_MARKERS):
            code_hits += 1
    caps_ratio = upper / max(alpha, 1)
    digit_ratio = digit / max(n, 1)
    punct_ratio = punct / max(n, 1)
    sections = [part for part in blob.split('\n\n') if part.strip()]
    section_word_lens = [len(part.split()) for part in sections]
    section_lens = [str(nw) for nw in section_word_lens]
    head = lines[0][:90].lower() if lines else ''
    tail = lines[-1][:90].lower() if lines else ''
    head_probe = blob[:512]
    wrapper_hint = (
        'html' if '<html' in head_probe.casefold()
        else 'forum' if '@' in blob[:300]
        else 'plain'
    )
    line_shape_parts = []
    for ln in lines[:24]:
        tokens = ln.split()
        if not tokens:
            continue
        codey = any(tok in ln for tok in _PCI_CODE_HINTS)
        has_digit = False
        for ch in ln:
            oc = ord(ch)
            if 48 <= oc <= 57:
                has_digit = True
                break
        line_shape_parts.append(
            f'{len(tokens)}:{"c" if codey else "p"}:{int(has_digit)}',
        )
    fp = FingerprintBundle(
        structural=_hash((
            str(word_count),
            str(len(lines)),
            f'{caps_ratio:.4f}',
            f'{digit_ratio:.4f}',
            f'{punct_ratio:.4f}',
            head,
            tail,
        )),
        wrapper=_hash((wrapper_hint, head[:40], tail[:40])),
        section_shape=_hash(tuple(section_lens[:20]) or ('0',)),
        line_shape=_hash(tuple(line_shape_parts) or ('0',)),
        chars=len(blob),
        lines=len(lines),
    )
    sample = blob[:2048]
    words = sample.lower().split()
    vocab = len(set(words))
    q_ratio = sample.count('?') / max(len(sample), 1)
    semantic = _hash((
        fp.line_shape,
        str(vocab),
        f'{q_ratio:.4f}',
        sample[:64].lower(),
        sample[-64:].lower() if len(sample) > 64 else '',
    ))
    scan = FingerprintScanMetrics(
        entropy=_entropy_norm(char_counts, n),
        line_count=len(lines),
        code_hits=code_hits,
        section_count=len(sections),
        uniq_section_lens=len(set(section_word_lens)),
        word_count=word_count,
        semantic=semantic,
    )
    return fp, scan


def build_fingerprint_bundle(text: str) -> FingerprintBundle:
    return build_fingerprint_bundle_detail(text)[0]

def _load_snapshot(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload.get('entries')
    if not isinstance(entries, dict):
        return {}
    out: dict[str, str] = {}
    for key, family in entries.items():
        if isinstance(key, str) and isinstance(family, str):
            out[key] = family
    return out


class PCISession:
    def __init__(self, merge_work: Path) -> None:
        self.merge_work = Path(merge_work)
        self.enabled = _env_flag('INSTANT_PCI_ENABLED', PCI_ENABLED)
        self.observe_only = _env_flag('INSTANT_PCI_OBSERVE_ONLY', PCI_OBSERVE_ONLY)
        self._events_enabled = self.enabled
        if self.enabled:
            from indw.schedule.monitor.obs import pci_events_enabled
            self._events_enabled = pci_events_enabled()
        self.snapshot_id = env_str('INSTANT_PCI_SNAPSHOT_ID', 'local-v0')
        self._lock = threading.Lock()
        self._docs = 0
        self._matched = 0
        self._families: dict[str, int] = {}
        self._events = 0
        self._started = time.perf_counter()
        self._dir = self.merge_work / 'pci'
        self._dir.mkdir(parents=True, exist_ok=True)
        self._snapshot = _load_snapshot(self._dir / 'pci_snapshot.json')
        self._events_fp = (
            (self._dir / 'pci_events.jsonl').open('a', encoding='utf-8')
            if self.enabled and self._events_enabled
            else None
        )

    def observe_preprocessed(self, line: dict[str, Any]) -> None:
        if not self.enabled:
            return
        fp = fingerprint_from_line(line)
        if fp is None:
            text = str(line.get('raw_text') or '')
            if not text:
                return
            fp = build_fingerprint_bundle(text)
        key = f'{fp.structural}:{fp.wrapper}:{fp.section_shape}'
        fam = self._snapshot.get(key, '')
        with self._lock:
            self._docs += 1
            if fam:
                self._matched += 1
                self._families[fam] = self._families.get(fam, 0) + 1
            if self._events_fp is not None:
                self._events_fp.write(json.dumps({
                    'snapshot_id': self.snapshot_id,
                    'observe_only': self.observe_only,
                    'seq': line.get('seq'),
                    'source': line.get('src_name', ''),
                    'kind': line.get('kind', ''),
                    'chars': fp.chars,
                    'fp': fp.to_dict(),
                    'matched_family': fam,
                }, ensure_ascii=False) + '\n')
                self._events += 1
                if self._events % 200 == 0:
                    self._events_fp.flush()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            docs = self._docs
            matched = self._matched
            elapsed = max(time.perf_counter() - self._started, 1e-9)
            return {
                'enabled': self.enabled,
                'observe_only': self.observe_only,
                'snapshot_id': self.snapshot_id,
                'docs_observed': docs,
                'template_matches': matched,
                'template_match_rate': round(matched / docs, 4) if docs else 0.0,
                'events_written': self._events,
                'families': dict(sorted(self._families.items(), key=lambda kv: (-kv[1], kv[0]))[:20]),
                'events_per_sec': round(self._events / elapsed, 3),
            }

    def close(self) -> None:
        if self._events_fp is not None:
            self._events_fp.flush()
            self._events_fp.close()
        if not self.enabled:
            return
        (self._dir / 'pci_run_stats.json').write_text(
            json.dumps(self.stats(), indent=2),
            encoding='utf-8',
        )

