from __future__ import annotations

import hashlib
import unicodedata


def _ctrl_skip(ord_c: int) -> bool:
    return 0 <= ord_c <= 8 or ord_c in (11, 12) or 14 <= ord_c <= 31


def normalize_for_dedup(text: str) -> str:
    if not text:
        return ''
    text = unicodedata.normalize('NFC', text)
    out: list[str] = []
    in_ws = False
    for c in text:
        oc = ord(c)
        if _ctrl_skip(oc):
            continue
        if c.isspace():
            if out and not in_ws:
                out.append(' ')
                in_ws = True
            continue
        in_ws = False
        out.append(c.lower())
    return ''.join(out).strip()


def content_hash(text: str, *, normalized: bool = False) -> str:
    if normalized:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    if not text:
        return hashlib.sha256(b'').hexdigest()
    text = unicodedata.normalize('NFC', text)
    h = hashlib.sha256()
    started = False
    pending_ws = False
    for c in text:
        oc = ord(c)
        if _ctrl_skip(oc):
            continue
        if c.isspace():
            if started:
                pending_ws = True
            continue
        if pending_ws:
            h.update(b' ')
            pending_ws = False
        started = True
        h.update(c.lower().encode('utf-8'))
    return h.hexdigest()


def stable_token_hash(token: str) -> int:
    digest = hashlib.sha256(token.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big')
