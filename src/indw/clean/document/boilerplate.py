from __future__ import annotations

import re

from indw.clean.artifact.registry import line_is_artifact
from indw.clean.document.patterns import _UI_LINE
from indw.clean.document.stats import StageStats

_INLINE_BOILER = re.compile(
    r'(?i)\b(?:'
    r'related\s+articles?|trending\s+now|read\s+more|you\s+may\s+also\s+like|'
    r'recommended\s+for\s+you|popular\s+posts?|sponsored\s+content|'
    r'subscribe\s+to\s+(?:our\s+)?newsletter|sign\s+up\s+for\s+free|'
    r'all\s+rights\s+reserved|copyright\s*©|terms\s+of\s+(?:use|service)|'
    r'privacy\s+policy|cookie\s+(?:policy|settings)|'
    r'article\s+free\s+pass|free\s+pass|main\s+page|special:\s*\w+|'
    r'affiliate\s+(?:link|disclosure)|paid\s+partnership|commission\s+may\s+be\s+earned|'
    r'add\s+to\s+cart|shopping\s+cart|checkout\s+now|buy\s+now|'
    r'session[_\s-]?id\s*[:=]\s*[\w-]+|utm_[a-z]+=|gtag\(|google-analytics|'
    r'click\s+here|learn\s+more|see\s+more|view\s+all|continue\s+reading'
    r')\b'
)

_LINE_BOILER = re.compile(
    r'(?i)^\s*(?:'
    r'related\s*:?|trending\s+now|read\s+more|previous\s+page|next\s+page|'
    r'share\s+on|follow\s+us|join\s+our|register\s+now|log\s*in|sign\s*up|'
    r'advertisement|sponsored|promoted|editor\'?s\s+pick|'
    r'also\s+read|more\s+from|popular\s+stories|top\s+stories|'
    r'email\s+signup|mailing\s+list|add\s+to\s+cart|view\s+cart|'
    r'affiliate\s+disclosure|product\s+recommendations?|'
    r'customers?\s+also\s+bought|frequently\s+bought\s+together'
    r')\b.*$'
)

_SHORT_CTA = re.compile(r'(?i)^\s*(?:click\s+here|learn\s+more|see\s+more|view\s+all)\s*[!.]?\s*$')
_BRAND_REPEAT = re.compile(r'(?i)^\s*[\w\s]{2,30}\s*\|\s*[\w\s]{2,30}\s*\|\s*[\w\s]{2,30}\s*$')

def remove_boilerplate(text: str, *, stats: StageStats | None = None) -> str:
    if not text:
        return text
    removed = 0
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append('')
            continue
        drop = (
            line_is_artifact(stripped)
            or _UI_LINE.search(stripped)
            or _LINE_BOILER.search(stripped)
            or _SHORT_CTA.search(stripped)
            or (_BRAND_REPEAT.match(stripped) and len(stripped) < 100)
        )
        if drop:
            removed += 1
            continue
        cleaned = _INLINE_BOILER.sub(' ', stripped)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        if cleaned:
            kept.append(cleaned)
        else:
            removed += 1
    out = '\n'.join(kept)
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    if stats is not None:
        stats.lines_removed += removed
        stats.in_docs += 1
        stats.out_docs += 1 if out else 0
    return out
