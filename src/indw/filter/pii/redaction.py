from __future__ import annotations

from indw.filter.pii.entities import ExtractedEntity
from indw.filter.pii.secrets import SecretSpan

def redact_text(
    text: str,
    *,
    entities: list[ExtractedEntity],
    secrets: list[SecretSpan],
) -> str:
    spans: list[tuple[int, int, str]] = []
    for ent in entities:
        spans.append((ent.start, ent.end, f'<{ent.type}>'))
    for sec in secrets:
        spans.append((sec.start, sec.end, '<SECRET>'))
    if not spans:
        return text
    spans.sort(key=lambda x: x[0], reverse=True)
    out = text
    for start, end, label in spans:
        if start < 0 or end > len(out) or start >= end:
            continue
        out = out[:start] + label + out[end:]
    return out
