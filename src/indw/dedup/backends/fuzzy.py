from __future__ import annotations

from indw.dedup.fuzzy import StreamingFuzzyDedup


def create_fuzzy_dedup(**kwargs) -> StreamingFuzzyDedup:
    return StreamingFuzzyDedup(**kwargs)
