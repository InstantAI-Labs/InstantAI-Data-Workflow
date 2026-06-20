from __future__ import annotations

DOMAIN_PARENTS: dict[str, str] = {
    'documentation': 'technical_writing',
    'technical_writing': 'article',
    'code': 'technical_writing',
    'forum': 'conversational',
    'conversational': 'article',
    'encyclopedia': 'reference',
    'wikipedia': 'encyclopedia',
    'scientific': 'research_article',
    'research_article': 'article',
    'mathematical': 'scientific',
    'educational': 'instructional',
    'instructional': 'article',
    'literary': 'prose',
    'legal': 'formal_document',
    'news': 'article',
    'web_article': 'article',
    'general': 'article',
}


def domain_lineage(domain_id: str, *, max_depth: int = 6) -> tuple[str, ...]:
    out: list[str] = []
    cur = domain_id
    seen: set[str] = set()
    for _ in range(max_depth):
        if not cur or cur in seen:
            break
        seen.add(cur)
        out.append(cur)
        cur = DOMAIN_PARENTS.get(cur, '')
    return tuple(out)


def inherited_confidence(
    *,
    domain_confidence: float,
    parent_confidences: dict[str, float],
    domain_id: str,
    decay_per_hop: float = 0.12,
) -> float:
    if domain_confidence <= 0 and not parent_confidences:
        return 0.0
    total = domain_confidence
    weight = 1.0
    for i, ancestor in enumerate(domain_lineage(domain_id)[1:], start=1):
        pc = parent_confidences.get(ancestor, 0.0)
        if pc <= 0:
            continue
        hop = max(0.0, 1.0 - decay_per_hop * i)
        total += pc * hop
        weight += hop
    return min(1.0, total / max(weight, 1e-9))
