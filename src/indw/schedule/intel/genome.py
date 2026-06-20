from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from indw.schedule.intel.pci import _hash


class GeneKind(str, Enum):
    PROSE = 'prose'
    PUBLICATION_SCAFFOLD = 'publication_scaffold'
    FORUM_QUESTION = 'forum_question'
    FORUM_ANSWER = 'forum_answer'
    EDUCATIONAL = 'educational'
    CODE = 'code'
    MATHEMATICAL = 'mathematical'
    NAVIGATION = 'navigation'
    DOCUMENTATION = 'documentation'
    SCIENTIFIC = 'scientific'
    LEGAL = 'legal'
    NEWS = 'news'
    HTML_LAYOUT = 'html_layout'
    UNKNOWN = 'unknown'


_GENE_PATTERNS: tuple[tuple[GeneKind, re.Pattern[str]], ...] = (
    (GeneKind.CODE, re.compile(r'```|^\s*(def |class |import |from |function )', re.M)),
    (GeneKind.FORUM_QUESTION, re.compile(r'^\s*(Q:|Question:|\?)', re.M | re.I)),
    (GeneKind.FORUM_ANSWER, re.compile(r'^\s*(A:|Answer:)', re.M | re.I)),
    (GeneKind.MATHEMATICAL, re.compile(r'\$[^$]+\$|\\begin\{|\\frac\{|∑|∫|∀|∃')),
    (GeneKind.HTML_LAYOUT, re.compile(r'<(?:html|div|table|nav|header|footer)\b', re.I)),
    (GeneKind.PUBLICATION_SCAFFOLD, re.compile(
        r'(published|copyright|all rights reserved|subscribe|newsletter|by\s+\w+\s+\w+)',
        re.I,
    )),
    (GeneKind.NAVIGATION, re.compile(r'(home\s*\||\bmenu\b|breadcrumb|table of contents)', re.I)),
    (GeneKind.LEGAL, re.compile(r'(whereas|hereby|pursuant to|section\s+\d+\.\d+)', re.I)),
    (GeneKind.NEWS, re.compile(r'(breaking:|associated press|reuters|for immediate release)', re.I)),
    (GeneKind.SCIENTIFIC, re.compile(r'(abstract|introduction|methodology|references|doi:)', re.I)),
    (GeneKind.EDUCATIONAL, re.compile(r'(learning objective|exercise|homework|chapter\s+\d+)', re.I)),
    (GeneKind.DOCUMENTATION, re.compile(r'(api reference|parameters:|returns:|usage:)', re.I)),
)


@dataclass(frozen=True)
class StructuralGene:
    kind: GeneKind
    gene_key: str
    span_index: int
    word_count: int
    line_count: int
    shape_sig: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind.value,
            'gene_key': self.gene_key,
            'span_index': self.span_index,
            'word_count': self.word_count,
            'line_count': self.line_count,
            'shape_sig': self.shape_sig,
        }


@dataclass(frozen=True)
class GenomeProfile:
    genes: tuple[StructuralGene, ...]
    domain_id: str
    genome_key: str
    gene_kinds: tuple[str, ...]
    novel_gene_keys: tuple[str, ...] = ()
    known_gene_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            'genes': [g.to_dict() for g in self.genes],
            'domain_id': self.domain_id,
            'genome_key': self.genome_key,
            'gene_kinds': list(self.gene_kinds),
            'novel_gene_keys': list(self.novel_gene_keys),
            'known_gene_keys': list(self.known_gene_keys),
            'gene_count': len(self.genes),
            'novel_gene_ratio': round(
                len(self.novel_gene_keys) / max(len(self.genes), 1), 4,
            ),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> GenomeProfile | None:
        genes_raw = raw.get('genes')
        if not isinstance(genes_raw, list):
            return None
        genes: list[StructuralGene] = []
        for item in genes_raw:
            if not isinstance(item, dict):
                continue
            kind_raw = str(item.get('kind') or GeneKind.UNKNOWN.value)
            try:
                kind = GeneKind(kind_raw)
            except ValueError:
                kind = GeneKind.UNKNOWN
            genes.append(StructuralGene(
                kind=kind,
                gene_key=str(item.get('gene_key') or ''),
                span_index=int(item.get('span_index', 0)),
                word_count=int(item.get('word_count', 0)),
                line_count=int(item.get('line_count', 0)),
                shape_sig=str(item.get('shape_sig') or ''),
            ))
        genome_key = str(raw.get('genome_key') or '')
        if not genome_key and not genes:
            return None
        domain_id = str(raw.get('domain_id') or 'general')
        kinds_raw = raw.get('gene_kinds')
        if isinstance(kinds_raw, list):
            gene_kinds = tuple(str(k) for k in kinds_raw)
        else:
            gene_kinds = tuple(dict.fromkeys(g.kind.value for g in genes))
        novel_raw = raw.get('novel_gene_keys')
        known_raw = raw.get('known_gene_keys')
        return GenomeProfile(
            genes=tuple(genes),
            domain_id=domain_id,
            genome_key=genome_key or _hash(tuple(g.gene_key for g in genes[:24]) or ('0',)),
            gene_kinds=gene_kinds,
            novel_gene_keys=tuple(str(k) for k in novel_raw) if isinstance(novel_raw, list) else (),
            known_gene_keys=tuple(str(k) for k in known_raw) if isinstance(known_raw, list) else (),
        )


def _classify_span(text: str) -> GeneKind:
    sample = text[:1200]
    for kind, pat in _GENE_PATTERNS:
        if pat.search(sample):
            return kind
    words = sample.split()
    if len(words) >= 12 and sample.count('.') >= 2:
        return GeneKind.PROSE
    return GeneKind.UNKNOWN


def _shape_sig(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:12]
    parts = [f'{len(ln.split())}:{ln[:1]}' for ln in lines]
    return _hash(tuple(parts) or ('0',))


def _gene_key(kind: GeneKind, shape: str, words: int) -> str:
    return _hash((kind.value, shape, str(min(words, 9999))))


def extract_genes(text: str, *, domain_id: str = 'general') -> GenomeProfile:
    spans = [s.strip() for s in re.split(r'\n{2,}', text) if s.strip()]
    if not spans:
        spans = [text.strip()] if text.strip() else []
    genes: list[StructuralGene] = []
    kinds: list[str] = []
    keys: list[str] = []
    for i, span in enumerate(spans[:48]):
        kind = _classify_span(span)
        shape = _shape_sig(span)
        words = len(span.split())
        gkey = _gene_key(kind, shape, words)
        genes.append(StructuralGene(
            kind=kind,
            gene_key=gkey,
            span_index=i,
            word_count=words,
            line_count=len([ln for ln in span.splitlines() if ln.strip()]),
            shape_sig=shape,
        ))
        kinds.append(kind.value)
        keys.append(gkey)
    genome_key = _hash(tuple(keys[:24]) or ('0',))
    return GenomeProfile(
        genes=tuple(genes),
        domain_id=domain_id,
        genome_key=genome_key,
        gene_kinds=tuple(dict.fromkeys(kinds)),
    )


def resolve_source_domain(source: str) -> str:
    src = (source or '').lower()
    if 'wikipedia' in src:
        return 'encyclopedia'
    if 'stack-exchange' in src or 'forum' in src:
        return 'forum'
    if 'starcoder' in src or src.endswith('-python') or src.endswith('-java'):
        return 'code'
    if 'open-web-math' in src or 'math' in src:
        return 'mathematical'
    if 'gutenberg' in src:
        return 'literary'
    if 'fineweb-edu' in src or 'cosmopedia' in src or 'orca' in src:
        return 'educational'
    if 'fineweb' in src or 'dclm' in src:
        return 'web_article'
    return 'general'


def extract_genes_from_line(
    line: dict[str, Any],
    *,
    source: str = '',
) -> tuple[GenomeProfile | None, GenomeProfile | None]:
    raw_text = str(line.get('raw_text') or '')
    src = source or str(line.get('src_name') or '')
    domain = resolve_source_domain(src)
    raw: GenomeProfile | None = None
    lci_raw = line.get('lci')
    if isinstance(lci_raw, dict) and raw_text.strip():
        raw = GenomeProfile.from_dict(lci_raw)
    if raw is None and raw_text.strip():
        raw = extract_genes(raw_text, domain_id=domain)
    parts: list[str] = []
    for chunk in line.get('chunks') or []:
        if not isinstance(chunk, dict):
            continue
        ct = str(chunk.get('chunk_text') or '').strip()
        if ct:
            parts.append(ct)
    if not parts:
        return raw, None
    cleaned = extract_genes('\n\n'.join(parts), domain_id=domain)
    return raw, cleaned

