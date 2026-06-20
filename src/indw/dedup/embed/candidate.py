from __future__ import annotations

import re

from indw.dedup.embed.config import EmbeddingDedupConfig
from indw.dedup.embed.contracts import DedupDocumentMeta
from indw.dedup.semantic import _simhash64, _token_set

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)

def _length_bucket(length: int, bucket: int) -> int:
    if bucket <= 0:
        return 0
    return max(0, int(length) // bucket)

def _token_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)

def build_document_meta(
    text: str,
    *,
    language: str = 'unknown',
    domain: str = 'unknown',
    quality: float = 0.0,
    knowledge: float = 0.0,
    utility: float = 0.0,
    toxicity: float = 0.0,
    artifact: float = 0.0,
) -> DedupDocumentMeta:
    toks = _token_set(text)
    return DedupDocumentMeta(
        text=text,
        language=(language or 'unknown').lower()[:16],
        domain=(domain or 'unknown').lower()[:32],
        quality=float(quality),
        knowledge=float(knowledge),
        utility=float(utility),
        toxicity=float(toxicity),
        artifact=float(artifact),
        length=len(text),
        token_sig=frozenset(toks),
    )

class BlockingCandidateGenerator:
    def __init__(self, cfg: EmbeddingDedupConfig):
        self.cfg = cfg
        mask = (1 << cfg.simhash_bucket_bits) - 1
        self._simhash_mask = mask

    def block_key(self, meta: DedupDocumentMeta) -> tuple:
        sim_bucket = (_simhash64(meta.text) >> (64 - self.cfg.simhash_bucket_bits)) & self._simhash_mask
        parts: list[Any] = []
        if self.cfg.same_language_only:
            parts.append(meta.language)
        if self.cfg.block_by_domain:
            parts.append(meta.domain)
        if self.cfg.block_by_length:
            parts.append(_length_bucket(meta.length, self.cfg.length_bucket_chars))
        parts.append(sim_bucket)
        return tuple(parts)

    def filter_candidates(
        self,
        meta: DedupDocumentMeta,
        entries: list[tuple[int, DedupDocumentMeta]],
    ) -> list[tuple[int, DedupDocumentMeta]]:
        out: list[tuple[int, DedupDocumentMeta]] = []
        for cid, cand in entries[: self.cfg.max_candidates]:
            if self.cfg.same_language_only and meta.language != cand.language:
                continue
            if self.cfg.block_by_domain and meta.domain != cand.domain:
                continue
            if self.cfg.block_by_length:
                if _length_bucket(meta.length, self.cfg.length_bucket_chars) != _length_bucket(
                    cand.length, self.cfg.length_bucket_chars
                ):
                    continue
            if self.cfg.token_overlap_min > 0:
                if _token_overlap(set(meta.token_sig), set(cand.token_sig)) < self.cfg.token_overlap_min:
                    continue
            out.append((cid, cand))
        return out
