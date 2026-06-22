from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from indw.store.export.packing.config import PackingConfig

_IGNORE_LABEL = -100


@dataclass(slots=True)
class Doc:
    tokens: np.ndarray


@dataclass(slots=True)
class PackRow:
    input_ids: list[int]
    labels: list[int]
    segment_lengths: list[int]
    padding_count: int = 0
    documents_packed: int = 0


def pack_used(p: PackRow) -> int:
    return len(p.input_ids) - p.padding_count


def pack_density(p: PackRow) -> float:
    sl = len(p.input_ids)
    return pack_used(p) / max(sl, 1)


@dataclass(slots=True)
class PackStats:
    seqs: int = 0
    docs: int = 0
    used: int = 0
    slots: int = 0
    pad: int = 0
    segs: int = 0

    def eff(self) -> float:
        return self.used / max(self.slots, 1)

    def avg_segs(self) -> float:
        return self.segs / max(self.seqs, 1)

    def record(self, packed: PackRow, seq_len: int) -> None:
        self.seqs += 1
        self.docs += packed.documents_packed
        self.used += pack_used(packed)
        self.slots += seq_len
        self.pad += packed.padding_count
        self.segs += len(packed.segment_lengths)

    def record_stride_window(self, seq_len: int) -> None:
        self.seqs += 1
        self.docs += 1
        self.used += seq_len
        self.slots += seq_len
        self.segs += 1


Document = Doc
PackedSequence = PackRow
PackingStats = PackStats


def _as_int64_tokens(tokens: np.ndarray | list[int]) -> np.ndarray:
    if isinstance(tokens, np.ndarray):
        arr = tokens
    else:
        arr = np.asarray(tokens, dtype=np.int64)
    if arr.dtype != np.int64:
        arr = arr.astype(np.int64, copy=False)
    return arr.ravel()


def _doc_tokens_list(doc: Doc) -> list[int]:
    t = doc.tokens
    if t.dtype == np.int64 and t.flags['C_CONTIGUOUS']:
        return t.tolist()
    return [int(x) for x in t]


class DynamicSequencePacker:

    def __init__(self, seq_len: int, cfg: PackingConfig):
        self.seq_len = seq_len
        self.cfg = cfg
        self._doc_queue: deque[Doc] = deque()
        self._stats = PackStats()
        self._carry: Optional[np.ndarray] = None
        if cfg.mode == 'multi_sample' and cfg.delimiter_token is None:
            raise ValueError(
                'multi_sample packing requires delimiter_token (set pack_delimiter: eos in data config)'
            )

    def feed_tokens(self, tokens: np.ndarray | list[int]) -> None:
        arr = _as_int64_tokens(tokens)
        if self._carry is not None and len(self._carry):
            arr = np.concatenate([self._carry, arr])
            self._carry = None
        min_doc = self.cfg.min_document_tokens
        if len(arr) < min_doc:
            self._carry = arr
            return

        if self.cfg.delimiter_token is not None:
            delim = int(self.cfg.delimiter_token)
            start = 0
            for idx in np.where(arr == delim)[0]:
                doc = arr[start:idx]
                if len(doc) >= min_doc:
                    self._doc_queue.append(Doc(doc))
                start = int(idx) + 1
            if start < len(arr):
                tail = arr[start:]
                if len(tail) >= min_doc:
                    self._doc_queue.append(Doc(tail))
                elif len(tail):
                    self._carry = tail
        else:
            self._doc_queue.append(Doc(arr))

    def _take_document(self) -> Optional[Doc]:
        return self._doc_queue.popleft() if self._doc_queue else None

    def pack_one(self) -> Optional[PackRow]:
        need = self.seq_len + 1
        packed_tokens: list[int] = []
        segment_lengths: list[int] = []
        documents_packed = 0
        overhead = 1 if self.cfg.delimiter_token is not None else 0

        while len(packed_tokens) < need:
            doc = self._take_document()
            if doc is None:
                break

            remaining = need - len(packed_tokens)
            doc_tokens = _doc_tokens_list(doc)
            doc_len = len(doc_tokens)
            sep = overhead if packed_tokens and self.cfg.delimiter_token is not None else 0
            required = doc_len + sep

            if required > remaining:
                if not packed_tokens:
                    fit = need - sep
                    chunk = doc_tokens[:fit]
                    if sep and self.cfg.delimiter_token is not None:
                        packed_tokens.extend(chunk + [self.cfg.delimiter_token])
                    else:
                        packed_tokens.extend(chunk)
                    segment_lengths.append(len(chunk))
                    documents_packed = 1
                    remainder = doc_tokens[len(chunk):]
                    if remainder:
                        self._doc_queue.appendleft(Doc(np.asarray(remainder, dtype=np.int64)))
                else:
                    self._doc_queue.appendleft(doc)
                break

            if sep and self.cfg.delimiter_token is not None:
                packed_tokens.append(self.cfg.delimiter_token)
            packed_tokens.extend(doc_tokens)
            segment_lengths.append(doc_len + (sep if sep else 0))
            documents_packed += 1

            if len(segment_lengths) >= self.cfg.max_segments_per_seq:
                break

        if len(packed_tokens) < need:
            if not packed_tokens:
                return None
            if self.cfg.allow_padding_tail:
                pad = need - len(packed_tokens)
                packed_tokens.extend([self.cfg.pad_token_id] * pad)
                padding_count = pad
            else:
                while len(packed_tokens) < need and self._doc_queue:
                    doc = self._take_document()
                    if doc:
                        packed_tokens.extend(
                            _doc_tokens_list(doc)[:need - len(packed_tokens)]
                        )
                padding_count = max(0, need - len(packed_tokens))
                if padding_count:
                    return None
                padding_count = 0
        else:
            padding_count = 0

        packed_tokens = packed_tokens[:need]
        input_ids = packed_tokens[:-1]
        labels = packed_tokens[1:]
        if padding_count > 0:
            for i in range(padding_count):
                labels[-(i + 1)] = _IGNORE_LABEL
        result = PackRow(
            input_ids=input_ids,
            labels=labels,
            segment_lengths=self._normalize_segment_lengths(
                segment_lengths,
                len(input_ids),
            ),
            padding_count=padding_count,
            documents_packed=documents_packed,
        )
        self._stats.record(result, self.seq_len)
        return result

    def _normalize_segment_lengths(self, lengths: list[int], total: int) -> list[int]:
        if not lengths:
            return [total]
        s = sum(lengths)
        if s != total and lengths:
            lengths[-1] += total - s
        return [l for l in lengths if l > 0]

    def pack_many(self, count: int) -> Iterator[PackRow]:
        for _ in range(count):
            p = self.pack_one()
            if p is None:
                break
            yield p

    @property
    def stats(self) -> PackStats:
        return self._stats
