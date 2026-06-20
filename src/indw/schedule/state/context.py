from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indw.schedule.intel.pci import FingerprintBundle


@dataclass
class MergeDocumentContext:
    seq: int
    src_name: str
    line_no: int
    row: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    text: str = ''
    meaningful_chars: int = 0
    kind: str = 'processed'
    fp: FingerprintBundle | None = None
    fp_scan: Any = None
    pci_fp: dict[str, Any] | None = None
    language_assessment: Any = None
    acim_intel: dict[str, Any] | None = None
    acim_route: dict[str, Any] | None = None
    lci: dict[str, Any] | None = None
    doc_tier: str = ''
    admission: dict[str, Any] | None = None
    ingest_meta: dict[str, Any] | None = None
    raw_features: Any = None
    doc_content_hash: str = ''
    cleaning_rejects: list[tuple[str, int]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    stage_trace: list[str] = field(default_factory=list)

    def mark(self, stage: str) -> None:
        if not self.stage_trace or self.stage_trace[-1] != stage:
            self.stage_trace.append(stage)

    def reject(self, reason: str, *, text_len: int | None = None) -> None:
        ln = text_len if text_len is not None else len(self.text)
        self.cleaning_rejects.append((reason, ln))

    def survivor_payload(self, *, work_dir: str | None = None) -> dict[str, Any]:
        text = self.text
        text_store_key = None
        store_key = self.doc_content_hash or str(self.seq)
        if work_dir and text:
            from indw.schedule.state.survivor import externalize_survivor_text
            text, text_store_key = externalize_survivor_text(
                work_dir, doc_key=store_key, text=text,
            )
        payload: dict[str, Any] = {
            'seq': self.seq,
            'src_name': self.src_name,
            'line_no': self.line_no,
            'provenance': self.provenance,
            'text': text,
            'kind': self.kind,
            'meaningful_chars': self.meaningful_chars,
            'language_assessment': self.language_assessment,
            'stage_trace': list(self.stage_trace),
            'doc_tier': self.doc_tier,
            'admission': self.admission,
            'ingest_meta': self.ingest_meta,
            'doc_content_hash': self.doc_content_hash,
        }
        if text_store_key:
            payload['text_store_key'] = text_store_key
            payload['_work_dir'] = work_dir
        if self.raw_features is not None:
            payload['raw_features'] = self.raw_features
        if self.row is not None:
            row = dict(self.row)
            if row.get('text') == self.text:
                row.pop('text', None)
            if row:
                payload['row'] = row
        return payload

    @classmethod
    def from_survivor_payload(cls, payload: dict[str, Any]) -> MergeDocumentContext:
        from indw.filter.language.detect import LanguageAssessment
        from indw.schedule.intel.pci import fingerprint_from_raw

        lang_raw = payload.get('language_assessment')
        lang = None
        if isinstance(lang_raw, dict):
            lang = LanguageAssessment(
                primary_language=str(lang_raw.get('primary_language', 'en')),
                confidence=float(lang_raw.get('confidence', 0.0)),
                languages=dict(lang_raw.get('languages') or {}),
                mixed_language=bool(lang_raw.get('mixed_language', False)),
                fragmentation=float(lang_raw.get('fragmentation', 0.0)),
                reject_reason=lang_raw.get('reject_reason'),
                should_reject=bool(lang_raw.get('should_reject', False)),
            )
        elif lang_raw is not None:
            lang = lang_raw
        text = str(payload.get('text') or '')
        store_key = payload.get('text_store_key')
        if store_key and not text.strip():
            from indw.schedule.state.survivor import resolve_survivor_text
            text = resolve_survivor_text(payload)
        row = payload.get('row')
        if row is None and text:
            row = {'text': text}
        elif isinstance(row, dict) and text and 'text' not in row:
            row = {**row, 'text': text}
        meaningful = int(payload.get('meaningful_chars') or 0)
        if not meaningful and text:
            meaningful = len(text)
        return cls(
            seq=int(payload['seq']),
            src_name=str(payload['src_name']),
            line_no=int(payload['line_no']),
            row=row,
            provenance=payload.get('provenance'),
            text=text,
            meaningful_chars=meaningful,
            kind=str(payload.get('kind', 'processed')),
            language_assessment=lang,
            stage_trace=list(payload.get('stage_trace') or []),
            pci_fp=payload.get('pci_fp'),
            fp=fingerprint_from_raw(payload.get('pci_fp') or payload.get('fp')),
            doc_tier=str(payload.get('doc_tier') or ''),
            admission=payload.get('admission'),
            ingest_meta=payload.get('ingest_meta'),
            doc_content_hash=str(payload.get('doc_content_hash') or ''),
            raw_features=payload.get('raw_features'),
        )
