from __future__ import annotations

from typing import Any

from indw.extract.core.context import DocumentExecutionContext, bind_document_context
from indw.schedule.dispatch.alloc import STAGE_INTERMEDIATE
from indw.schedule.intel.pci import build_fingerprint_bundle_detail
from indw.schedule.monitor.doc import monitor_begin_doc
from indw.schedule.state.context import MergeDocumentContext


def process_pci_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    survivors: list[dict[str, Any]] = []
    for item in batch:
        ctx = MergeDocumentContext.from_survivor_payload(item)
        src_name = str(item['src_name'])
        ctx.mark(STAGE_INTERMEDIATE)
        fp, scan = build_fingerprint_bundle_detail(ctx.text, raw=ctx.raw_features)
        ctx.fp = fp
        ctx.fp_scan = scan
        ctx.pci_fp = fp.to_dict()
        doc_id = f'{src_name}:{ctx.line_no}'
        monitor_begin_doc(
            seq=ctx.seq,
            src_name=src_name,
            line_no=ctx.line_no,
            chars=len(ctx.text),
            fingerprint=fp.structural,
            doc_id=doc_id,
            words=scan.word_count,
        )
        bind_document_context(DocumentExecutionContext(
            normalized_text=ctx.text,
            pci_fp=ctx.pci_fp,
            source=src_name,
            gate_raw=ctx.raw_features,
        ))
        payload = ctx.survivor_payload(work_dir=item.get('_work_dir'))
        payload['pci_fp'] = ctx.pci_fp
        survivors.append(payload)
    return {'survivors': survivors}
