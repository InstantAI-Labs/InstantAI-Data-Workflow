from __future__ import annotations

from typing import Any

from indw.extract.core.context import DocumentExecutionContext, bind_document_context
from indw.schedule.dispatch.alloc import STAGE_INTEL_PREVIEW
from indw.schedule.intel.session import get_acim_session
from indw.schedule.state.context import MergeDocumentContext


def process_acim_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    survivors: list[dict[str, Any]] = []
    for item in batch:
        ctx = MergeDocumentContext.from_survivor_payload(item)
        src_name = str(item['src_name'])
        ctx.mark(STAGE_INTEL_PREVIEW)
        acim_sess = get_acim_session()
        if acim_sess is not None and acim_sess.enabled and ctx.fp is not None:
            acim_intel, acim_route, lci_payload = acim_sess.preview_for_cleaning(
                ctx.text, source=src_name, fp=ctx.fp, scan=ctx.fp_scan,
            )
            ctx.acim_intel = acim_intel
            ctx.acim_route = acim_route
            ctx.lci = lci_payload
            if acim_intel is not None:
                bind_document_context(DocumentExecutionContext(
                    normalized_text=ctx.text,
                    pci_fp=acim_intel.get('fp'),
                    source=src_name,
                    gate_raw=ctx.raw_features,
                ))
                ctx.pci_fp = acim_intel.get('fp')
        payload = ctx.survivor_payload(work_dir=item.get('_work_dir'))
        if ctx.pci_fp is not None:
            payload['pci_fp'] = ctx.pci_fp
        if ctx.acim_intel is not None:
            payload['acim_intel'] = ctx.acim_intel
        if ctx.acim_route is not None:
            payload['acim_route'] = ctx.acim_route
        if ctx.lci is not None:
            payload['lci'] = ctx.lci
        survivors.append(payload)
    return {'survivors': survivors}
