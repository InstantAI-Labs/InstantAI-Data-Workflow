from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SectionAnalysis:
    evidence: Any
    profile: Any
    structural: Any
    bundle: Any
    content_value: Any


def build_section_analysis(text: str) -> SectionAnalysis:
    from indw.extract.core.context import get_document_context
    from indw.extract.structure.analyze import analyze_structure
    from indw.clean.document.value import (
        analyze_content_value,
        compute_structure_profile,
        resolve_analysis_bundle,
    )

    bundle = resolve_analysis_bundle(text)
    ev = bundle.evidence(text)
    dctx = get_document_context()
    if dctx is not None:
        structural = dctx.structure_analysis(text, lambda: analyze_structure(text))
        profile = dctx.structure_profile(
            text, lambda: compute_structure_profile(text, evidence=ev, bundle=bundle._bundle),
        )
    else:
        structural = analyze_structure(text)
        profile = compute_structure_profile(text, evidence=ev, bundle=bundle._bundle)
    cv = analyze_content_value(text, bundle=bundle)
    return SectionAnalysis(evidence=ev, profile=profile, structural=structural, bundle=bundle, content_value=cv)
