from __future__ import annotations

from indw.filter.content.code import analyze_code_dump, strip_dominant_code
from indw.filter.refine.truncation import analyze_truncation, repair_truncation
from indw.filter.spec.pipeline import PipelinePolicy
from indw.filter.spec.document import CorpusDocument

def repair_structure(doc: CorpusDocument, policy: PipelinePolicy) -> CorpusDocument:
    cfg = policy.structural_repair
    th = policy.structural_thresholds
    if not doc.text:
        return doc.with_stage('structural_repair')
    working = doc.text
    flags = list(doc.flags)

    if cfg.get('repair_truncation'):
        trunc = analyze_truncation(working)
        if trunc.probability > th.trunc_repair_probability and trunc.repairable and cfg.get('repair_truncation'):
            repaired = repair_truncation(working)
            if repaired and repaired != working:
                working = repaired
                flags.append('truncation_repaired')
        if cfg.get('remove_heavy_truncation') and trunc.probability > th.trunc_remove_probability:
            return doc.with_flags(tuple(flags)).with_stage(
                'structural_repair',
                text='',
                text_modified=True,
            )

    if cfg.get('remove_code_dumps'):
        dump = analyze_code_dump(working)
        from indw.filter.content.code import analyze_code
        code = analyze_code(working)
        educational_code = (
            code is not None
            and (
                code.educational_score >= th.educational_code_score
                or (code.syntax_balance >= th.educational_syntax_balance and 'procedure' in working.lower())
            )
        )
        if dump.probability > th.code_dump_probability and dump.should_remove and not educational_code:
            flags.append('code_dump')
            return doc.with_flags(tuple(flags)).with_stage(
                'structural_repair',
                text='',
                text_modified=True,
            )

    if cfg.get('strip_dominant_code'):
        stripped, changed_code = strip_dominant_code(working)
        if changed_code:
            working = stripped
            flags.append('code_stripped')

    min_chars = int(cfg.get('min_chars_after_repair', policy.structural_repair.get('min_chars_after_repair', 60)))
    if len(working.strip()) < min_chars:
        flags.append('too_short_after_repair')

    return doc.with_text(working, modified=working != doc.text).with_flags(tuple(flags)).with_stage(
        'structural_repair',
    )
