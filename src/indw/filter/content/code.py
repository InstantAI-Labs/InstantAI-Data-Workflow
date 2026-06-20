from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from indw.clean.document.patterns import _CODE_FENCE, _WORD

_COMMENT = re.compile(r'(#|//|/\*).*')
_GENERATED = re.compile(
    r'(?i)\b(?:auto[\s-]?generated|generated\s+by|do\s+not\s+edit|'
    r'machine[\s-]?generated|@generated|file\s+was\s+generated)\b'
)

_CODE_LINE = re.compile(
    r'^\s*(?:'
    r'(?:def|class|import|from|package|#include|#define|using|namespace|'
    r'public|private|protected|static|void|int|float|double|bool|'
    r'function|const|let|var|return|if|else|for|while|switch|'
    r'procedure|pragma|subtype|type|record|begin|end|with|use)\b|'
    r'[@#][!]?include|'
    r'@echo\s+off|'
    r'(?:set|cd|cmake|make|npm|pip|gradle|mvn)\b|'
    r'[{}\[\]();]|'
    r'%\w+%|'
    r'0x[0-9A-Fa-f]+|'
    r'\w+\s*=\s*[^=]|'
    r':=\s*'
    r')',
    re.M | re.I,
)

_FILE_EXT_LINE = re.compile(
    r'(?im)^(?:.*/)?[\w.-]+\.(?:'
    r'java|cs|cpp|cc|c|h|hpp|py|js|ts|go|rs|rb|php|sh|bash|zsh|'
    r'cmd|ps1|agda|ml|hs|lua|sql|yaml|yml|toml|json|xml|gradle|cmake'
    r')\s*$'
)

_BUILD_SCRIPT = re.compile(
    r'(?im)^(?:'
    r'@echo\s+off|'
    r'#!/(?:bin/)?(?:ba)?sh|'
    r'#!/usr/bin/env\s+\w+|'
    r'FROM\s+\S+|'
    r'RUN\s+|'
    r'COPY\s+|'
    r'WORKDIR\s+|'
    r'ENV\s+|'
    r'cmake_minimum_required|'
    r'project\s*\(|'
    r'all\s*:|'
    r'\.PHONY\s*:|'
    r'npm\s+install|'
    r'pip\s+install'
    r')'
)

_README_ONLY = re.compile(r'(?im)^#+\s*(?:readme|license|changelog|contributing)\s*$')

_VENDOR_SDK = re.compile(
    r'(?i)\b(?:'
    r'STM32|CMSIS|HAL_[A-Z]|NVIC_|RCC_|GPIO[A-Z]?|USART\d|SPI\d|I2C\d|'
    r'GNAT\s+LIBRARY|GNARL|LLVM\s+Ada|freetype_c|eGL\.|UxAS\.|'
    r'System\.Startup|Ada\.Streams|Interfaces\.C\.|'
    r'STM32\.|stm32f4xx_hal|STMicroelectronics|package\s+body\s+STM32'
    r')\b'
)

_COMPILER_INTERNAL = re.compile(
    r'(?i)\b(?:'
    r'compiler\s+internals?|runtime\s+library|standard\s+library|'
    r'auto[\s-]?generated\s+(?:interface|binding|wrapper)|'
    r'bindings?\s+for\s+(?:the\s+)?(?:API|SDK)'
    r')\b'
)

_ADA_PACKAGE = re.compile(r'(?im)^\s*(?:private\s+)?package(?:\s+body)?\s+[\w.]+')
_ADA_TYPE_DUMP = re.compile(
    r'(?i)\btype\s+\w+\s+is\s+(?:new\s+)?(?:record|access|range|mod|array|'
    r'aliased|limited|interface)'
)
_ADA_ROUTINE = re.compile(r'(?i)\b(?:procedure|function)\s+\w+')
_REGISTER_DEF = re.compile(
    r'(?im)^\s*#\s*define\s+(?:'
    r'\w+(?:_REG|_ADDR|_BASE|_MASK)|HAL_\w+|REG_\w+|GPIO\w*|NVIC_\w+'
    r')\b'
)
_HEX_CONST_LINE = re.compile(r'(?im)^\s*(?:0x[0-9A-Fa-f]{3,}|#[ \t]*define\s+\w+\s+0x)')
_ENUM_MAPPING = re.compile(r'(?im)^\s*\w+\s*=>\s*\w+|^\s*\w+\s*:\s*=\s*\d+;')
_SECTION_CODE = re.compile(r'(?i)\bSECTION\s+code\b')
_PACKAGE_DECL = re.compile(r'(?im)^\s*(?:public\s+)?package\s+[\w.]+;')
_CONVERSION_TABLE = re.compile(r'(?i)\b(?:unchecked_conversion|to_integer|to_unsigned)\b')

_COMPILER_TESTSUITE = re.compile(
    r'(?i)(?:'
    r'gcc[\w.-]*/testsuite/|'
    r'testsuite/gnat\.(?:dg|pp)|'
    r'/gnat\.dg/'
    r')'
)

_GOV_TECH_DISCLAIMER = re.compile(
    r'(?i)\b(?:'
    r'released\s+technical\s+data|'
    r'government\s+makes\s+no\s+express|'
    r'use,?\s+duplicate,?\s+release\s+or\s+disclose|'
    r'redistributions?\s+of\s+source\s+code|'
    r'use\s+in\s+source\s+and\s+binary\s+forms'
    r')'
)

_COMPILER_SOURCE_REF = re.compile(
    r'(?i)[\w./\\-]+\.(?:ads|adb|svd)\s+(?:with\s+\w|package\s+)'
)

@dataclass
class CodeQualitySignals:
    comment_ratio: float = 0.0
    syntax_balance: float = 0.0
    generated_score: float = 0.0
    duplicate_line_ratio: float = 0.0
    educational_score: float = 0.0
    code_ratio: float = 0.0
    dump_probability: float = 0.0

@dataclass
class CodeDumpResult:
    probability: float = 0.0
    classification: Literal['prose', 'educational_code', 'mixed', 'raw_dump'] = 'prose'
    code_ratio: float = 0.0
    prose_words: int = 0
    should_remove: bool = False
    should_strip_code: bool = False
    reason: str = ''

def analyze_code(text: str) -> CodeQualitySignals:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return CodeQualitySignals()
    comments = sum(1 for ln in lines if _COMMENT.match(ln.strip()))
    comment_ratio = comments / len(lines)
    opens = text.count('{') + text.count('(') + text.count('[')
    closes = text.count('}') + text.count(')') + text.count(']')
    balance = 1.0 - min(1.0, abs(opens - closes) / max(opens + closes, 1))
    gen = 1.0 if _GENERATED.search(text) else 0.0
    line_counts = Counter(lines)
    dup_ratio = line_counts.most_common(1)[0][1] / len(lines) if lines else 0.0
    doc_hits = text.count('"""') + text.count("'''")
    defs = len(re.findall(r'^\s*def ', text, re.M))
    educational = min(1.0, doc_hits * 0.2 + comment_ratio * 0.5 + defs * 0.1)
    code_ratio = _code_line_ratio(text)
    dump_prob = 0.0
    if code_ratio > 0.25:
        dump_prob = min(1.0, code_ratio * 0.45 + generated_code_score(text) * 0.35)
    return CodeQualitySignals(
        comment_ratio=comment_ratio,
        syntax_balance=balance,
        generated_score=gen,
        duplicate_line_ratio=dup_ratio,
        educational_score=educational,
        code_ratio=code_ratio,
        dump_probability=dump_prob,
    )

def code_passes(signals: CodeQualitySignals, *, min_educational: float = 0.08) -> bool:
    if signals.generated_score > 0.5:
        return False
    if signals.duplicate_line_ratio > 0.55:
        return False
    if signals.syntax_balance < 0.55:
        return False
    return (
        signals.educational_score >= min_educational
        or signals.comment_ratio >= 0.03
        or signals.syntax_balance >= 0.72
    )

def vendor_sdk_hits(text: str) -> int:
    return len(_VENDOR_SDK.findall(text))

def _code_line_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    code_hits = sum(1 for ln in lines if _CODE_LINE.match(ln))
    fence_chars = sum(len(m.group(0)) for m in _CODE_FENCE.finditer(text))
    if fence_chars > len(text) * 0.5:
        return 0.9
    line_ratio = min(1.0, code_hits / len(lines))
    if len(lines) <= 3:
        keywords = len(re.findall(
            r'(?i)\b(?:'
            r'package|procedure|function|pragma|begin|end|loop|if|then|'
            r'type|record|#define|#include|def |class |namespace |import '
            r')\b',
            text,
        ))
        words = max(len(_WORD.findall(text)), 1)
        collapsed = min(1.0, keywords / max(words * 0.06, 1))
        semicolons = text.count(';')
        if semicolons >= 6:
            collapsed = max(collapsed, min(1.0, semicolons / max(words * 0.04, 1)))
        return max(line_ratio, collapsed)
    return line_ratio

def _prose_word_count(text: str) -> int:
    stripped = _CODE_FENCE.sub(' ', text)
    return len(_WORD.findall(stripped))

def _substantive_prose_words(text: str) -> int:
    parts: list[str] = []
    for ln in text.splitlines():
        if (
            _CODE_LINE.match(ln)
            or _HEX_CONST_LINE.match(ln)
            or _REGISTER_DEF.match(ln)
            or _ADA_PACKAGE.match(ln)
            or _ADA_TYPE_DUMP.search(ln)
        ):
            continue
        parts.append(ln)
    return len(_WORD.findall(' '.join(parts)))

def _count_pattern_lines(text: str, pattern: re.Pattern[str]) -> int:
    return sum(1 for ln in text.splitlines() if pattern.search(ln))

def _is_handwritten_algorithm(text: str) -> bool:
    if not _ADA_ROUTINE.search(text):
        return False
    if not re.search(r'(?i)\bbegin\b', text):
        return False
    return bool(re.search(
        r'(?i)\b(?:loop|while|for\s+\w+\s+in|if\s+.+\s+then|raise|exit)\b',
        text,
    ))

def generated_code_score(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    n = len(lines)
    score = 0.0
    if _GENERATED.search(text):
        score += 0.35
    counts = Counter(lines)
    dup_ratio = counts.most_common(1)[0][1] / n if counts else 0.0
    if dup_ratio > 0.35:
        score += min(0.35, dup_ratio * 0.5)
    enum_hits = _count_pattern_lines(text, _ENUM_MAPPING)
    hex_hits = _count_pattern_lines(text, _HEX_CONST_LINE)
    reg_hits = _count_pattern_lines(text, _REGISTER_DEF)
    pkg_hits = len(_PACKAGE_DECL.findall(text)) + len(_ADA_PACKAGE.findall(text))
    if enum_hits >= 12 and enum_hits / n > 0.25:
        score += 0.30
    if hex_hits >= 10 and hex_hits / n > 0.20:
        score += 0.28
    if reg_hits >= 6:
        score += 0.25
    if pkg_hits >= 4:
        score += 0.22
    if _CONVERSION_TABLE.search(text) and hex_hits >= 4:
        score += 0.15
    if _SECTION_CODE.search(text):
        score += 0.20
    return min(1.0, score)

def _actionable_code_lines(text: str, *, min_hits: int = 2) -> int:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    hits = sum(1 for ln in lines if _CODE_LINE.match(ln) or _ADA_PACKAGE.match(ln))
    if hits >= min_hits:
        return hits
    if _CODE_FENCE.search(text):
        return max(hits, min_hits)
    inline = len(re.findall(
        r'(?i)\b(?:package\s+(?:body\s+)?\w|with\s+[A-Z]\w*(?:\.[A-Z]\w*)*;)',
        text,
    ))
    return hits + inline

def _has_actionable_code(text: str) -> bool:
    return _actionable_code_lines(text) >= 2

def _is_declaration_only_dump(text: str) -> bool:
    begins = len(re.findall(r'(?i)\bbegin\b', text))
    decls = len(re.findall(r'(?i)\b(?:type|procedure|function)\s+\w+', text))
    if decls < 2 or begins > 0:
        return False
    ratio = _code_line_ratio(text)
    if ratio < 0.45:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1 and text.count(';') >= 6:
        return True
    return decls >= 3 and ratio >= 0.50

def _strip_inline_prose_prefix(text: str) -> tuple[str, bool]:
    m = re.search(
        r'(?is)^(.{40,}?)(?:\s)(?:'
        r'examples/[\w./-]+\.(?:adb|ads|adb)\b|'
        r'with\s+[A-Z][\w.]*(?:\s*;\s*use\s+[A-Z][\w.]*)?|'
        r'package\s+(?:body\s+)?[A-Z]\w*'
        r')',
        text.strip(),
    )
    if not m:
        return text, False
    prefix = m.group(1).strip()
    if _substantive_prose_words(prefix) < 12:
        return text, False
    return prefix, True

def _has_educational_context(text: str) -> bool:
    sample = _CODE_FENCE.sub(' ', text)
    return bool(re.search(
        r'(?i)\b(?:'
        r'tutorial|example|explain|algorithm|step[\s-]by[\s-]step|'
        r'how\s+to|implementation|overview|documentation|'
        r'the\s+following\s+code|this\s+(?:function|class|method)|'
        r'we\s+(?:define|implement|create)|'
        r'output|input|parameter|returns?'
        r')\b',
        sample,
    ))

def analyze_code_dump(text: str) -> CodeDumpResult:
    if not text or not text.strip():
        return CodeDumpResult(classification='raw_dump', probability=1.0, should_remove=True, reason='empty')

    if _COMPILER_TESTSUITE.search(text):
        return CodeDumpResult(
            probability=0.92,
            classification='raw_dump',
            code_ratio=_code_line_ratio(text),
            prose_words=_prose_word_count(text),
            should_remove=True,
            reason='compiler_testsuite',
        )

    code_ratio = _code_line_ratio(text)
    prose_words = _prose_word_count(text)
    substantive_words = _substantive_prose_words(text)
    code_sig = analyze_code(text)
    educational = (
        _has_educational_context(text)
        or (code_sig.educational_score > 0.12 and substantive_words >= 10)
    )
    build_hits = len(_BUILD_SCRIPT.findall(text))
    ext_hits = len(_FILE_EXT_LINE.findall(text))
    readme_only = bool(_README_ONLY.search(text)) and prose_words < 80
    generated = bool(_GENERATED.search(text))
    gen_score = generated_code_score(text)
    vendor_sdk = bool(_VENDOR_SDK.search(text))
    compiler_internal = bool(_COMPILER_INTERNAL.search(text))
    ada_packages = len(_ADA_PACKAGE.findall(text))
    ada_type_lines = _count_pattern_lines(text, _ADA_TYPE_DUMP)
    handwritten_algo = _is_handwritten_algorithm(text)

    if _GOV_TECH_DISCLAIMER.search(text) and code_ratio > 0.20:
        return CodeDumpResult(
            probability=0.88,
            classification='raw_dump',
            code_ratio=code_ratio,
            prose_words=prose_words,
            should_remove=True,
            reason='license_or_disclaimer_dump',
        )

    if (
        _COMPILER_SOURCE_REF.search(text)
        and substantive_words < 50
        and not educational
        and not handwritten_algo
    ):
        return CodeDumpResult(
            probability=0.85,
            classification='raw_dump',
            code_ratio=code_ratio,
            prose_words=prose_words,
            should_remove=True,
            reason='compiler_source_ref',
        )

    if _is_declaration_only_dump(text) and substantive_words < 25 and not handwritten_algo:
        return CodeDumpResult(
            probability=0.84,
            classification='raw_dump',
            code_ratio=code_ratio,
            prose_words=prose_words,
            should_remove=True,
            reason='declaration_only_dump',
        )

    if vendor_sdk and substantive_words < 20 and code_ratio > 0.70 and not handwritten_algo:
        return CodeDumpResult(
            probability=0.86,
            classification='raw_dump',
            code_ratio=code_ratio,
            prose_words=prose_words,
            should_remove=True,
            reason='vendor_sdk_dump',
        )

    prob = 0.0
    if code_ratio < 0.25:
        return CodeDumpResult(
            probability=0.0,
            classification='prose',
            code_ratio=code_ratio,
            prose_words=prose_words,
        )

    prob += code_ratio * 0.45
    reg_hits = _count_pattern_lines(text, _REGISTER_DEF)
    hex_hits = _count_pattern_lines(text, _HEX_CONST_LINE)
    line_count = max(len([ln for ln in text.splitlines() if ln.strip()]), 1)

    if build_hits >= 1 and substantive_words < 25:
        prob = max(prob, 0.80)
    if build_hits >= 2:
        prob = max(prob, 0.88)
    if ext_hits >= 3 and substantive_words < 60:
        prob = max(prob, 0.78)
    if readme_only:
        prob = max(prob, 0.85)
    if generated and substantive_words < 40:
        prob = max(prob, 0.80)
    if gen_score >= 0.55 and substantive_words < 50:
        prob = max(prob, 0.72 + gen_score * 0.2)
    if vendor_sdk and substantive_words < 45 and not handwritten_algo:
        prob = max(prob, 0.78)
    if compiler_internal and substantive_words < 35:
        prob = max(prob, 0.82)
    if ada_packages >= 1 and substantive_words < 25 and ada_type_lines >= 3 and not handwritten_algo:
        prob = max(prob, 0.80)
    if ada_packages >= 2 and substantive_words < 40 and not educational and not handwritten_algo:
        prob = max(prob, 0.76)
    if (reg_hits >= 6 or hex_hits >= 10) and hex_hits / line_count > 0.45:
        if not educational and not handwritten_algo:
            prob = max(prob, 0.86)
    if code_ratio > 0.80 and substantive_words < 35:
        prob = max(prob, 0.88)

    if (reg_hits >= 6 or hex_hits >= 10) and hex_hits / line_count > 0.45:
        if not handwritten_algo and not _has_educational_context(text):
            return CodeDumpResult(
                probability=max(prob, 0.86),
                classification='raw_dump',
                code_ratio=code_ratio,
                prose_words=prose_words,
                should_remove=True,
                reason='register_or_constant_table',
            )

    if vendor_sdk and substantive_words < 60 and not handwritten_algo:
        return CodeDumpResult(
            probability=max(prob, 0.88),
            classification='raw_dump',
            code_ratio=code_ratio,
            prose_words=prose_words,
            should_remove=True,
            reason='vendor_sdk_dump',
        )

    if handwritten_algo and not vendor_sdk:
        return CodeDumpResult(
            probability=min(prob, 0.30),
            classification='educational_code',
            code_ratio=code_ratio,
            prose_words=prose_words,
        )

    if educational and code_passes(code_sig):
        return CodeDumpResult(
            probability=min(prob, 0.35),
            classification='educational_code',
            code_ratio=code_ratio,
            prose_words=prose_words,
        )

    if (
        educational
        and code_ratio > 0.45
        and prose_words >= 35
        and substantive_words >= 15
        and _has_actionable_code(text)
        and ada_packages >= 1
        and not vendor_sdk
    ):
        return CodeDumpResult(
            probability=min(prob, 0.42),
            classification='mixed',
            code_ratio=code_ratio,
            prose_words=prose_words,
            should_strip_code=True,
            reason='educational_code_tail',
        )

    if code_ratio > 0.55 and prose_words >= 50 and educational:
        return CodeDumpResult(
            probability=min(prob, 0.40),
            classification='mixed',
            code_ratio=code_ratio,
            prose_words=prose_words,
        )

    if code_ratio > 0.70 and substantive_words < 45 and not handwritten_algo:
        if (
            prose_words >= 28
            and substantive_words >= 10
            and not vendor_sdk
            and _has_actionable_code(text)
        ):
            return CodeDumpResult(
                probability=max(prob, 0.55),
                classification='mixed',
                code_ratio=code_ratio,
                prose_words=prose_words,
                should_strip_code=True,
                reason='code_dominated_mixed',
            )
        return CodeDumpResult(
            probability=max(prob, 0.75),
            classification='raw_dump',
            code_ratio=code_ratio,
            prose_words=prose_words,
            should_remove=True,
            reason='raw_code_dump',
        )

    if prob >= 0.50 and not handwritten_algo and not educational:
        return CodeDumpResult(
            probability=prob,
            classification='raw_dump',
            code_ratio=code_ratio,
            prose_words=prose_words,
            should_remove=True,
            reason='configuration_or_build_script',
        )
    if prob >= 0.65 and not handwritten_algo:
        return CodeDumpResult(
            probability=prob,
            classification='raw_dump',
            code_ratio=code_ratio,
            prose_words=prose_words,
            should_remove=True,
            reason='configuration_or_build_script',
        )

    if (
        code_ratio > 0.55
        and substantive_words < 100
        and not educational
        and not handwritten_algo
        and prose_words >= 12
        and _has_actionable_code(text)
    ):
        return CodeDumpResult(
            probability=prob,
            classification='mixed',
            code_ratio=code_ratio,
            prose_words=prose_words,
            should_strip_code=True,
            reason='code_dominated_mixed',
        )

    return CodeDumpResult(
        probability=prob,
        classification='mixed' if code_ratio > 0.35 else 'prose',
        code_ratio=code_ratio,
        prose_words=prose_words,
    )

def _code_line_ratio_in_block(block: str) -> float:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    hits = sum(1 for ln in lines if _CODE_LINE.match(ln) or _ADA_PACKAGE.match(ln))
    return hits / len(lines)

def _is_code_heavy_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    if _CODE_FENCE.search(stripped):
        return True
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    ratio = _code_line_ratio_in_block(stripped)
    ada_routines = len(_ADA_ROUTINE.findall(stripped))
    if ada_routines >= 2 and ratio >= 0.35:
        return True
    if ratio >= 0.65 and len(lines) >= 3:
        return True
    if _GENERATED.search(stripped) and ratio >= 0.45:
        return True
    if _VENDOR_SDK.search(stripped) and ratio >= 0.50:
        return True
    return False

def _strip_mixed_single_paragraph(text: str) -> tuple[str, bool]:
    inline, inline_changed = _strip_inline_prose_prefix(text)
    if inline_changed:
        return inline, True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return text, False
    code_start: int | None = None
    for i, ln in enumerate(lines):
        if i > 0 and (_CODE_LINE.match(ln) or _ADA_PACKAGE.match(ln)):
            code_start = i
            break
    if code_start is None:
        return text, False
    prose_part = '\n'.join(lines[:code_start]).strip()
    code_part = '\n'.join(lines[code_start:]).strip()
    if _substantive_prose_words(prose_part) < 12:
        return text, False
    if not _is_code_heavy_block(code_part) and _code_line_ratio_in_block(code_part) < 0.50:
        return text, False
    return prose_part, True

def _strip_unfenced_code_blocks(text: str) -> tuple[str, bool]:
    paragraphs = re.split(r'\n\s*\n', text.strip())
    if len(paragraphs) == 1:
        return _strip_mixed_single_paragraph(paragraphs[0])
    if len(paragraphs) < 2:
        return text, False

    start = 0
    end = len(paragraphs)
    changed = False
    while start < end and _is_code_heavy_block(paragraphs[start]):
        start += 1
        changed = True
    while end > start and _is_code_heavy_block(paragraphs[end - 1]):
        end -= 1
        changed = True

    if not changed or start >= end:
        return text, False
    kept = [p.strip() for p in paragraphs[start:end] if p.strip()]
    if not kept:
        return text, False
    if _substantive_prose_words('\n\n'.join(kept)) < 12:
        return text, False
    return '\n\n'.join(kept), True

def strip_isolated_code_blocks(text: str, *, min_prose_words: int = 30) -> tuple[str, bool]:
    prose = _CODE_FENCE.sub(' ', text)
    if len(_WORD.findall(prose)) >= min_prose_words and _has_educational_context(text):
        return text, False

    def _replace_fence(match: re.Match[str]) -> str:
        block = match.group(0)
        inner = re.sub(r'^```\w*\n?', '', block)
        inner = re.sub(r'\n?```$', '', inner)
        if _has_educational_context(inner) or analyze_code(inner).educational_score > 0.15:
            return block
        return ''

    cleaned = _CODE_FENCE.sub(_replace_fence, text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    fenced_changed = cleaned != text.strip()
    if fenced_changed:
        return cleaned, True
    return _strip_unfenced_code_blocks(text)

def strip_dominant_code(text: str, *, code_dump: CodeDumpResult | None = None) -> tuple[str, bool]:
    if (
        code_dump is not None
        and code_dump.classification == 'educational_code'
        and not code_dump.should_strip_code
    ):
        return text, False
    cleaned, changed = strip_isolated_code_blocks(text, min_prose_words=25)
    if changed:
        return cleaned, True
    if code_dump is not None and code_dump.should_strip_code:
        return _strip_unfenced_code_blocks(text)
    return text, False
