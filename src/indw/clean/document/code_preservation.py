from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Literal

from indw.clean.document.patterns import _CODE_FENCE

_LANG_HINT = re.compile(r'^```\s*(\w[\w+-]*)')
_SHEBANG = re.compile(r'^#!.*?(?:python|perl|ruby|bash|sh|node|php)', re.I)
_FILE_EXT = re.compile(
    r'(?i)\.(?:py|js|ts|jsx|tsx|java|cs|cpp|cc|c|h|hpp|go|rs|rb|php|sh|bash|sql|yaml|yml|toml|json|xml|ada|adb|ads|vhdl|sv|scala|kt|swift|lua|r|m|pl)\b'
)

_LANG_SIGNATURES: dict[str, list[re.Pattern[str]]] = {
    'python': [
        re.compile(r'^\s*(?:def |class |import |from \w+ import|async def )', re.M),
        re.compile(r'^\s*@\w+', re.M),
        re.compile(r'^\s*if __name__\s*==', re.M),
    ],
    'javascript': [
        re.compile(r'^\s*(?:const|let|var|function|export |import )', re.M),
        re.compile(r'=>|console\.|require\(', re.M),
    ],
    'typescript': [
        re.compile(r'^\s*(?:interface |type \w+\s*=|enum )', re.M),
        re.compile(r':\s*(?:string|number|boolean|void)\b', re.M),
    ],
    'java': [
        re.compile(r'^\s*(?:public|private|protected)\s+(?:static\s+)?(?:class|interface|void|int)', re.M),
        re.compile(r'^\s*package\s+[\w.]+;', re.M),
    ],
    'c': [
        re.compile(r'^\s*#include\s*[<"]', re.M),
        re.compile(r'^\s*(?:int|void|char|float|double)\s+\w+\s*\(', re.M),
    ],
    'cpp': [
        re.compile(r'^\s*#include\s*[<"]', re.M),
        re.compile(r'^\s*(?:namespace |template\s*<|std::)', re.M),
    ],
    'go': [
        re.compile(r'^\s*(?:package |func |import \()', re.M),
        re.compile(r':=\s*', re.M),
    ],
    'rust': [
        re.compile(r'^\s*(?:fn |impl |pub |use |mod )', re.M),
        re.compile(r'->\s*\w+', re.M),
    ],
    'sql': [
        re.compile(r'(?i)^\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b', re.M),
    ],
    'shell': [
        re.compile(r'^\s*(?:#!/|export |if \[ |fi$|done$|for \w+ in)', re.M),
    ],
    'ada': [
        re.compile(r'(?i)^\s*(?:package|procedure|function|pragma|with\s+\w)', re.M),
    ],
    'html': [
        re.compile(r'(?i)<(?:html|div|span|body|head|script|style)\b'),
    ],
    'json': [
        re.compile(r'^\s*[\[{]', re.M),
    ],
}

_BRACE_PAIRS = {'(': ')', '[': ']', '{': '}'}
_SINGLE_LINE_COLLAPSE = re.compile(
    r'(?m)^(\s*)(?:def |class |function |if |for |while |switch |package |procedure |import )'
    r'[^;\n]{80,};?\s*$'
)

@dataclass
class CodeBlockInfo:
    language: str = 'unknown'
    line_count: int = 0
    syntax_valid: bool = True
    syntax_issues: list[str] = field(default_factory=list)
    fence_normalized: bool = False

@dataclass
class CodePreservationStats:
    blocks_processed: int = 0
    blocks_repaired: int = 0
    fences_added: int = 0
    fences_removed: int = 0
    syntax_failures: int = 0
    collapsed_lines_restored: int = 0
    chars_before: int = 0
    chars_after: int = 0

def detect_code_language(block: str, *, hint: str = '') -> str:
    if hint:
        lang = hint.lower().split('-')[0]
        if lang in _LANG_SIGNATURES or lang in ('py', 'js', 'ts', 'sh', 'bash'):
            return {'py': 'python', 'js': 'javascript', 'ts': 'typescript', 'sh': 'shell', 'bash': 'shell'}.get(lang, lang)

    scores: dict[str, int] = {}
    sample = block[:8000]
    if _SHEBANG.search(sample):
        scores['shell'] = scores.get('shell', 0) + 3
    ext = _FILE_EXT.search(sample)
    if ext:
        ext_lang = ext.group(0).lstrip('.').lower()
        ext_map = {'adb': 'ada', 'ads': 'ada', 'sh': 'shell', 'bash': 'shell', 'rb': 'ruby', 'rs': 'rust', 'kt': 'kotlin'}
        scores[ext_map.get(ext_lang, ext_lang)] = scores.get(ext_map.get(ext_lang, ext_lang), 0) + 2

    for lang, patterns in _LANG_SIGNATURES.items():
        hits = sum(1 for p in patterns if p.search(sample))
        if hits:
            scores[lang] = scores.get(lang, 0) + hits

    if not scores:
        return 'text'
    return max(scores, key=scores.get)

def validate_code_syntax(block: str, language: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    lang = language.lower()
    stripped = block.strip()
    if not stripped:
        return True, issues

    opens = {k: stripped.count(k) for k in _BRACE_PAIRS}
    closes = {v: stripped.count(v) for v in _BRACE_PAIRS.values()}
    for open_ch, close_ch in _BRACE_PAIRS.items():
        diff = opens[open_ch] - closes[close_ch]
        if abs(diff) > max(2, len(stripped) // 500):
            issues.append(f'unbalanced_{open_ch}{close_ch}')

    if lang == 'python':
        try:
            ast.parse(stripped)
        except SyntaxError as exc:
            if exc.lineno and exc.lineno > len(stripped.splitlines()) * 0.9:
                issues.append('truncated_python')
            else:
                issues.append('python_syntax')
    elif lang == 'json':
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            if not stripped.rstrip().endswith(('}', ']')):
                issues.append('truncated_json')
            else:
                issues.append('json_syntax')

    if _SINGLE_LINE_COLLAPSE.search(stripped) and stripped.count('\n') < 3:
        issues.append('collapsed_code')

    return len(issues) == 0, issues

def _split_fences(text: str) -> list[tuple[Literal['prose', 'code'], str, str]]:
    parts: list[tuple[Literal['prose', 'code'], str, str]] = []
    pos = 0
    for match in _CODE_FENCE.finditer(text):
        if match.start() > pos:
            parts.append(('prose', text[pos:match.start()], ''))
        raw = match.group(0)
        inner = re.sub(r'^```[^\n]*\n?', '', raw)
        inner = re.sub(r'\n?```\s*$', '', inner)
        hint_m = _LANG_HINT.match(raw)
        hint = hint_m.group(1) if hint_m else ''
        parts.append(('code', inner, hint))
        pos = match.end()
    if pos < len(text):
        parts.append(('prose', text[pos:], ''))
    return parts

def _normalize_fence_block(block: str, hint: str, *, add_fences: bool) -> tuple[str, CodeBlockInfo]:
    info = CodeBlockInfo()
    lines = block.splitlines()
    info.line_count = len(lines)

    restored = 0
    for ln in lines:
        if ';' in ln and ln.count(';') >= 3 and ' ' not in ln.strip(';'):
            restored += 1
    if restored:
        info.syntax_issues.append('collapsed_code')

    lang = detect_code_language(block, hint=hint)
    info.language = lang
    valid, issues = validate_code_syntax(block, lang)
    info.syntax_valid = valid
    info.syntax_issues.extend(issues)

    body = block
    if not body.endswith('\n') and '\n' in body:
        body = body + '\n'

    if add_fences and info.line_count >= 2:
        tag = lang if lang not in ('text', 'unknown') else ''
        fenced = f'```{tag}\n{body.rstrip()}\n```' if tag else f'```\n{body.rstrip()}\n```'
        info.fence_normalized = True
        return fenced, info
    return body, info

def _detect_unfenced_code_blocks(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    lines = text.splitlines(keepends=True)
    offset = 0
    run_start: int | None = None
    run_lines = 0
    for line in lines:
        stripped = line.strip()
        is_code = bool(stripped and (
            _LANG_SIGNATURES['python'][0].match(line)
            or _LANG_SIGNATURES['javascript'][0].match(line)
            or _LANG_SIGNATURES['c'][0].match(line)
            or _LANG_SIGNATURES['java'][0].match(line)
            or (stripped.startswith(('    ', '\t')) and any(c in stripped for c in '={}();'))
        ))
        if is_code:
            if run_start is None:
                run_start = offset
            run_lines += 1
        elif run_start is not None and run_lines >= 3:
            spans.append((run_start, offset))
            run_start = None
            run_lines = 0
        else:
            run_start = None
            run_lines = 0
        offset += len(line)
    if run_start is not None and run_lines >= 3:
        spans.append((run_start, offset))
    return spans

def preserve_code_blocks(
    text: str,
    *,
    normalize_fences: bool = True,
    fence_unfenced: bool = False,
    validate_syntax: bool = True,
) -> tuple[str, CodePreservationStats]:
    if not text or '```' not in text and not fence_unfenced:
        return text, CodePreservationStats(chars_before=len(text), chars_after=len(text))

    stats = CodePreservationStats(chars_before=len(text))
    segments = _split_fences(text)
    out_parts: list[str] = []

    for kind, content, hint in segments:
        if kind == 'prose':
            out_parts.append(content)
            continue
        stats.blocks_processed += 1
        block, info = _normalize_fence_block(
            content, hint, add_fences=normalize_fences,
        )
        if validate_syntax and not info.syntax_valid:
            stats.syntax_failures += 1
        if info.fence_normalized:
            stats.fences_added += 1
        if info.syntax_issues:
            stats.blocks_repaired += 1
        out_parts.append(block)

    result = ''.join(out_parts)
    stats.chars_after = len(result)
    return result, stats

def audit_code_integrity(text: str) -> dict[str, int | float | list[str]]:
    segments = _split_fences(text)
    blocks = 0
    valid = 0
    langs: dict[str, int] = {}
    issues: list[str] = []
    for kind, content, hint in segments:
        if kind != 'code':
            continue
        blocks += 1
        lang = detect_code_language(content, hint=hint)
        langs[lang] = langs.get(lang, 0) + 1
        ok, block_issues = validate_code_syntax(content, lang)
        if ok:
            valid += 1
        else:
            issues.extend(f'{lang}:{i}' for i in block_issues[:2])
    return {
        'code_blocks': blocks,
        'syntax_valid_rate': round(valid / max(blocks, 1), 4),
        'languages': langs,
        'issues': issues[:20],
    }
