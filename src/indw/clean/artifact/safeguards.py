from __future__ import annotations

import re

_DOI = re.compile(r'\b10\.\d{4,9}/\S+', re.I)
_ISSN = re.compile(r'\b\d{4}-\d{3}[\dxX]\b')
_ARXIV = re.compile(r'\barxiv:\d+\.\d+', re.I)
_MATH = re.compile(r'(?:\$\$[\s\S]+?\$\$|\\begin\{|\\frac\{|\\sum_|\\int_)')
_CODE_FENCE = re.compile(r'```[\s\S]*?```')
_TABLE_ROW = re.compile(r'^\s*\|?.+\|.+\|?\s*$')
_FORM_FIELD = re.compile(r'(?m)^\s*(?:date|score|grade|rating|violation|inspector|facility)\s*:', re.I)
_MEDICAL_DENSE = re.compile(
    r'\b(?:mg/dl|mmhg|bpm|diagnosis|patient|symptom|pathology|inspection|violation)\b',
    re.I,
)
_TECH_DENSE = re.compile(r'\b(?:def |class |import |function |theorem|proof|lemma)\b')

def is_protected_unit(text: str, *, kind: str = '', in_fence: bool = False) -> bool:
    if in_fence or kind == 'code':
        return True
    if not text or not text.strip():
        return False
    t = text.strip()
    if _CODE_FENCE.search(t):
        return True
    if _MATH.search(t):
        return True
    if _DOI.search(t) or _ISSN.search(t) or _ARXIV.search(t):
        return True
    lines = t.splitlines()
    table_rows = sum(1 for ln in lines if _TABLE_ROW.match(ln))
    if table_rows >= 2 and table_rows / max(len(lines), 1) >= 0.4:
        return True
    if _FORM_FIELD.search(t) and len(lines) >= 2:
        return True
    words = len(re.findall(r'\b\w+\b', t, flags=re.UNICODE))
    if words >= 8:
        med = len(_MEDICAL_DENSE.findall(t))
        tech = len(_TECH_DENSE.findall(t))
        if med >= 2 or tech >= 2:
            return True
    return False
