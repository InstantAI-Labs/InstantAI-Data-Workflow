from __future__ import annotations

import base64
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from indw.filter.pii.config import SecretConfig

_B64URL = re.compile(r'[A-Za-z0-9_\-]{16,}={0,2}')
_JWT = re.compile(r'\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b')
_PEM = re.compile(r'-----BEGIN [A-Z ]+-----')
_HEX_RUN = re.compile(r'\b[0-9a-fA-F]{24,}\b')
_UUID = re.compile(
    r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
    re.I,
)

def _is_natural_language_token(token: str) -> bool:
    core = re.sub(r'[^A-Za-z\-]', '', token)
    if len(core) < 5 or len(core) > 48:
        return False
    if not core.replace('-', '').isalpha():
        return False
    vowels = sum(1 for c in core.lower() if c in 'aeiou')
    return vowels / len(core) >= 0.28

def _is_schema_template_token(token: str) -> bool:
    low = token.lower().strip('\'",;')
    if '@' not in low:
        return False
    local, _, domain = low.partition('@')
    if domain.endswith('.tld') or domain.endswith('.invalid') or domain == 'domain.tld':
        return True
    if domain.startswith('example.') or (local.isalpha() and len(local) <= 12 and 'domain' in domain):
        return True
    return False

def _is_placeholder_token(token: str) -> bool:
    if token in {'xxx', 'xxxx', 'redacted', 'redacted_sample', 'placeholder', 'sample', 'example'}:
        return True
    if 'redacted' in token or 'placeholder' in token:
        return True
    if re.fullmatch(r'[x*#.\-]{6,}', token):
        return True
    return False

_SK_LIKE = re.compile(r'\b[a-z]{2,8}[-_][A-Za-z0-9_\-]{16,}\b')
_TECH_EDU_CONTEXT = re.compile(
    r'(?i)\b(?:tutorial|documentation|example|mysql|postgres|authentication|'
    r'plugin|configure|parameter|schema|function|procedure|implementation|'
    r'we begin by|in this section|step \d+|how to|package body|procedure\s+\w+)\b'
)

@dataclass
class SecretSpan:
    text: str
    start: int
    end: int
    probability: float
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'text': self.text[:8] + '…' if len(self.text) > 12 else self.text,
            'start': self.start,
            'end': self.end,
            'probability': round(self.probability, 4),
            'signals': {k: round(v, 4) for k, v in self.signals.items()},
        }

@dataclass
class SecretAnalysisResult:
    spans: list[SecretSpan] = field(default_factory=list)
    secret_probability: float = 0.0

    def to_dict(self) -> dict:
        return {
            'secret_probability': round(self.secret_probability, 4),
            'spans': [s.to_dict() for s in self.spans[:32]],
        }

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def _charset_diversity(s: str) -> float:
    classes = set()
    for ch in s:
        if ch.islower():
            classes.add('l')
        elif ch.isupper():
            classes.add('u')
        elif ch.isdigit():
            classes.add('d')
        else:
            classes.add('s')
    return len(classes) / 4.0

class SecretAnalyzer:

    def __init__(self, config: SecretConfig):
        self.config = config

    def _score_candidate(self, token: str, *, context: str = '') -> tuple[float, dict[str, float]]:
        if len(token) < self.config.min_token_len or len(token) > self.config.max_token_len:
            return 0.0, {}
        low = token.lower().strip('[]')
        if _is_placeholder_token(low) or _is_natural_language_token(token) or _is_schema_template_token(token):
            return 0.0, {'placeholder': 1.0}
        if len(set(token)) <= 4:
            counts = Counter(token)
            if counts and counts.most_common(1)[0][1] / len(token) > 0.65:
                return 0.0, {'placeholder': 1.0}
        ent = _entropy(token)
        div = _charset_diversity(token)
        signals: dict[str, float] = {
            'entropy': min(1.0, ent / 5.0),
            'diversity': div,
            'length': min(1.0, len(token) / 48.0),
        }
        score = 0.0
        if _JWT.fullmatch(token):
            score = 0.98
            signals['jwt'] = 1.0
        elif token.startswith('eyJ'):
            score = max(score, 0.9)
            signals['jwt_prefix'] = 1.0
        elif _PEM.search(token):
            score = 0.95
            signals['pem'] = 1.0
        if ent >= self.config.min_entropy and div >= 0.5:
            score = max(score, min(1.0, 0.35 + ent / 6.0 + div * 0.25))
        if _SK_LIKE.fullmatch(token):
            score = max(score, min(1.0, 0.55 + ent / 8.0))
            signals['structured_prefix'] = 0.8
        try:
            pad = '=' * ((4 - len(token) % 4) % 4)
            base64.urlsafe_b64decode(token + pad)
            if ent >= 4.0 and len(token) >= 20:
                score = max(score, min(1.0, 0.5 + ent / 7.0))
                signals['base64'] = 1.0
        except Exception:
            pass
        ctx = context.lower()
        has_credential_context = any(
            k in ctx
            for k in ('password', 'secret', 'token', 'apikey', 'api_key', 'credential', 'private')
        )
        technical_edu = bool(_TECH_EDU_CONTEXT.search(ctx))
        if technical_edu and not has_credential_context:
            score *= 0.45
            signals['technical_edu'] = 0.7
        elif technical_edu and has_credential_context:
            score *= 0.65
            signals['technical_edu'] = 0.5
        if _UUID.fullmatch(token):
            return 0.0, {'uuid': 1.0}
        if _HEX_RUN.fullmatch(token) and ent >= 3.5:
            if has_credential_context:
                score = max(score, 0.72)
                signals['hex'] = 1.0
            else:
                score = max(score, 0.42)
                signals['hex_reference'] = 0.5
        has_credential_shape = bool(
            re.search(r'\d', token) or re.search(r'[_\-=+/]', token) or len(token) >= 20
        )
        if has_credential_shape and ent >= 3.2 and has_credential_context:
            score = min(1.0, score + 0.15)
            signals['context'] = 0.8
        return score, signals

    def analyze(self, text: str) -> SecretAnalysisResult:
        if not text:
            return SecretAnalysisResult()
        spans: list[SecretSpan] = []
        for m in _JWT.finditer(text):
            sc, sig = self._score_candidate(m.group(0), context=text[max(0, m.start() - 40) : m.end() + 20])
            if sc >= self.config.min_secret_score:
                spans.append(SecretSpan(m.group(0), m.start(), m.end(), sc, sig))
        tokens = re.finditer(r'\S+', text)
        for m in tokens:
            tok = m.group(0).strip('\'",;')
            if len(tok) < self.config.min_token_len:
                continue
            window = text[max(0, m.start() - 50) : m.end() + 50]
            sc, sig = self._score_candidate(tok, context=window)
            if sc >= self.config.min_secret_score:
                spans.append(SecretSpan(tok, m.start(), m.start() + len(tok), sc, sig))
        if not spans:
            return SecretAnalysisResult()
        merged: list[SecretSpan] = []
        for sp in sorted(spans, key=lambda s: -s.probability):
            if any(sp.start < k.end and sp.end > k.start for k in merged):
                continue
            merged.append(sp)
        peak = max(s.probability for s in merged)
        doc_prob = min(1.0, peak * 0.88 + 0.012 * min(len(merged), 5))
        if _TECH_EDU_CONTEXT.search(text[:4000]):
            doc_prob *= 0.55
        return SecretAnalysisResult(spans=merged, secret_probability=doc_prob)
