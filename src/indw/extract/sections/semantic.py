from __future__ import annotations

from dataclasses import dataclass, field

from indw.extract.structure.analyze import analyze_structure
from indw.filter.refine.truncation import _DANGLING_END, _last_sentence_boundary
from indw.clean.artifact.evidence_engine import resolve_semantic_evidence
from indw.clean.artifact.evidence_features import DocumentFeatureExtractor
from indw.clean.artifact.evidence_model import AdaptiveBaselineEstimator
from indw.clean.document.value import compute_structure_profile

@dataclass
class CompletionProfile:
    sentence: float = 0.0
    paragraph: float = 0.0
    quotation: float = 0.0
    dialogue: float = 0.0
    topic: float = 0.0
    narrative: float = 0.0
    explanation: float = 0.0
    overall: float = 0.0
    incomplete_probability: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            'sentence': round(self.sentence, 4),
            'paragraph': round(self.paragraph, 4),
            'quotation': round(self.quotation, 4),
            'dialogue': round(self.dialogue, 4),
            'topic': round(self.topic, 4),
            'narrative': round(self.narrative, 4),
            'explanation': round(self.explanation, 4),
            'overall': round(self.overall, 4),
            'incomplete_probability': round(self.incomplete_probability, 4),
            'signals': {k: round(v, 4) for k, v in self.signals.items()},
        }

def _tail_span(text: str) -> tuple[str, str]:
    t = text.strip()
    if not t:
        return '', ''
    boundary = _last_sentence_boundary(t)
    if boundary > 0 and boundary < len(t):
        head = t[:boundary].strip()
        tail = t[boundary:].strip()
        if tail:
            return head, tail

    comma = t.rfind(',')
    if comma > max(40, int(len(t) * 0.12)):
        head = t[:comma].strip()
        tail = t[comma + 1:].strip()
        if tail:
            return head, tail

    words = t.split()
    if len(words) > 7:
        head = ' '.join(words[:-6])
        tail = ' '.join(words[-6:])
        return head, tail
    return '', t

def _quote_open(text: str) -> float:
    score = 1.0
    pairs = (
        ('"', '"'),
        ("'", "'"),
        ('\u201c', '\u201d'),
        ('\u2018', '\u2019'),
        ('«', '»'),
    )
    for open_q, close_q in pairs:
        opens = text.count(open_q)
        closes = text.count(close_q)
        if opens != closes:
            delta = abs(opens - closes)
            score = min(score, max(0.0, 1.0 - delta * 0.40))
    return score

def _terminal_punct_valid(text: str) -> float:
    from indw.extract.sections.boundaries import period_ends_sentence

    t = text.rstrip()
    if not t:
        return 0.0
    last = t[-1]
    if last in '.!?':
        if last == '.':
            dot = t.rfind('.')
            if dot >= 0 and not period_ends_sentence(t, dot):
                return 0.22
        return 0.92
    if last in '"\'”»)]}':
        return 0.82
    if last in ',:;–—-':
        return 0.10
    if last in '({[<':
        return 0.06
    return 0.38

def _tail_continuation_risk(tail: str, *, full: str) -> float:
    if not tail.strip():
        return 0.0
    t = tail.strip()
    words = t.split()
    if not words:
        return 0.5
    raw = DocumentFeatureExtractor().extract(t)
    structural = analyze_structure(t)
    ev = resolve_semantic_evidence(t)
    profile = compute_structure_profile(t, evidence=ev)
    baseline = AdaptiveBaselineEstimator()

    risk = 0.0
    last = words[-1].strip('\'",;:.!?)]}')
    last_lower = last.lower()

    if last_lower in _DANGLING_END:
        risk = max(risk, 0.88)
    if last_lower == 'such' and t.rstrip().endswith(('.', '!', '?')):
        risk = max(risk, 0.86)
    if len(last) == 1 and last.isalpha():
        risk = max(risk, 0.86)
    if len(words) <= 3 and structural.sentence_completeness_mean < 0.55:
        risk = max(risk, 0.72)
    if raw.word_count <= 4 and profile.explanation_ratio < 0.08 and profile.fact_ratio < 0.12:
        risk = max(risk, 0.68)
    tail_words = words[-6:] if len(words) > 6 else words
    tail_raw = DocumentFeatureExtractor().extract(' '.join(tail_words))
    if tail_raw.year_hits > 0:
        last_tw = tail_words[-1].strip('.,;:.!?)]}').lower()
        if last_tw.isalpha() and len(last_tw) <= 8:
            if not t.rstrip().endswith(('.', '!', '?')):
                risk = max(risk, 0.76)
            else:
                risk = max(risk, 0.82)
    if (
        len(last) <= 5
        and last.isalpha()
        and last_lower == last
        and len(words) <= 4
        and profile.explanation_ratio < 0.14
    ):
        risk = max(risk, 0.76)
    if t[-1] in '.!?' and risk < 0.55:
        body_words = full.strip().split()
        if len(body_words) >= 12 and len(words) <= 5:
            risk = max(risk, baseline.baseline([0.58, 1.0 - structural.sentence_completeness_mean]))
    if ev.coherence < 0.42 and raw.word_count < 8:
        risk = max(risk, 0.62)
    if len(words) >= 3:
        w1 = words[-2].strip('.,;:!?)]}').lower()
        w2 = words[-1].strip('.,;:!?)]}')
        prep = words[-3].strip('.,;:!?)]}').lower()
        if (
            w1 == 'the'
            and w2[:1].isupper()
            and len(w2) > 2
            and prep in {'at', 'to', 'into', 'toward', 'towards'}
            and len(words) <= 6
        ):
            risk = max(risk, 0.78)

    titleish = sum(1 for w in words if w[:1].isupper() and not w.isupper()) / max(len(words), 1)
    if titleish > 0.65 and profile.explanation_ratio < 0.10 and len(words) <= 4:
        risk = max(risk, 0.74)
    if (
        not t.rstrip().endswith(('.', '!', '?', '"', '\u201d', ')', ']', '}'))
        and titleish >= 0.50
        and len(words) <= 10
        and profile.explanation_ratio < 0.14
        and structural.sentence_completeness_mean < 0.72
    ):
        risk = max(risk, 0.80)
    if (
        not t.rstrip().endswith(('.', '!', '?', '"', '\u201d', ')', ']', '}'))
        and titleish >= 0.55
        and len(words) <= 12
        and profile.explanation_ratio < 0.12
        and ev.utility < 0.22
        and any(w[:1].isupper() for w in words[-3:])
    ):
        risk = max(risk, 0.78)
    if t.rstrip().endswith(',') and structural.sentence_completeness_mean < 0.74:
        risk = max(risk, 0.76)
    if '(' in t and t.count('(') > t.count(')'):
        risk = max(risk, 0.80)
    if len(words) >= 2:
        prep2 = words[-2].strip('.,;:!?)]}').lower()
        w_end = words[-1].strip('.,;:!?)]}')
        if prep2 == 'to' and w_end[:1].isupper() and len(words) <= 8:
            if not t.rstrip().endswith(('.', '!', '?')) and ',' not in words[-1]:
                risk = max(risk, 0.76)
        if prep2 == 'on' and w_end[:1].isupper() and (
            tail_raw.schedule_token_ratio > 0.06 or raw.schedule_token_ratio > 0.06
        ):
            risk = max(risk, 0.76)
        if prep2 in {'at', 'on', 'to'} and w_end[:1].isupper() and len(words) <= 8:
            if tail_raw.schedule_token_ratio > 0.05 or 'invite' in w_end.lower():
                risk = max(risk, 0.74)
    if 'invite' in t.lower() and not t.rstrip().endswith(('.', '!', '?')):
        risk = max(risk, 0.78)
    if (
        words[-1].strip('.,;:!?)]}').lower() in {
            'saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
        }
        and not t.rstrip().endswith(('.', '!', '?'))
        and len(words) <= 8
        and len(full) < 260
    ):
        risk = max(risk, 0.76)

    return min(1.0, risk)

def _code_tail_incomplete(text: str) -> float:
    if 'import ' not in text and 'def ' not in text:
        return 0.0
    t = text.rstrip()
    import re
    if re.search(r'=\s*\w+\.\s*$', t):
        return 0.86
    if t.count('(') > t.count(')'):
        return 0.80
    return 0.0

def continuation_incomplete(text: str) -> float:
    return analyze_completion_cached(text).incomplete_probability


def analyze_completion_cached(text: str) -> CompletionProfile:
    from indw.extract.core.context import get_document_context

    dctx = get_document_context()
    if dctx is None:
        return analyze_completion(text)
    return dctx.completion(text, lambda: analyze_completion(text))


def analyze_completion(text: str) -> CompletionProfile:
    if not text or not text.strip():
        return CompletionProfile(incomplete_probability=1.0)

    t = text.strip()
    head, tail = _tail_span(t)
    if not tail:
        tail = t
        head = ''

    tail_risk = _tail_continuation_risk(tail, full=t)
    terminal = _terminal_punct_valid(t)
    quote = _quote_open(t)
    structural = analyze_structure(t)
    ev = resolve_semantic_evidence(t)
    profile = compute_structure_profile(t, evidence=ev)
    baseline = AdaptiveBaselineEstimator()

    sentence = baseline.baseline([
        structural.sentence_completeness_mean,
        terminal,
        1.0 - tail_risk,
    ])
    paragraph = 1.0
    if '\n\n' in t:
        paras = [p.strip() for p in t.split('\n\n') if p.strip()]
        if paras:
            last_para = paras[-1]
            paragraph = analyze_completion(last_para).sentence
    quotation = quote
    dialogue = sentence
    if '"' in t or '\u201c' in t:
        head_q, tail_q = _tail_span(t)
        if tail_q and ('"' in tail_q or '\u201c' in tail_q):
            dialogue = min(dialogue, quotation * 0.85)
    topic = baseline.baseline([
        ev.coherence,
        profile.explanation_ratio,
        structural.information_density,
    ])
    narrative = baseline.baseline([
        sentence,
        profile.fact_ratio,
        ev.utility,
    ])
    explanation = baseline.baseline([
        profile.explanation_ratio,
        structural.paragraph_quality_mean,
        ev.coherence,
    ])
    overall = baseline.baseline([
        sentence * 0.34,
        paragraph * 0.14,
        quotation * 0.10,
        topic * 0.14,
        narrative * 0.14,
        explanation * 0.14,
    ])
    incomplete = min(1.0, max(
        tail_risk * 0.68,
        (1.0 - terminal) * 0.35,
        (1.0 - quote) * 0.45,
        (1.0 - sentence) * 0.40,
    ))
    if head and tail:
        head_ev = resolve_semantic_evidence(head[-min(400, len(head)):])
        tail_ev = resolve_semantic_evidence(tail)
        if tail_ev.utility < head_ev.utility * 0.55 and tail_risk > 0.45:
            incomplete = min(1.0, incomplete + 0.12)
    if not t.rstrip().endswith(('.', '!', '?', '"', '\u201d', ')', ']', '}')):
        incomplete = min(1.0, max(incomplete, tail_risk * 0.50 + 0.20))

    from indw.filter.refine.truncation import base_truncation_signal
    trunc_signal = base_truncation_signal(t)
    if trunc_signal >= 0.50:
        incomplete = min(1.0, max(incomplete, trunc_signal * 0.88))
    if t.rstrip().endswith((',', ':', '–', '—')) and len(t) > 80:
        incomplete = min(1.0, max(incomplete, 0.40))
    code_risk = _code_tail_incomplete(t)
    if code_risk > 0.0:
        incomplete = min(1.0, max(incomplete, code_risk * 0.90))

    return CompletionProfile(
        sentence=sentence,
        paragraph=paragraph,
        quotation=quotation,
        dialogue=dialogue,
        topic=topic,
        narrative=narrative,
        explanation=explanation,
        overall=overall,
        incomplete_probability=incomplete,
        signals={
            'tail_risk': tail_risk,
            'terminal': terminal,
            'quote': quote,
            'coherence': ev.coherence,
        },
    )

def discover_emit_boundaries(text: str) -> list[int]:
    if not text or not text.strip():
        return []
    t = text.strip()
    offsets: set[int] = set()
    end = len(t)
    while end > 0:
        boundary = _last_sentence_boundary(t[:end])
        if boundary <= 0 or boundary >= end:
            break
        offsets.add(boundary)
        end = boundary - 1
    for i, ch in enumerate(t):
        if ch == ',' and i > max(40, int(len(t) * 0.12)):
            offsets.add(i + 1)
    pos = 0
    while True:
        idx = t.find('\n\n', pos)
        if idx < 0:
            break
        offsets.add(idx + 2)
        pos = idx + 2
    return sorted(set(offsets))

def last_complete_boundary(
    text: str,
    *,
    min_chars: int = 40,
    min_completion: float = 0.58,
) -> int:
    if not text or not text.strip():
        return -1
    t = text.strip()
    candidates = discover_emit_boundaries(t)
    best = -1
    for off in candidates:
        if off <= 0 or off >= len(t):
            continue
        prefix = t[:off].strip()
        if len(prefix) < min_chars:
            continue
        comp = analyze_completion(prefix)
        if comp.overall >= min_completion and comp.incomplete_probability < 0.42:
            best = off
    return best

def trim_to_complete_boundary(
    text: str,
    *,
    min_chars: int = 40,
    min_retain_ratio: float = 0.35,
    min_completion: float = 0.58,
) -> tuple[str, int]:
    t = text.strip()
    if not t:
        return t, 0
    original = len(t)
    comp = analyze_completion(t)
    if comp.incomplete_probability < 0.38 and comp.overall >= min_completion:
        return t, 0

    boundary = last_complete_boundary(
        t, min_chars=max(min_chars, int(original * min_retain_ratio)), min_completion=min_completion,
    )
    if boundary > 0:
        trimmed = t[:boundary].strip()
        if len(trimmed) >= max(min_chars, int(original * min_retain_ratio)):
            return trimmed, original - len(trimmed)

    from indw.filter.refine.truncation import repair_truncation
    repaired, res = repair_truncation(t)
    if res.trimmed and len(repaired.strip()) >= max(min_chars, int(original * min_retain_ratio)):
        return repaired.strip(), original - len(repaired.strip())

    words = t.split()
    if words:
        last = words[-1].strip('\'",;:.!?)]}').lower()
        if last in _DANGLING_END or comp.incomplete_probability >= 0.50:
            parts = [p.strip() for p in t.split(',') if p.strip()]
            while len(parts) > 1:
                tail_words = parts[-1].split()
                tail_last = tail_words[-1].strip('\'",;:.!?)]}').lower() if tail_words else ''
                if tail_last not in _DANGLING_END and analyze_completion(parts[-1]).incomplete_probability < 0.48:
                    break
                parts.pop()
            if parts:
                candidate = ', '.join(parts).strip()
                removed = original - len(candidate)
                if removed > 0 and len(candidate) >= max(min_chars, int(original * min_retain_ratio)):
                    return candidate, removed

    head, tail = _tail_span(t)
    if head and comp.incomplete_probability >= 0.38:
        tail_comp = analyze_completion(tail)
        if tail_comp.signals.get('tail_risk', 0) >= 0.58 or tail_comp.incomplete_probability >= 0.48:
            candidate = head.strip().rstrip(',')
            if len(candidate) >= max(min_chars, int(original * min_retain_ratio)):
                return candidate, original - len(candidate)

    return t, 0

def repair_chunk_start(text: str) -> tuple[str, int]:
    t = text.strip()
    if not t:
        return t, 0
    original = len(t)
    removed = 0

    if t[0] in '"\'”»)]}':
        boundary = _last_sentence_boundary(t)
        if boundary > 0 and boundary < len(t) - 20:
            t = t[boundary:].strip()
            removed = original - len(t)

    words = t.split()
    if t and t[0].islower() and not t[0].startswith(('http', 'www.')):
        caps_run = 0
        for w in words[:6]:
            core = w.strip('\'",;:.!?)]}')
            if core and core[0].isupper() and not core.isupper():
                caps_run += 1
            else:
                break
        if caps_run >= 2 and analyze_completion(' '.join(words[caps_run:])).overall > analyze_completion(t).overall:
            candidate = ' '.join(words[caps_run:]).strip()
            if len(candidate) >= 30:
                removed = original - len(candidate)
                return candidate, removed

    if words and words[0][:1].islower() and not words[0].startswith(('http', 'www.')):
        for off in discover_emit_boundaries(t):
            if off <= 0 or off >= len(t):
                continue
            candidate = t[off:].strip()
            if len(candidate) < 30:
                continue
            if candidate.split()[0][:1].isupper() or candidate[0] in '"\'(':
                head_comp = analyze_completion(t[:off])
                cand_comp = analyze_completion(candidate)
                if (
                    head_comp.incomplete_probability > 0.45
                    or cand_comp.overall > analyze_completion(t).overall + 0.04
                ):
                    removed = original - len(candidate)
                    return candidate, removed

    if t.count('"') % 2 != 0:
        quote_at = max(t.rfind('"'), t.rfind('\u201c'))
        if quote_at > 0:
            after = t[quote_at + 1:].strip()
            if after and analyze_completion(after).overall > analyze_completion(t).overall:
                removed = original - len(after)
                return after, removed

    return t, removed

def lookahead_complete_within(text: str, *, window_chars: int = 320) -> bool:
    if not text or len(text) < 40:
        return True
    window = text[:window_chars]
    comp = analyze_completion(window)
    if comp.incomplete_probability < 0.45:
        return True
    boundary = last_complete_boundary(window, min_chars=30, min_completion=0.55)
    return boundary > 0

def boundary_confidence_from_completion(text: str) -> float:
    comp = analyze_completion(text)
    return min(1.0, comp.overall * 0.58 + (1.0 - comp.incomplete_probability) * 0.42)

def boundary_before_orphan_quote(text: str) -> int:
    if text.count('"') % 2 == 0 and text.count("'") % 2 == 0:
        return _last_sentence_boundary(text)
    quote_at = -1
    for i, ch in enumerate(text):
        if ch in '"\'':
            quote_at = i
    if quote_at <= 0:
        return _last_sentence_boundary(text)
    head_boundary = _last_sentence_boundary(text[:quote_at])
    if head_boundary > max(40, int(len(text) * 0.15)):
        return head_boundary
    return _last_sentence_boundary(text)
