from __future__ import annotations

import re
from typing import Any

from indw.clean.gate.evaluate import (
    REPLACEMENT_CHAR,
    disambig_list_line_count,
    disambiguation_match,
    document_gate_raw,
    html_dom_pattern_count,
    html_tag_count,
    keyboard_smash_hits,
    resolve_gate_policy,
)
from indw.clean.meta.foundation import is_metadata_only_document
from indw.clean.document.normalize import meaningful_char_count
from indw.clean.document.patterns import _CONTROL, _UI_LINE
from indw.filter.score.artifacts import _ocr_corruption_score
from indw.clean.artifact.evidence import PopulationAdaptiveScaler

_ERROR_PAGE = re.compile(
    r'(?im)^(?:'
    r'(?:404|403|500|502|503)\s+(?:not\s+found|forbidden|error|internal\s+server\s+error|bad\s+gateway)|'
    r'page\s+not\s+found|access\s+denied|login\s+required|sign\s+in\s+to\s+continue'
    r')'
)
_LOGIN_TEMPLATE = re.compile(
    r'(?im)(?:'
    r'(?:sign\s+in|log\s+in|create\s+account|forgot\s+password).{0,120}(?:password|username|email)|'
    r'(?:password|username|email).{0,120}(?:sign\s+in|log\s+in)'
    r')'
)


def run_stage0_content_filters(
    text: str,
    *,
    meaningful_chars: int,
    raw: Any | None = None,
    pol: Any | None = None,
    normalized: bool = True,
) -> str | None:
    if not text:
        return 'empty'
    t = text if normalized else text.strip()
    if not t:
        return 'empty'
    raw = raw or document_gate_raw(t)
    pol = pol or resolve_gate_policy()
    short_bound = PopulationAdaptiveScaler.short_doc_boundary(raw)
    reason = evaluate_stage0_deterministic(
        t,
        raw=raw,
        meaningful_chars=meaningful_chars,
        pol=pol,
        short_bound=short_bound,
        normalized=True,
    )
    if reason:
        return reason
    return evaluate_stage0_structural(
        t,
        raw=raw,
        meaningful_chars=meaningful_chars,
        normalized=True,
    )


def evaluate_stage0_deterministic(
    text: str,
    *,
    raw: Any | None = None,
    meaningful_chars: int | None = None,
    pol: Any | None = None,
    short_bound: float | None = None,
    normalized: bool = False,
) -> str | None:
    if not text:
        return 'empty'
    t = text if normalized else text.strip()
    if not t:
        return 'empty'
    if raw is None:
        raw = document_gate_raw(t)
    meaningful = meaningful_chars if meaningful_chars is not None else meaningful_char_count(t)
    pol = pol or resolve_gate_policy()
    short_bound = (
        short_bound if short_bound is not None
        else PopulationAdaptiveScaler.short_doc_boundary(raw)
    )

    reason = _corruption_reject(t, raw=raw, meaningful=meaningful, pol=pol, short_bound=short_bound)
    if reason:
        return reason
    reason = _html_reject(t, raw=raw, meaningful=meaningful, pol=pol, short_bound=short_bound)
    if reason:
        return reason
    reason = _navigation_reject(t, raw=raw, meaningful=meaningful, pol=pol, short_bound=short_bound)
    if reason:
        return reason
    reason = _disambiguation_reject(t, raw=raw, meaningful=meaningful, pol=pol, short_bound=short_bound)
    if reason:
        return reason
    if is_metadata_only_document(t):
        return 'metadata_only'
    return None


def evaluate_stage0_structural(
    text: str,
    *,
    raw: Any | None = None,
    meaningful_chars: int | None = None,
    normalized: bool = False,
) -> str | None:
    if not text:
        return None
    t = text if normalized else text.strip()
    if not t:
        return None
    if raw is None:
        raw = document_gate_raw(t)
    meaningful = meaningful_chars if meaningful_chars is not None else meaningful_char_count(t)
    n = len(t)

    repl = t.count(REPLACEMENT_CHAR)
    if repl >= max(8, n // 200) and _ocr_corruption_score(t) >= 0.78:
        return 'ocr_noisy'

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        error_hits = sum(1 for ln in lines[:12] if _ERROR_PAGE.search(ln))
        substantive = raw.word_count
        if (
            error_hits >= 3
            and meaningful < 600
            and substantive < 80
            and error_hits / len(lines[:12]) >= 0.6
        ):
            return 'error_page'

    ui_ratio = _line_pattern_ratio(lines, (_UI_LINE,))
    if ui_ratio >= 0.82 and meaningful < 2500 and raw.nav_line_ratio >= 0.55:
        return 'template_page'

    if _LOGIN_TEMPLATE.search(t[:4096]) and meaningful < 900 and ui_ratio >= 0.45:
        return 'login_page'

    return None


def _line_pattern_ratio(lines: list[str], patterns: tuple[re.Pattern[str], ...]) -> float:
    if not lines:
        return 1.0
    hits = sum(1 for ln in lines if any(p.match(ln) for p in patterns))
    return hits / len(lines)


def _corruption_reject(
    text: str,
    *,
    raw: Any,
    meaningful: int,
    pol: Any,
    short_bound: float,
) -> str | None:
    n = len(text)
    repl = text.count(REPLACEMENT_CHAR)
    ctrl = len(_CONTROL.findall(text))
    alpha = sum(c.isalpha() for c in text) / max(n, 1)
    smash = keyboard_smash_hits(text)
    if raw.fence_char_ratio > 0 or raw.structured_line_ratio > 0.08:
        return None
    corruption = min(
        1.0,
        repl / max(n / pol.repl_char_divisor, 1)
        + ctrl / max(n / pol.ctrl_char_divisor, 1)
        + smash * pol.smash_hit_weight,
    )
    alpha_floor = max(pol.alpha_floor, 0.35)
    if repl >= pol.max_replacement_chars and (
        alpha < max(alpha_floor, 0.45)
        or smash >= pol.min_keyboard_smash_hits
        or repl / max(meaningful, 1) >= 0.15
    ):
        return 'extreme_corruption'
    if repl >= 8 and repl / max(meaningful, 1) >= 0.12 and smash >= 1 and meaningful < short_bound * 2:
        return 'extreme_corruption'
    if (
        smash >= pol.min_keyboard_smash_hits
        and alpha < alpha_floor
        and meaningful < short_bound
        and corruption > 0.45
    ):
        return 'extreme_corruption'
    return None


def _html_reject(
    text: str,
    *,
    raw: Any,
    meaningful: int,
    pol: Any,
    short_bound: float,
) -> str | None:
    n = len(text)
    html_tags = html_tag_count(text)
    dom_hits = html_dom_pattern_count(text)
    html_ratio = min(1.0, (html_tags * pol.html_tag_weight + dom_hits * pol.html_dom_weight) / max(n, 1))
    nav_ratio = raw.nav_line_ratio
    tag_floor = max(pol.html_min_tags_floor, 20)
    if html_ratio >= 0.35 and html_tags >= tag_floor and dom_hits >= 8:
        return 'html_dump'
    if (
        html_tags >= tag_floor
        and html_ratio >= 0.10
        and nav_ratio >= 0.40
        and raw.word_count < 500
        and meaningful < short_bound * 2
    ):
        return 'html_dump'
    return None


def _navigation_reject(
    text: str,
    *,
    raw: Any,
    meaningful: int,
    pol: Any,
    short_bound: float,
) -> str | None:
    nav_ratio = raw.nav_line_ratio
    if (
        nav_ratio >= 0.70
        and meaningful < short_bound * pol.nav_meaningful_short_multiplier
        and raw.word_count < 600
    ):
        return 'navigation_boilerplate'
    return None


def _disambiguation_reject(
    text: str,
    *,
    raw: Any,
    meaningful: int,
    pol: Any,
    short_bound: float,
) -> str | None:
    del raw
    if not disambiguation_match(text):
        return None
    list_lines = disambig_list_line_count(text)
    bound = short_bound * max(pol.disambig_short_bound_line_multiplier, list_lines) * 0.75
    if list_lines >= pol.disambig_min_list_lines and meaningful < bound:
        return 'disambiguation_page'
    return None


_GATE_POLICY: Any = None
_MAX_DOC_CHARS: int | None = None


def bind_fast_stage0_worker(gate: Any) -> None:
    global _GATE_POLICY, _MAX_DOC_CHARS
    from indw.clean.gate.evaluate import resolve_gate_policy
    from indw.schedule.monitor.budget import resolve_doc_max_chars
    from indw.schedule.read.gates import warmup_language_detector

    _GATE_POLICY = resolve_gate_policy()
    _MAX_DOC_CHARS = resolve_doc_max_chars()
    warmup_language_detector(gate)


def worker_gate_policy() -> Any:
    if _GATE_POLICY is not None:
        return _GATE_POLICY
    from indw.clean.gate.evaluate import resolve_gate_policy
    return resolve_gate_policy()


def worker_doc_max_chars() -> int:
    if _MAX_DOC_CHARS is not None:
        return _MAX_DOC_CHARS
    from indw.schedule.monitor.budget import resolve_doc_max_chars
    return resolve_doc_max_chars()
