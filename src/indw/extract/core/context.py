from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from indw.clean.artifact.evidence_cache import text_fingerprint

T = TypeVar('T')

_tls = threading.local()
_BOUND: DocumentExecutionContext | None = None
_EMIT_UNIT_MISS = object()


@dataclass
class DocumentExecutionContext:
    normalized_text: str = ''
    pci_fp: dict[str, Any] | None = None
    source: str = ''
    gate_raw: Any | None = None
    _completion: dict[tuple[int, bytes], Any] = field(default_factory=dict)
    _forum: dict[tuple[int, bytes], Any] = field(default_factory=dict)
    _scaffold_stripped: dict[tuple[int, bytes], str] = field(default_factory=dict)
    _pub_spans: dict[tuple[int, bytes], list[Any]] = field(default_factory=dict)
    _section_evidence: dict[tuple[int, bytes], Any] = field(default_factory=dict)
    _content_value: dict[tuple[int, bytes], Any] = field(default_factory=dict)
    _analysis_bundle: dict[tuple[int, bytes], Any] = field(default_factory=dict)
    _structure: dict[tuple[int, bytes], Any] = field(default_factory=dict)
    _structure_profile: dict[tuple[int, bytes], Any] = field(default_factory=dict)
    _layout: dict[tuple[int, bytes], Any] = field(default_factory=dict)
    _terminal: dict[tuple[int, bytes], float] = field(default_factory=dict)
    _clean_unit: dict[tuple[Any, ...], str] = field(default_factory=dict)
    _emit_unit: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    _document_structure: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def document_structure(self, cache_key: tuple[Any, ...], compute: Callable[[], T]) -> T:
        hit = self._document_structure.get(cache_key)
        if hit is not None:
            return hit
        result = compute()
        self._document_structure[cache_key] = result
        return result

    @staticmethod
    def _key(text: str) -> tuple[int, bytes] | None:
        if not text:
            return None
        return text_fingerprint(text)

    def remember_scaffold_stripped(self, text: str, stripped: str) -> None:
        key = self._key(text)
        if key is not None and stripped:
            self._scaffold_stripped[key] = stripped

    def scaffold_stripped_for(self, text: str) -> str:
        key = self._key(text)
        if key is None:
            return ''
        return self._scaffold_stripped.get(key, '')

    def publication_spans(self, text: str, compute: Callable[[], list[Any]]) -> list[Any]:
        key = self._key(text)
        if key is None:
            return compute()
        hit = self._pub_spans.get(key)
        if hit is not None:
            return hit
        result = compute()
        self._pub_spans[key] = result
        return result

    def section_evidence(self, text: str, compute: Callable[[], T]) -> T:
        key = self._key(text)
        if key is None:
            return compute()
        hit = self._section_evidence.get(key)
        if hit is not None:
            self._record_semantic('semantic_evidence', key, cache_hit=True)
            return hit
        bundle_hit = self._analysis_bundle.get(key)
        if bundle_hit is not None:
            try:
                ev = bundle_hit.evidence(text)
                self._section_evidence[key] = ev
                self._record_semantic('semantic_evidence', key, cache_hit=True)
                return ev
            except Exception:
                pass
        result = compute()
        self._section_evidence[key] = result
        self._record_semantic('semantic_evidence', key, cache_hit=False)
        return result

    @staticmethod
    def _record_semantic(op: str, key: tuple[int, bytes], *, cache_hit: bool) -> None:
        try:
            from indw.extract.core.profile import ke_record
            ke_record(op, dedupe_key=key, cache_hit=cache_hit)
        except Exception:
            pass

    def content_value(self, text: str, compute: Callable[[], T]) -> T:
        key = self._key(text)
        if key is None:
            return compute()
        hit = self._content_value.get(key)
        if hit is not None:
            return hit
        result = compute()
        self._content_value[key] = result
        return result

    def analysis_bundle(self, text: str, compute: Callable[[], T]) -> T:
        key = self._key(text)
        if key is None:
            return compute()
        hit = self._analysis_bundle.get(key)
        if hit is not None:
            self._record_semantic('analysis_bundle', key, cache_hit=True)
            return hit
        result = compute()
        self._analysis_bundle[key] = result
        self._record_semantic('analysis_bundle', key, cache_hit=False)
        if hasattr(result, 'evidence'):
            try:
                self._section_evidence[key] = result.evidence(text)
            except Exception:
                pass
        return result

    def structure_analysis(self, text: str, compute: Callable[[], T]) -> T:
        key = self._key(text)
        if key is None:
            return compute()
        hit = self._structure.get(key)
        if hit is not None:
            self._record_semantic('structure_analysis', key, cache_hit=True)
            return hit
        result = compute()
        self._structure[key] = result
        self._record_semantic('structure_analysis', key, cache_hit=False)
        return result

    def structure_profile(self, text: str, compute: Callable[[], T]) -> T:
        key = self._key(text)
        if key is None:
            return compute()
        hit = self._structure_profile.get(key)
        if hit is not None:
            self._record_semantic('structure_profile', key, cache_hit=True)
            return hit
        result = compute()
        self._structure_profile[key] = result
        self._record_semantic('structure_profile', key, cache_hit=False)
        return result

    def layout(self, text: str, compute: Callable[[], T]) -> T:
        key = self._key(text)
        if key is None:
            return compute()
        hit = self._layout.get(key)
        if hit is not None:
            self._record_semantic('layout', key, cache_hit=True)
            return hit
        result = compute()
        self._layout[key] = result
        self._record_semantic('layout', key, cache_hit=False)
        return result

    def completion(self, text: str, compute: Callable[[], T]) -> T:
        key = self._key(text)
        if key is None:
            return compute()
        hit = self._completion.get(key)
        if hit is not None:
            return hit
        result = compute()
        self._completion[key] = result
        return result

    def forum_structure(self, text: str, compute: Callable[[], T]) -> T:
        key = self._key(text)
        if key is None:
            return compute()
        hit = self._forum.get(key)
        if hit is not None:
            return hit
        result = compute()
        self._forum[key] = result
        return result

    def terminal_boundary(self, text: str, compute: Callable[[], float]) -> float:
        key = self._key(text)
        if key is None:
            return compute()
        hit = self._terminal.get(key)
        if hit is not None:
            return hit
        result = compute()
        self._terminal[key] = result
        return result

    def _unit_input_key(
        self,
        text: str,
        *,
        role: str,
        forum_strip: bool,
        scaffold_stripped: str,
    ) -> tuple[Any, ...] | None:
        fp = self._key(text)
        if fp is None:
            return None
        sc = self._key(scaffold_stripped) if scaffold_stripped else ()
        return (fp, role, forum_strip, sc)

    def clean_unit(
        self,
        text: str,
        *,
        role: str,
        forum_strip: bool,
        scaffold_stripped: str,
        compute: Callable[[], str],
    ) -> str:
        ck = self._unit_input_key(
            text, role=role, forum_strip=forum_strip, scaffold_stripped=scaffold_stripped,
        )
        if ck is None:
            return compute()
        hit = self._clean_unit.get(ck)
        if hit is not None:
            return hit
        result = compute()
        self._clean_unit[ck] = result
        return result

    def emit_unit(
        self,
        text: str,
        *,
        role: str,
        forum_strip: bool,
        min_chars: int,
        preserve_code_fences: bool,
        scaffold_stripped: str,
        compute: Callable[[], T | None],
    ) -> T | None:
        base = self._unit_input_key(
            text, role=role, forum_strip=forum_strip, scaffold_stripped=scaffold_stripped,
        )
        if base is None:
            return compute()
        ck = (*base, min_chars, preserve_code_fences)
        hit = self._emit_unit.get(ck)
        if hit is not None:
            if hit is _EMIT_UNIT_MISS:
                return None
            return hit
        result = compute()
        self._emit_unit[ck] = _EMIT_UNIT_MISS if result is None else result
        return result

    def clear(self) -> None:
        self._completion.clear()
        self._forum.clear()
        self._scaffold_stripped.clear()
        self._pub_spans.clear()
        self._section_evidence.clear()
        self._content_value.clear()
        self._analysis_bundle.clear()
        self._structure.clear()
        self._structure_profile.clear()
        self._layout.clear()
        self._terminal.clear()
        self._clean_unit.clear()
        self._emit_unit.clear()
        self._document_structure.clear()
        self.normalized_text = ''
        self.pci_fp = None
        self.source = ''


def bind_document_context(ctx: DocumentExecutionContext | None) -> None:
    global _BOUND
    _BOUND = ctx
    _tls.ctx = ctx


def get_document_context() -> DocumentExecutionContext | None:
    ctx = getattr(_tls, 'ctx', None)
    if ctx is not None:
        return ctx
    return _BOUND


def clear_document_context(*, trim_caches: bool | None = None) -> None:
    ctx = getattr(_tls, 'ctx', None) or _BOUND
    if ctx is not None:
        ctx.clear()
    bind_document_context(None)
    if trim_caches is None:
        from indw.schedule.config.resolve import env_flag
        trim_caches = env_flag('INSTANT_TRIM_DOC_CACHES', False)
    if trim_caches:
        try:
            from indw.clean.artifact.evidence_cache import trim_document_caches, set_session_cache_boost
            set_session_cache_boost(1)
            trim_document_caches()
        except Exception:
            pass
