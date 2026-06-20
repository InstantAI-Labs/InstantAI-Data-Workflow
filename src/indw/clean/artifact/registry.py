from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from indw.clean.document.patterns import (
    _ACK_LINE,
    _CODE_FENCE,
    _HTML_SCRIPT_STYLE,
    _HTML_TAG,
    _METADATA_LINE,
    _UI_LINE,
)

@dataclass
class ArtifactPattern:
    name: str
    pattern: re.Pattern[str]
    category: str
    line_only: bool = True

@dataclass
class ArtifactPatternRegistry:
    patterns: list[ArtifactPattern] = field(default_factory=list)
    _discovery: Any = field(default=None, repr=False)

    @classmethod
    def production(cls) -> ArtifactPatternRegistry:
        reg = cls()
        for p in (
            ArtifactPattern('ui_line', _UI_LINE, 'ui', line_only=True),
            ArtifactPattern('metadata_line', _METADATA_LINE, 'metadata', line_only=True),
            ArtifactPattern('ack_line', _ACK_LINE, 'forum', line_only=True),
            ArtifactPattern('html_tag', _HTML_TAG, 'html', line_only=False),
            ArtifactPattern('html_script_style', _HTML_SCRIPT_STYLE, 'html', line_only=False),
            ArtifactPattern('code_fence', _CODE_FENCE, 'code', line_only=False),
        ):
            reg.patterns.append(p)
        reg._register_extended()
        return reg

    def _discovery_engine(self) -> Any:
        if self._discovery is None:
            from indw.clean.artifact.discovery_engine import get_discovery_engine
            self._discovery = get_discovery_engine()
        return self._discovery

    def _register_extended(self) -> None:
        extended = (
            ('cookie_notice', r'(?i)\b(?:accept\s+(?:all\s+)?cookies|cookie\s+(?:policy|settings|preferences))\b', 'ui'),
            ('login_prompt', r'(?i)\b(?:sign\s*in|log\s*in|register\s+to\s+(?:view|comment|reply))\b', 'ui'),
            ('share_button', r'(?i)\b(?:share\s+(?:this|on)|tweet\s+this|share\s+on\s+(?:facebook|twitter|linkedin))\b', 'ui'),
            ('forum_nav', r'(?i)\b(?:jump\s+to\s*:\s*navigation|view\s+topic|reply\s+with\s+quote)\b', 'forum'),
            ('forum_profile', r'(?i)\b(?:originally\s+posted\s+by|reputation\s*:|username\s*:)\b', 'forum'),
            ('view_counter', r'(?i)\b\d+(?:\.\d+)?[km]?\s+(?:views?|replies?|comments?|likes?)\b', 'metadata'),
            ('copyright_block', r'(?i)\b(?:all\s+rights\s+reserved|copyright\s*(?:©|\(c\)))\b', 'legal'),
            ('repo_path', r'(?i)(?:^|[\s/])(?:[\w.-]+/)+[\w.-]+\.(?:py|js|ts|go|rs|java|adb|ads)\b', 'repo'),
            ('commit_id', r'(?i)\bcommit\s+[0-9a-f]{7,40}\b', 'repo'),
            ('wiki_nav', r'(?i)\bjump\s+to\s*:\s*navigation\s*,\s*search\b', 'ui'),
        )
        for name, expr, category in extended:
            self.patterns.append(
                ArtifactPattern(name, re.compile(expr), category, line_only=False)
            )

    def match_line(self, line: str) -> list[str]:
        stripped = line.strip()
        if not stripped:
            return []
        hits: list[str] = []
        for entry in self.patterns:
            if not entry.line_only:
                continue
            if entry.pattern.match(stripped):
                hits.append(entry.name)
        return hits

    def scan_text(self, text: str) -> dict[str, int]:
        learned = self._discovery_engine().scan_text(text)
        if learned:
            return learned
        counts: dict[str, int] = {}
        for entry in self.patterns:
            n = len(entry.pattern.findall(text))
            if n:
                counts[entry.name] = counts.get(entry.name, 0) + n
        lines = [ln for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            for name in self.match_line(ln):
                counts[name] = counts.get(name, 0) + 1
        return counts

    def artifact_ratio(self, text: str) -> float:
        from indw.clean.artifact.engine import get_artifact_engine
        eng = get_artifact_engine()
        disc = self._discovery_engine().document_artifact_ratio(text)
        combined = eng.artifact_ratio(text)
        if disc > 0:
            return max(disc, combined)
        return combined

    def audit_flags(self, text: str) -> list[str]:
        flags = self._discovery_engine().audit_flags(text)
        if flags:
            return flags
        counts = self.scan_text(text)
        out: list[str] = []
        if any(k in counts for k in ('cookie_notice', 'login_prompt', 'share_button', 'wiki_nav')):
            out.append('website_artifact')
        if any(k in counts for k in ('forum_nav', 'forum_profile', 'ack_line')):
            out.append('forum_junk')
        if any(k in counts for k in ('repo_path', 'commit_id')):
            out.append('repo_metadata')
        if counts.get('copyright_block', 0) > 0:
            out.append('copyright_notice')
        return out

    def compile_audit_pattern(self) -> re.Pattern[str]:
        parts = [p.pattern.pattern for p in self.patterns if p.category in ('ui', 'forum')]
        if not parts:
            return re.compile(r'(?!x)x')
        return re.compile('|'.join(f'(?:{part})' for part in parts[:12]), re.I)

_REGISTRY: ArtifactPatternRegistry | None = None

def get_artifact_registry() -> ArtifactPatternRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ArtifactPatternRegistry.production()
    return _REGISTRY

def line_is_artifact(line: str, *, registry: ArtifactPatternRegistry | None = None) -> bool:
    reg = registry or get_artifact_registry()
    eng = reg._discovery_engine()
    entry = eng.registry.lookup(line.strip(), eng.accumulator)
    if entry and entry.promoted and entry.artifact_confidence >= 0.55:
        return True
    return bool(reg.match_line(line))
