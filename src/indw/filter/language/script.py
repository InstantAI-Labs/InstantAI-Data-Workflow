from __future__ import annotations

import re

import unicodedata

from dataclasses import dataclass, field

from indw.filter.language.script_table import script_family_spec, script_for_codepoint

_DIGIT = re.compile(r'\d')

_TRANSLIT_HINT = re.compile(

    r'[A-Za-z]{2,}[\u0900-\u097F\u0600-\u06FF\u4E00-\u9FFF]'

)

@dataclass

class ScriptProfile:

    char_len: int = 0

    script_fractions: dict[str, float] = field(default_factory=dict)

    dominant_script: str = 'other'

    script_count: int = 0

    mixed_script_score: float = 0.0

    rtl_ratio: float = 0.0

    fragmentation_risk: float = 0.0

    transliteration_score: float = 0.0

    unicode_instability: float = 0.0

    grapheme_density: float = 0.0

    punctuation_density: float = 0.0

    whitespace_anomaly: float = 0.0

    def bucket_key(self, mapping: dict[str, str] | None = None) -> str:

        from indw.filter.language.script_table import script_bucket

        return script_bucket(self.dominant_script, mapping)

@dataclass(frozen=True)

class ScriptTextScan:

    profile: ScriptProfile

    segments: tuple[tuple[str, str], ...] = ()

def _segment_min_chars(script: str, min_chars: int) -> int:

    spec = script_family_spec(script)

    if spec is not None and not spec.whitespace_segmented:

        return 1

    return min_chars

def _profile_from_counts(

    text: str,

    *,

    counts: dict[str, int],

    rtl: int,

    punct: int,

    space: int,

    unstable: int,

    graphemes: int,

    latin_runs: int,

    n: int,

) -> ScriptProfile:

    if n <= 0:

        return ScriptProfile(char_len=len(text))

    fracs = {k: v / n for k, v in counts.items()}

    dominant = max(fracs, key=fracs.get)

    active = sum(1 for v in fracs.values() if v >= 0.05)

    mixed = min(1.0, max(0.0, (active - 1) * 0.28)) if active > 1 else 0.0

    translit = 1.0 if _TRANSLIT_HINT.search(text) else 0.0

    if latin_runs and dominant not in ('latin', 'other'):

        translit = max(translit, min(1.0, latin_runs / max(n, 1)))

    punct_risk = min(1.0, punct / max(n, 1) * 8.0) * 0.25

    unstable_risk = min(1.0, unstable / max(n, 1) * 20.0) * 0.2

    frag = mixed * 0.55 + punct_risk + unstable_risk

    ws_anom = 0.0

    space_ratio = space / max(len(text), 1)

    if dominant in ('cjk', 'hiragana_katakana') and space_ratio > 0.2:

        ws_anom = min(1.0, space_ratio * 3.0)

    return ScriptProfile(

        char_len=len(text),

        script_fractions=fracs,

        dominant_script=dominant,

        script_count=active,

        mixed_script_score=mixed,

        rtl_ratio=rtl / n,

        fragmentation_risk=min(1.0, frag),

        transliteration_score=translit,

        unicode_instability=min(1.0, unstable / max(n, 1)),

        grapheme_density=graphemes / n,

        punctuation_density=punct / n,

        whitespace_anomaly=ws_anom,

    )

def scan_script_text(

    text: str,

    *,

    segment_min_chars: int | None = None,

) -> ScriptTextScan:

    if not text:

        return ScriptTextScan(ScriptProfile(), ())

    counts: dict[str, int] = {}

    rtl = 0

    punct = 0

    space = 0

    unstable = 0

    graphemes = 0

    latin_runs = 0

    in_latin_run = False

    n = 0

    script_cache: dict[int, str] = {}

    collect_segments = segment_min_chars is not None

    min_chars = int(segment_min_chars or 0)

    segments: list[tuple[str, str]] = []

    seg_buf: list[str] = []

    seg_current = ''

    def fam_for(ch: str) -> str:

        cp = ord(ch)

        fam = script_cache.get(cp)

        if fam is None:

            fam = script_for_codepoint(cp)

            script_cache[cp] = fam

        return fam

    for ch in text:

        if ch.isspace():

            space += 1

            in_latin_run = False

            if collect_segments and seg_buf:

                chunk = ''.join(seg_buf).strip()

                floor = _segment_min_chars(seg_current, min_chars)

                if len(chunk) >= floor and seg_current:

                    segments.append((chunk, seg_current))

                seg_buf = []

                seg_current = ''

            continue

        n += 1

        is_latin = ('A' <= ch <= 'Z') or ('a' <= ch <= 'z')

        if is_latin:

            if not in_latin_run:

                latin_runs += 1

            in_latin_run = True

        else:

            in_latin_run = False

        fam = fam_for(ch)

        counts[fam] = counts.get(fam, 0) + 1

        cat = unicodedata.category(ch)

        if cat.startswith('P'):

            punct += 1

        if unicodedata.combining(ch):

            graphemes += 1

        cp = ord(ch)

        if cat == 'Cn' or cp == 0xFFFD:

            unstable += 1

        spec = script_family_spec(fam)

        if spec is not None and spec.rtl:

            rtl += 1

        if not collect_segments:

            continue

        if seg_current and fam != seg_current:

            chunk = ''.join(seg_buf).strip()

            floor = _segment_min_chars(seg_current, min_chars)

            if len(chunk) >= floor:

                segments.append((chunk, seg_current))

            seg_buf = [ch]

            seg_current = fam

            continue

        seg_current = fam or seg_current

        seg_buf.append(ch)

    profile = _profile_from_counts(

        text,

        counts=counts,

        rtl=rtl,

        punct=punct,

        space=space,

        unstable=unstable,

        graphemes=graphemes,

        latin_runs=latin_runs,

        n=n,

    )

    if collect_segments:

        tail = ''.join(seg_buf).strip()

        floor = _segment_min_chars(seg_current, min_chars)

        if len(tail) >= floor and seg_current:

            segments.append((tail, seg_current))

    return ScriptTextScan(profile, tuple(segments))

def analyze_script_profile(text: str) -> ScriptProfile:

    return scan_script_text(text).profile
