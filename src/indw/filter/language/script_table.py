from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from orchestration.resolver.immutable import thaw
from orchestration.resolver.refs import ConfigRef
from orchestration.resolver.resolver import Resolver

@dataclass(frozen=True)
class ScriptFamilySpec:
    name: str
    rtl: bool = False
    whitespace_segmented: bool = True
    grapheme_sensitive: bool = False
    mixed_script_common: bool = False

_SCRIPT_SPECS: dict[str, ScriptFamilySpec] = {}
SCRIPT_FAMILIES: tuple[ScriptFamilySpec, ...] = ()
SCRIPT_BUCKETS: dict[str, str] = {}

def _normalize_script_name(name: str) -> str:
    n = str(name or '').strip().lower().replace('-', '_').replace(' ', '_')
    if not n:
        return 'other'
    if n.startswith('latin_'):
        return 'latin'
    if n.startswith('devanagari_'):
        return 'devanagari'
    if n.startswith('arabic_'):
        return 'arabic'
    if n.startswith('hebrew_'):
        return 'hebrew'
    if n.startswith('cyrillic_'):
        return 'cyrillic'
    if n.startswith('hangul_'):
        return 'hangul'
    if n.startswith('hiragana_') or n.startswith('katakana_'):
        return 'hiragana_katakana'
    if n.startswith('cjk_'):
        return 'cjk'
    if 'hangul' in n:
        return 'hangul'
    if 'hiragana' in n or 'katakana' in n:
        return 'hiragana_katakana'
    if 'ideograph' in n or 'han' in n or 'cjk' in n:
        return 'cjk'
    if 'emoji' in n or 'emoticon' in n:
        return 'emoji'
    return n

def _infer_spec(script: str, ch: str) -> ScriptFamilySpec:
    bidi = unicodedata.bidirectional(ch)
    rtl = bidi in {'R', 'AL', 'AN'}
    ws = script not in {'cjk', 'hiragana_katakana'}
    return ScriptFamilySpec(
        name=script,
        rtl=rtl,
        whitespace_segmented=ws,
        grapheme_sensitive=(script in {'devanagari', 'arabic'}),
        mixed_script_common=(script in {'cjk', 'hiragana_katakana'}),
    )

def _load_script_manifest() -> None:
    ref = os.environ.get('INSTANT_LANGUAGE_MANIFEST')
    if not ref:
        return
    try:
        resolver = Resolver.default()
        cfg_ref = ConfigRef(kind='language_manifest', id=str(ref))
        raw = thaw(resolver.resolve(cfg_ref).raw)
    except Exception:
        return

    families = raw.get('script_families') or []
    if isinstance(families, list):
        for f in families:
            if not isinstance(f, dict):
                continue
            name = _normalize_script_name(f.get('name'))
            default_ws = name not in {'cjk', 'hiragana_katakana'}
            _SCRIPT_SPECS[name] = ScriptFamilySpec(
                name=name,
                rtl=bool(f.get('rtl', False)),
                whitespace_segmented=bool(
                    f.get('whitespace_segmented', default_ws)
                ),
                grapheme_sensitive=bool(f.get('grapheme_sensitive', False)),
                mixed_script_common=bool(f.get('mixed_script_common', False)),
            )

    buckets = raw.get('bucket_map') or {}
    if isinstance(buckets, dict):
        for k, v in buckets.items():
            SCRIPT_BUCKETS[_normalize_script_name(str(k))] = str(v)

def script_for_codepoint(cp: int) -> str:
    global SCRIPT_FAMILIES
    ch = chr(int(cp))
    cat = unicodedata.category(ch)
    if cat[0] in {'N', 'P', 'S'} and cp < 0x1F000:
        script = 'common'
    elif cat == 'So' and cp >= 0x1F000:
        script = 'emoji'
    else:
        uname = unicodedata.name(ch, '')
        script = _normalize_script_name(uname if uname else 'other')

    if script not in _SCRIPT_SPECS:
        _SCRIPT_SPECS[script] = _infer_spec(script, ch)
        SCRIPT_BUCKETS.setdefault(script, script)
        SCRIPT_FAMILIES = tuple(_SCRIPT_SPECS.values())
    return script

def script_bucket(script: str, mapping: dict[str, str] | None = None) -> str:
    mp = mapping or SCRIPT_BUCKETS
    key = _normalize_script_name(script)
    return mp.get(key, key)

def script_family_spec(name: str) -> ScriptFamilySpec | None:
    return _SCRIPT_SPECS.get(_normalize_script_name(name))

def iter_script_specs() -> Iterable[ScriptFamilySpec]:
    return tuple(_SCRIPT_SPECS.values())

_load_script_manifest()
SCRIPT_FAMILIES = tuple(_SCRIPT_SPECS.values())
