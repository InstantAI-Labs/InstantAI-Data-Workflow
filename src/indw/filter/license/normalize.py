from __future__ import annotations

import re
from typing import Optional

from indw.filter.license.schema import LICENSE_CATEGORIES

_SPDX_ALIASES: dict[str, str] = {
    'publicdomain': 'Public Domain',
    'public domain': 'Public Domain',
    'public-domain': 'Public Domain',
    'pd': 'Public Domain',
    'cc0': 'CC0',
    'cc0-1.0': 'CC0',
    'cc-zero': 'CC0',
    'cc by': 'CC-BY',
    'cc-by': 'CC-BY',
    'cc-by-4.0': 'CC-BY',
    'cc-by-3.0': 'CC-BY',
    'cc by-sa': 'CC-BY-SA',
    'cc-by-sa': 'CC-BY-SA',
    'cc-by-sa-4.0': 'CC-BY-SA',
    'cc-by-sa-3.0': 'CC-BY-SA',
    'cc by-nc': 'CC-BY-NC',
    'cc-by-nc': 'CC-BY-NC',
    'cc-by-nc-4.0': 'CC-BY-NC',
    'mit': 'MIT',
    'mit license': 'MIT',
    'apache': 'Apache-2.0',
    'apache-2.0': 'Apache-2.0',
    'apache 2.0': 'Apache-2.0',
    'apache license 2.0': 'Apache-2.0',
    'bsd': 'BSD',
    'bsd-2-clause': 'BSD',
    'bsd-3-clause': 'BSD',
    'bsd-4-clause': 'BSD',
    'gpl': 'GPL',
    'gpl-2.0': 'GPL',
    'gpl-3.0': 'GPL',
    'gnu general public license': 'GPL',
    'lgpl': 'LGPL',
    'lgpl-2.1': 'LGPL',
    'lgpl-3.0': 'LGPL',
    'mpl': 'MPL',
    'mpl-2.0': 'MPL',
    'mozilla public license': 'MPL',
    'isc': 'ISC',
    'isc license': 'ISC',
    'proprietary': 'Proprietary',
    'all rights reserved': 'Proprietary',
    'restricted': 'Restricted',
    'unknown': 'Unknown',
}

_SPDX_TOKEN = re.compile(
    r'\b(?:SPDX-License-Identifier:\s*)?'
    r'(MIT|Apache-2\.0|BSD-2-Clause|BSD-3-Clause|BSD-4-Clause|'
    r'GPL-2\.0(?:-only|-or-later)?|GPL-3\.0(?:-only|-or-later)?|'
    r'LGPL-2\.1(?:-only|-or-later)?|LGPL-3\.0(?:-only|-or-later)?|'
    r'MPL-2\.0|ISC|CC0-1\.0|CC-BY(?:-SA|-NC)?(?:-\d\.\d)?)\b',
    re.I,
)

_CC_URL = re.compile(
    r'creativecommons\.org/(?:publicdomain/(?:zero|mark)/|licenses/(by|by-sa|by-nc)(?:-nc)?(?:-nd)?/[\d.]+/)',
    re.I,
)

def _canonicalize(raw: str) -> Optional[str]:
    key = re.sub(r'\s+', ' ', raw.strip().lower())
    key = key.replace('license identifier:', '').strip()
    if key in _SPDX_ALIASES:
        return _SPDX_ALIASES[key]
    for alias, label in _SPDX_ALIASES.items():
        if alias in key or key in alias:
            return label
    if 'apache' in key and '2' in key:
        return 'Apache-2.0'
    if key.startswith('gpl'):
        return 'GPL'
    if key.startswith('lgpl'):
        return 'LGPL'
    if key.startswith('bsd'):
        return 'BSD'
    if 'cc-by-sa' in key or 'by-sa' in key:
        return 'CC-BY-SA'
    if 'cc-by-nc' in key or 'by-nc' in key:
        return 'CC-BY-NC'
    if 'cc-by' in key or key == 'by':
        return 'CC-BY'
    if 'cc0' in key or 'cc-zero' in key:
        return 'CC0'
    if 'public domain' in key or key == 'pd':
        return 'Public Domain'
    if 'proprietary' in key or 'all rights reserved' in key:
        return 'Proprietary'
    if 'restricted' in key:
        return 'Restricted'
    return None

def normalize_license_string(raw: Optional[str]) -> tuple[str, float]:
    if not raw or not str(raw).strip():
        return 'Unknown', 0.0
    text = str(raw).strip()
    canon = _canonicalize(text)
    if canon and canon in LICENSE_CATEGORIES:
        if re.match(r'^[A-Za-z0-9._\-/ ]+$', text) and len(text) <= 64:
            return canon, 0.95
        return canon, 0.85
    return 'Unknown', 0.0

def detect_license_in_text(text: str) -> tuple[str, float]:
    if not text:
        return 'Unknown', 0.0
    sample = text[:12000]
    m = _SPDX_TOKEN.search(sample)
    if m:
        canon = _canonicalize(m.group(1))
        if canon:
            return canon, 0.90
    m = _CC_URL.search(sample)
    if m:
        kind = (m.group(1) or '').lower()
        if kind == 'by-sa':
            return 'CC-BY-SA', 0.88
        if kind == 'by-nc':
            return 'CC-BY-NC', 0.88
        if kind == 'by':
            return 'CC-BY', 0.88
        if 'zero' in sample[m.start():m.end() + 20].lower():
            return 'CC0', 0.88
    explicit = re.search(
        r'(?i)\b(?:licensed under|license:\s*|released under)\s+([^\n.;]{3,48})',
        sample,
    )
    if explicit:
        canon, conf = normalize_license_string(explicit.group(1))
        if canon != 'Unknown':
            return canon, min(conf, 0.82)
    return 'Unknown', 0.0
