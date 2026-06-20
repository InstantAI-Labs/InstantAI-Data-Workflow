from __future__ import annotations

from typing import Any, Literal

from indw.config.defaults import LICENSE_PIPELINE_VERSION as PIPELINE_VERSION

LicenseCategory = Literal[
    'Public Domain',
    'CC0',
    'CC-BY',
    'CC-BY-SA',
    'CC-BY-NC',
    'MIT',
    'Apache-2.0',
    'BSD',
    'GPL',
    'LGPL',
    'MPL',
    'ISC',
    'Unknown',
    'Proprietary',
    'Restricted',
]

LICENSE_CATEGORIES: tuple[str, ...] = (
    'Public Domain',
    'CC0',
    'CC-BY',
    'CC-BY-SA',
    'CC-BY-NC',
    'MIT',
    'Apache-2.0',
    'BSD',
    'GPL',
    'LGPL',
    'MPL',
    'ISC',
    'Unknown',
    'Proprietary',
    'Restricted',
)

CopyrightStatus = Literal[
    'clear',
    'attribution_required',
    'share_alike',
    'unknown',
    'restricted',
    'prohibited',
]

DocumentType = Literal[
    'web',
    'wiki',
    'code_repository',
    'book',
    'news',
    'government',
    'academic',
    'forum',
    'unknown',
]

FilterAction = Literal['KEEP', 'FLAG', 'REMOVE']

PROVENANCE_FIELDS: tuple[str, ...] = (
    'text',
    'source',
    'url',
    'domain',
    'license',
    'license_confidence',
    'crawl_date',
    'language',
    'document_type',
    'copyright_status',
    'attribution_required',
)

PROVENANCE_JSON_SCHEMA: dict[str, Any] = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    '$id': 'https://instant.ai/schemas/pretraining-record-v1.json',
    'title': 'PretrainingCorpusRecord',
    'type': 'object',
    'required': list(PROVENANCE_FIELDS),
    'properties': {
        'text': {'type': 'string', 'minLength': 1},
        'source': {'type': 'string'},
        'url': {'type': 'string'},
        'domain': {'type': 'string'},
        'license': {'type': 'string', 'enum': list(LICENSE_CATEGORIES)},
        'license_confidence': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
        'crawl_date': {'type': 'string'},
        'language': {'type': 'string'},
        'document_type': {'type': 'string'},
        'copyright_status': {'type': 'string'},
        'attribution_required': {'type': 'boolean'},
        'meta': {'type': 'object'},
    },
    'additionalProperties': True,
}
