from __future__ import annotations

from typing import Any


def provenance_for_merge_row(
    row: dict[str, Any],
    src_name: str,
    *,
    license_policy: Any,
    source_registry: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not license_policy.include_provenance_in_jsonl:
        return None
    return provenance_for_row(row, source_name=src_name, source_registry=source_registry)


def provenance_for_row(
    row: dict[str, Any],
    *,
    source_name: str,
    source_registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from indw.filter.license.record import extract_row_provenance

    prov = extract_row_provenance(row)
    prov['source'] = prov['source'] or source_name
    declared = source_registry.get(source_name) or {}
    for key in ('declared_license', 'hf_id', 'url', 'domain'):
        if not prov.get(key if key != 'declared_license' else 'license') and declared.get(key):
            if key == 'declared_license':
                prov['license'] = declared[key]
            else:
                prov[key] = declared[key]
    meta = row.get('meta') if isinstance(row.get('meta'), dict) else {}
    prov['meta'] = {**declared, **meta}
    return prov
