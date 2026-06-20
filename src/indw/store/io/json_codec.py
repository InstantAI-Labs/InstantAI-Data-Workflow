from __future__ import annotations

import orjson
from typing import Any


def loads(raw: str | bytes) -> Any:
    if isinstance(raw, str):
        raw = raw.encode('utf-8')
    return orjson.loads(raw)


def dumps(obj: Any, *, ensure_ascii: bool = False) -> str:
    opts = 0
    if ensure_ascii:
        opts |= orjson.OPT_NON_STR_KEYS
    return orjson.dumps(obj, option=opts).decode('utf-8')


def dumps_pretty(obj: Any) -> str:
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode('utf-8')


def dumps_canonical(obj: Any) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)


def dumps_line(obj: Any, *, ensure_ascii: bool = False) -> str:
    return dumps(obj, ensure_ascii=ensure_ascii) + '\n'


def backend_name() -> str:
    return 'orjson'
