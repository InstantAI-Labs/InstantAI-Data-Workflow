from __future__ import annotations
import io
import logging
from pathlib import Path
from typing import Any, Callable, Iterator
from indw.config.defaults import (
    DEFAULT_WRITE_BUFFER_BYTES,
    HF_STREAM_MIN_ALPHA_RATIO,
    MAX_CHARS_GATE,
    MIN_CHARS_GATE,
)
from indw.ingest.log import human_bytes
from indw.store.io.json_codec import dumps_line
from indw.store.io.jsonl import write_source_line_meta
logger = logging.getLogger(__name__)
DEFAULT_WRITE_BUFFER = DEFAULT_WRITE_BUFFER_BYTES

class BufferedJsonlWriter:

    def __init__(
        self,
        path: Path,
        buffer_bytes: int = DEFAULT_WRITE_BUFFER,
        mode: str = 'w',
        on_flush: Callable[[], None] | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer = io.StringIO()
        self._buffer_bytes = 0
        self._limit = buffer_bytes
        self._file = self.path.open(mode, encoding='utf-8', buffering=8192)
        self.lines = 0
        self.bytes_written = 0
        self._on_flush = on_flush

    def write_row(self, text: str, *, record: dict[str, Any] | None = None) -> int:
        if record is not None:
            payload = dict(record)
            payload['text'] = text
            line = dumps_line(payload)
        else:
            line = dumps_line({'text': text})
        n = len(line.encode('utf-8'))
        self._buffer.write(line)
        self._buffer_bytes += n
        self.lines += 1
        self.bytes_written += n
        if self._buffer_bytes >= self._limit:
            self.flush()
        return n

    def flush(self) -> None:
        if self._buffer_bytes == 0:
            self._file.flush()
            if self._on_flush is not None:
                self._on_flush()
            return
        self._file.write(self._buffer.getvalue())
        self._buffer = io.StringIO()
        self._buffer_bytes = 0
        self._file.flush()
        if self._on_flush is not None:
            self._on_flush()

    def close(self) -> int:
        self.flush()
        self._file.close()
        return self.bytes_written

    def __enter__(self) -> BufferedJsonlWriter:
        return self

    def __exit__(self, *_) -> None:
        self.close()

def stream_hf_to_jsonl(dataset: Iterator[dict[str, Any]], out_file: Path, *, text_fn: Callable[[dict[str, Any]], str], max_bytes: int, min_chars: int=MIN_CHARS_GATE, max_chars: int=MAX_CHARS_GATE, min_alpha_ratio: float=HF_STREAM_MIN_ALPHA_RATIO, languages: set[str] | None=None, desc: str='stream', write_buffer_bytes: int=DEFAULT_WRITE_BUFFER, show_progress: bool=True) -> dict[str, int]:
    from indw.clean.document.normalize import minimal_normalize
    written = 0
    docs = 0
    skipped = 0
    langs = languages or set()
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_bytes, unit='B', unit_scale=True, desc=desc[:24], leave=True)
        except ImportError:
            pbar = None
    with BufferedJsonlWriter(out_file, buffer_bytes=write_buffer_bytes) as sink:
        for row in dataset:
            if written >= max_bytes:
                break
            if langs:
                lang = row.get('language') or row.get('lang', '')
                if lang and lang not in langs:
                    skipped += 1
                    continue
            text = minimal_normalize(text_fn(row))
            if not text:
                skipped += 1
                continue
            n = len(text)
            if n < min_chars or n > max_chars:
                skipped += 1
                continue
            alpha = sum((c.isalpha() for c in text))
            if alpha / max(n, 1) < min_alpha_ratio:
                skipped += 1
                continue
            line_bytes = sink.write_row(text)
            if written + line_bytes > max_bytes:
                sink.flush()
                break
            written += line_bytes
            docs += 1
            if pbar is not None:
                pbar.update(line_bytes)
                if docs % 2000 == 0:
                    pbar.set_postfix(docs=docs, skip=skipped, written=human_bytes(written))
    if pbar is not None:
        pbar.close()
    stats = {'bytes': written, 'docs': docs, 'skipped': skipped}
    write_source_line_meta(out_file, line_count=docs, bytes_written=written)
    logger.info(
        'Streamed %s: %d docs, %d skipped, %s',
        desc,
        docs,
        skipped,
        human_bytes(written)
    )
    return stats
