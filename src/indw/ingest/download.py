from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator

from indw.ingest.format import FORMATTERS, plain_text
from indw.ingest.hf_datasets import hf_datasets_available, load_dataset as hf_load_dataset
from indw.ingest.hf_env import configure_hf_fast
from indw.ingest.log import human_bytes
from indw.ingest.sink import DEFAULT_WRITE_BUFFER, stream_hf_to_jsonl
from indw.store.io.jsonl import count_jsonl_lines, write_source_line_meta

logger = logging.getLogger(__name__)


class DatasetDownloader:

    def __init__(self, output_dir: Path, *, write_buffer_bytes: int = DEFAULT_WRITE_BUFFER):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_buffer_bytes = write_buffer_bytes

    def fetch_all(
        self,
        sources: dict[str, Any],
        *,
        skip_if_exists: bool = False,
        existing_aliases: dict[str, str] | None = None,
    ) -> Path:
        configure_hf_fast()
        spec = sources or {}
        budget = int(spec.get('budget_bytes', 1000000000))
        sources_list = spec.get('sources', [])
        if not isinstance(sources_list, list):
            raise TypeError('sources must be list[dict]')
        total_pct = sum((float(s.get('mix_pct', 0)) for s in sources_list))
        if total_pct and abs(total_pct - 100.0) > 0.01:
            logger.warning('Source mix_pct sums to %.1f (expected 100)', total_pct)
        logger.info(
            'STREAMING DOWNLOAD | budget=%s | sources=%d | config=%s',
            human_bytes(budget),
            len(sources_list),
            spec.get('meta', {}).get('id', 'inline'),
        )
        total_written = 0
        results: list[tuple[str, int, int, float]] = []
        for idx, source in enumerate(sources_list, start=1):
            if 'mix_pct' in source:
                source['max_bytes'] = int(budget * float(source['mix_pct']) / 100.0)
            name = source['name']
            cap = int(source.get('max_bytes', 0))
            pct = source.get('mix_pct', '?')
            out_dir = self.output_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / 'data.jsonl'
            logger.info(
                '[%d/%d] %s | %s%% | cap=%s',
                idx,
                len(sources_list),
                name,
                pct,
                human_bytes(cap)
            )
            if out_file.exists() and out_file.stat().st_size == 0:
                out_file.unlink()
            alias_name = (existing_aliases or {}).get(name)
            alias_file = self.output_dir / alias_name / 'data.jsonl' if alias_name else None
            existing_path = out_file
            if alias_file is not None and alias_file.exists() and alias_file.stat().st_size > 0:
                existing_path = alias_file
            if existing_path.exists() and existing_path.stat().st_size > 0:
                existing = existing_path.stat().st_size
                if skip_if_exists or (cap and existing >= int(0.6 * cap)):
                    label = name if existing_path == out_file else f'{name} (via {alias_name})'
                    logger.info(
                        '[%d/%d] SKIP %s — %s (cap %s, %.0f%%)',
                        idx,
                        len(sources_list),
                        label,
                        human_bytes(existing),
                        human_bytes(cap),
                        100.0 * existing / cap if cap else 0.0,
                    )
                    total_written += existing
                    pct_done = 100.0 * existing / cap if cap else 0.0
                    results.append((name, existing, cap, pct_done))
                    continue
            if source.get('local'):
                written = self._copy_local(Path(source['local']), out_file, cap)
            elif source.get('optional') and (not self._hf_available()):
                logger.warning('Skipping optional source %s', name)
                written = 0
            else:
                try:
                    written = self._download_hf(source, out_file, idx, len(sources))
                except Exception as e:
                    if source.get('optional'):
                        logger.warning('Optional source %s failed: %s', name, e)
                        written = 0
                    else:
                        raise
            total_written += written
            pct_done = 100.0 * written / cap if cap else 0.0
            results.append((name, written, cap, pct_done))
            logger.info(
                '[%d/%d] DONE %s %s (%.0f%% of cap)',
                idx,
                len(sources_list),
                name,
                human_bytes(written),
                pct_done,
            )
        logger.info('DOWNLOAD SUMMARY | total=%s -> %s', human_bytes(total_written), self.output_dir)
        for name, written, cap, pct_done in results:
            status = 'OK' if cap and written >= int(0.9 * cap) else 'PARTIAL' if written else 'EMPTY'
            logger.info(
                '  %-22s %8s / %-8s  %5.0f%%  [%s]',
                name,
                human_bytes(written),
                human_bytes(cap) if cap else '-',
                pct_done,
                status,
            )
        return self.output_dir

    @staticmethod
    def _hf_available() -> bool:
        return hf_datasets_available()

    def _copy_local(self, src: Path, dst: Path, max_bytes: int) -> int:
        if not src.exists():
            logger.warning('Local source missing: %s', src)
            return 0
        written = 0
        with src.open(encoding='utf-8', buffering=DEFAULT_WRITE_BUFFER) as fin, dst.open('w', encoding='utf-8', buffering=DEFAULT_WRITE_BUFFER) as fout:
            for line in fin:
                if max_bytes and written + len(line.encode('utf-8')) > max_bytes:
                    break
                fout.write(line)
                written += len(line.encode('utf-8'))
        write_source_line_meta(dst, line_count=count_jsonl_lines(dst), bytes_written=written)
        return written

    def _download_hf(self, source: dict[str, Any], out_file: Path, idx: int, total: int) -> int:
        hf_id = source['hf_id']
        kwargs: dict[str, Any] = {}
        if source.get('hf_name'):
            kwargs['name'] = source['hf_name']
        elif source.get('hf_config'):
            kwargs['name'] = source['hf_config']
        split = source.get('split', 'train')
        streaming = bool(source.get('streaming', True))
        max_bytes = int(source.get('max_bytes', 100000000))
        name = source['name']
        logger.info(
            '[%d/%d] HF %s split=%s streaming=%s cap=%s',
            idx,
            total,
            hf_id,
            split,
            streaming,
            human_bytes(max_bytes)
        )
        ds = hf_load_dataset(hf_id, split=split, streaming=streaming, **kwargs)
        text_fn = self._resolve_formatter(source)
        languages = set(source.get('languages') or [])
        stats = stream_hf_to_jsonl(
            ds,
            out_file,
            text_fn=text_fn,
            max_bytes=max_bytes,
            languages=languages,
            desc=name,
            write_buffer_bytes=self.write_buffer_bytes
        )
        return int(stats['bytes'])

    def _resolve_formatter(self, source: dict[str, Any]) -> Callable[[dict[str, Any]], str]:
        fmt = source.get('format')
        if fmt == 'conversation':
            return FORMATTERS['conversation']
        if fmt == 'ultrachat':
            return FORMATTERS['ultrachat']
        if fmt in ('frontier', 'ultrachat_frontier'):
            return FORMATTERS['ultrachat_frontier']
        if fmt in ('code_frontier',):
            return FORMATTERS['code_frontier']
        if fmt == 'stack_exchange':
            return FORMATTERS['stack_exchange']
        if fmt == 'instruction_qa':
            return FORMATTERS['instruction_qa']
        if fmt == 'oasst':
            return FORMATTERS['oasst']
        if source.get('text_field') == 'code':
            return FORMATTERS['code']
        field = source.get('text_field', 'text')
        return lambda row: plain_text(row, field)
