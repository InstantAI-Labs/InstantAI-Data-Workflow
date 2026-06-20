from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import numpy as np
from tokenizers import Tokenizer
from indw.config.defaults import DEFAULT_WRITE_BUFFER_BYTES
from indw.store.export.shard_meta import write_shard_meta
from indw.store.export.splits import assign_split_for_key, validate_split_ratios
from indw.store.export.export_items import ExportRecord
logger = logging.getLogger(__name__)
READ_BUFFER_BYTES = DEFAULT_WRITE_BUFFER_BYTES

def export_token_bins_fast(jsonl_path: str | Path, tokenizer_path: str | Path, output_dir: str | Path, shard_tokens: int=50000000, val_ratio: float=0.01, test_ratio: float=0.0, eos_token: str='<|endoftext|>', *, flush_tokens: int=2000000, replay_jsonl: str | Path | None=None, replay_ratio: float=0.0, replay_seed: int=42, shard_index_offset: int=0, val_shard_index_offset: int=0, test_shard_index_offset: int=0) -> dict[str, list[Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(jsonl_path)
    tok = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tok.token_to_id(eos_token)
    if eos_id is None:
        raise ValueError(f'eos token missing from vocab: {eos_token!r} tokenizer={tokenizer_path}')
    enc_eos = tok.encode(eos_token, add_special_tokens=False)
    if not hasattr(enc_eos, 'ids') or len(enc_eos.ids) != 1 or enc_eos.ids[0] != eos_id:
        raise ValueError(f'eos token not atomic: {eos_token!r} ids={getattr(enc_eos, "ids", None)}')
    vocab_size = int(tok.get_vocab_size()) if hasattr(tok, 'get_vocab_size') else 0
    if vocab_size <= 0:
        raise ValueError(f'cannot determine vocab size for tokenizer={tokenizer_path}')
    if eos_id < 0 or eos_id >= vocab_size:
        raise ValueError(f'eos id out of range: eos_id={eos_id} vocab={vocab_size}')
    validate_split_ratios(val_ratio, test_ratio)
    buffers: dict[str, list[int]] = {'train': [], 'val': [], 'test': []}
    counters = {
        'train': shard_index_offset,
        'val': val_shard_index_offset,
        'test': test_shard_index_offset,
    }
    written: dict[str, list[Path]] = {'train': [], 'val': [], 'test': []}
    tokens_exported = 0
    val_cap = max(shard_tokens // 10, 500000)
    test_cap = max(shard_tokens // 20, 250000)
    lines = 0
    mixture_plan_path = jsonl_path.with_suffix('').with_name(jsonl_path.stem).parent / 'quality' / 'corpus_mixture_plan.json'
    mixture_index_path = jsonl_path.with_suffix('.mixture_index.jsonl')
    if not mixture_plan_path.exists():
        alt = Path(jsonl_path).resolve().parent.parent / 'quality' / 'corpus_mixture_plan.json'
        if alt.exists():
            mixture_plan_path = alt
    record_iter: Any
    if mixture_plan_path.exists() and mixture_index_path.exists():
        from indw.schedule.mix.plan import CorpusMixturePlan
        from indw.schedule.mix.sampler import replay_safe_weighted_iterator
        plan = CorpusMixturePlan.load(mixture_plan_path)
        record_iter = replay_safe_weighted_iterator(
            jsonl_path,
            mixture_index_path,
            plan,
            replay_jsonl=Path(replay_jsonl) if replay_jsonl else None,
            replay_ratio=replay_ratio,
            seed=replay_seed,
        )
        logger.info('Mixture-aware export: plan=%s digest=%s', mixture_plan_path, plan.plan_digest)
    else:
        from indw.store.export.replay_export import mixed_text_iterator
        record_iter = mixed_text_iterator(
            jsonl_path,
            Path(replay_jsonl) if replay_jsonl else None,
            replay_ratio=replay_ratio,
            seed=replay_seed,
        )
    if replay_jsonl and replay_ratio > 0:
        logger.info('Replay mixing: ratio=%.2f pool=%s (train-only)', replay_ratio, replay_jsonl)

    def resolve_split(rec: ExportRecord) -> str:
        if rec.is_replay:
            return 'train'
        return assign_split_for_key(
            rec.split_key,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=replay_seed,
        )

    def flush_split(split: str, force: bool=False) -> None:
        buf = buffers[split]
        if split == 'test':
            cap = test_cap
        elif split == 'val':
            cap = val_cap
        else:
            cap = shard_tokens
        if not buf:
            return
        if not force and len(buf) < min(flush_tokens, cap):
            return
        if len(buf) >= cap:
            to_write = buf[:cap]
            buffers[split] = buf[cap:]
        else:
            to_write = buf
            buffers[split] = []
        path = output_dir / f'{split}_{counters[split]:05d}.bin'
        arr = np.asarray(to_write, dtype=np.uint32)
        if int(arr.max(initial=0)) >= vocab_size:
            raise ValueError(f'shard contains token id >= vocab: max={int(arr.max())} vocab={vocab_size} path={path}')
        part = path.with_suffix(path.suffix + '.part')
        meta_path = path.with_suffix(path.suffix + '.meta.json')
        try:
            arr.tofile(part)
            part.replace(path)
            write_shard_meta(
                bin_path=path,
                tokenizer_path=tokenizer_path,
                dtype=np.uint32,
                vocab_size=vocab_size,
                eos_id=int(eos_id),
                tokens=int(len(arr)),
            )
        except Exception as exc:
            from indw.store.io.atomic import is_no_space
            from indw.tools.metrics.recovery import record_recovery_event
            part.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            if is_no_space(exc):
                record_recovery_event(output_dir, 'disk_full', path=str(path), phase='export')
            else:
                record_recovery_event(output_dir, 'export_partial_aborted', path=str(path))
            raise
        written[split].append(path)
        counters[split] += 1
        nonlocal tokens_exported
        tokens_exported += len(to_write)
        logger.info('  flushed %s → %s (%s tokens)', split, path.name, len(to_write))
    for rec in record_iter:
        ids = tok.encode(rec.text, add_special_tokens=False).ids
        if not ids:
            continue
        ids.append(eos_id)
        split = resolve_split(rec)
        buffers[split].extend(ids)
        lines += 1
        if lines % 50000 == 0:
            flush_split('train')
            flush_split('val')
            flush_split('test')
        if len(buffers['val']) >= val_cap:
            flush_split('val', force=True)
        if test_ratio > 0 and len(buffers['test']) >= test_cap:
            flush_split('test', force=True)
        if len(buffers['train']) >= shard_tokens:
            flush_split('train', force=True)
    flush_split('train', force=True)
    flush_split('val', force=True)
    flush_split('test', force=True)
    shard_count = len(written['train']) + len(written['val']) + len(written['test'])
    logger.info(
        'Fast export: %d lines → train=%d val=%d test=%d shards (offset=%d, tokens=%d)',
        lines,
        len(written['train']),
        len(written['val']),
        len(written['test']),
        shard_index_offset,
        tokens_exported,
    )
    written['_export_stats'] = {
        'lines': lines,
        'tokens_exported': tokens_exported,
        'shards_written': shard_count,
        'partial': False,
        'checksum_failures': 0,
    }
    return written
