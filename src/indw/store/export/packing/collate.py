from __future__ import annotations

from typing import Any

import torch


def collate_packed_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    input_ids = torch.stack([b['input_ids'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    out: dict[str, Any] = {'input_ids': input_ids, 'labels': labels}

    if batch and 'segment_lengths' in batch[0]:
        out['segment_lengths'] = [
            b.get('segment_lengths', [input_ids.shape[1]])
            for b in batch
        ]
    elif batch and 'segment_ends' in batch[0]:
        out['segment_ends'] = [b.get('segment_ends', []) for b in batch]

    if batch and 'packing_efficiency' in batch[0]:
        out['packing_efficiency'] = (
            sum(b.get('packing_efficiency', 1.0) for b in batch) / len(batch)
        )
        out['documents_packed'] = sum(b.get('documents_packed', 1) for b in batch)

    return out


def collate_varlen_batch(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int = 0,
) -> dict[str, Any]:
    max_len = max(b['input_ids'].shape[0] for b in batch)
    ids, labs, lengths = [], [], []

    for b in batch:
        x, y = b['input_ids'], b['labels']
        pad = max_len - x.shape[0]
        if pad > 0:
            x = torch.cat([x, torch.full((pad,), pad_token_id, dtype=x.dtype)])
            y = torch.cat([y, torch.full((pad,), -100, dtype=y.dtype)])
        ids.append(x)
        labs.append(y)
        lengths.append(b['input_ids'].shape[0])

    return {
        'input_ids': torch.stack(ids),
        'labels': torch.stack(labs),
        'segment_lengths': [[l] for l in lengths],
        'actual_lengths': lengths,
    }
