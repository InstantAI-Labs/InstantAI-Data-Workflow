from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from core.runtime.stable_hash import stable_digest_int
from indw.schedule.mix.mixture_planner import document_sampling_weight
from indw.schedule.mix.plan import CorpusMixturePlan
from indw.schedule.mix.telemetry import stage_for_token_cursor
from indw.store.export.export_items import ExportRecord

@dataclass(frozen=True)
class MixtureDocumentMeta:
    line: int
    domain: str
    language: str
    score: float
    synthetic_score: float
    reasoning_density: float
    factual_density: float
    educational_value: float
    token_spam_score: float
    context_len: int
    token_count: int
    weight: float
    document_id: str = ''

def load_mixture_index(path: Path) -> list[MixtureDocumentMeta]:
    out: list[MixtureDocumentMeta] = []
    with Path(path).open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            raw = json.loads(line)
            out.append(
                MixtureDocumentMeta(
                    line=i,
                    domain=str(raw.get('domain', 'web')),
                    language=str(raw.get('language', 'other')),
                    score=float(raw.get('score', 0.0)),
                    synthetic_score=float(raw.get('synthetic_score', 0.0)),
                    reasoning_density=float(raw.get('reasoning_density', 0.0)),
                    factual_density=float(raw.get('factual_density', 0.0)),
                    educational_value=float(raw.get('educational_value', 0.0)),
                    token_spam_score=float(raw.get('token_spam_score', 0.0)),
                    context_len=int(raw.get('context_len', 0)),
                    token_count=int(raw.get('token_count', 0) or raw.get('clean_token_estimate', 0)),
                    weight=float(raw.get('weight', 1.0)),
                    document_id=str(raw.get('document_id', '') or ''),
                )
            )
    return out

def compute_doc_weight(meta: MixtureDocumentMeta, plan: CorpusMixturePlan, *, token_cursor: int = 0) -> float:
    pol = dict(plan.telemetry.get('policy') or {})
    min_q = pol.get('min_quality')
    if min_q is not None and float(meta.score) < float(min_q):
        return 0.0
    max_syn = pol.get('synthetic_cap')
    if max_syn is not None and float(meta.synthetic_score) > float(max_syn):
        return 0.0
    min_ctx_ratio = pol.get('min_long_context_ratio')
    if min_ctx_ratio is not None:
        min_ctx = int(pol.get('long_context_min_chars', 4096))
        if float(min_ctx_ratio) >= 1.0 and int(meta.context_len) < min_ctx:
            return 0.0
    lang_quota = dict(pol.get('language_quotas') or {})
    if lang_quota and meta.language not in lang_quota:
        return 0.0

    live = dict(plan.telemetry.get('live_data_intelligence') or {})
    if live.get('max_duplication_risk') is not None and float(meta.token_spam_score) > float(live['max_duplication_risk']):
        return 0.0
    if live.get('min_reasoning_density') is not None and float(meta.reasoning_density) < float(live['min_reasoning_density']):
        return 0.0
    if live.get('min_educational_value') is not None and float(meta.educational_value) < float(live['min_educational_value']):
        return 0.0
    stage = stage_for_token_cursor(plan, token_cursor)
    w = document_sampling_weight(
        domain=meta.domain,
        language=meta.language,
        score=meta.score,
        synthetic_score=meta.synthetic_score,
        reasoning_density=meta.reasoning_density,
        factual_density=meta.factual_density,
        educational_value=meta.educational_value,
        token_spam_score=meta.token_spam_score,
        context_len=meta.context_len,
        plan=plan,
        stage_name=stage,
    )
    if live.get('entropy_boost'):
        w *= max(1e-06, 1.0 + float(live['entropy_boost']) * float(meta.factual_density))
    return w

def deterministic_weighted_order(
    meta: list[MixtureDocumentMeta],
    plan: CorpusMixturePlan,
    *,
    seed: int,
) -> list[int]:
    ranked: list[tuple[float, int]] = []
    lang_seen: dict[str, int] = {}
    token_cursor = 0
    pol = dict(plan.telemetry.get('policy') or {})
    lang_quota = {str(k): int(v) for k, v in dict(pol.get('language_quotas') or {}).items()}
    for i, m in enumerate(meta):
        if lang_quota:
            cur = int(lang_seen.get(m.language, 0))
            cap = int(lang_quota.get(m.language, 0))
            if cap <= 0 or cur >= cap:
                continue
        stage = stage_for_token_cursor(plan, token_cursor)
        w = document_sampling_weight(
            domain=m.domain,
            language=m.language,
            score=m.score,
            synthetic_score=m.synthetic_score,
            reasoning_density=m.reasoning_density,
            factual_density=m.factual_density,
            educational_value=m.educational_value,
            token_spam_score=m.token_spam_score,
            context_len=m.context_len,
            plan=plan,
            stage_name=stage,
        )
        h = stable_digest_int({'seed': seed, 'line': i, 'digest': plan.plan_digest}, bits=64)
        ranked.append((float(h) / max(w, 1e-9), i))
        lang_seen[m.language] = int(lang_seen.get(m.language, 0)) + 1
        token_cursor += max(1, int(m.token_count) if int(m.token_count) > 0 else (m.context_len // 4))
    ranked.sort(reverse=True)
    return [i for _, i in ranked]

def replay_safe_weighted_iterator(
    jsonl_path: Path,
    index_path: Path,
    plan: CorpusMixturePlan,
    *,
    replay_jsonl: Optional[Path] = None,
    replay_ratio: float = 0.0,
    seed: Optional[int] = None,
) -> Iterator[ExportRecord]:
    seed = int(seed if seed is not None else plan.replay_seed)
    rng = random.Random(seed)
    meta = load_mixture_index(index_path)
    lines: list[str] = []
    with Path(jsonl_path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(line)
    if not meta:
        raise ValueError(f'mixture index empty: {index_path}')
    if len(meta) != len(lines):
        raise ValueError(
            f'mixture index/jsonl length mismatch: index={len(meta)} jsonl={len(lines)} '
            f'({index_path} vs {jsonl_path})'
        )
    order = deterministic_weighted_order(meta, plan, seed=seed)
    replay_iter: Optional[Iterator[str]] = None
    if replay_jsonl and Path(replay_jsonl).exists() and replay_ratio > 0:
        from indw.store.export.replay_export import _cycle_jsonl
        replay_iter = _cycle_jsonl(Path(replay_jsonl))
    for i in order:
        if replay_iter is not None and rng.random() < replay_ratio:
            try:
                replay_text = next(replay_iter)
                if replay_text:
                    yield ExportRecord(text=replay_text, split_key='__replay__', is_replay=True)
                continue
            except StopIteration:
                replay_iter = None
        row = json.loads(lines[i])
        text = row.get('text', '')
        if not text:
            continue
        m = meta[i]
        split_key = m.document_id or f'line:{i}'
        yield ExportRecord(text=text, split_key=split_key, is_replay=False)
