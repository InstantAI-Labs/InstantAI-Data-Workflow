from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from indw.filter.gate.quality import QualityGate, QualityRunStats

def length_histogram(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {}
    buckets = {'short': 0, 'medium': 0, 'long': 0, 'very_long': 0}
    for ln in lengths:
        if ln < 800:
            buckets['short'] += 1
        elif ln < 2400:
            buckets['medium'] += 1
        elif ln < 6000:
            buckets['long'] += 1
        else:
            buckets['very_long'] += 1
    total = len(lengths)
    return {k: round(v / total, 4) for k, v in buckets.items()}

@dataclass
class CorpusQualityReport:
    version: str = 'instant-quality-v1'
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    quality: dict[str, Any] = field(default_factory=dict)
    dedup: dict[str, Any] = field(default_factory=dict)
    balance: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    throughput: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return         {
            'version': self.version,
            'created_at': self.created_at,
            'quality': self.quality,
            'dedup': self.dedup,
            'balance': self.balance,
            'tokenizer': self.tokenizer,
            'throughput': self.throughput,
            'recommendations': self.recommendations
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')
        return path

def build_quality_report(gate: QualityGate, *, dedup_stats: Optional[dict[str, Any]]=None, merge_stats: Optional[dict[str, Any]]=None, tokenizer_stats: Optional[dict[str, Any]]=None, elapsed_sec: float=0.0) -> CorpusQualityReport:
    qs = gate.stats.to_dict()
    if merge_stats:
        if merge_stats.get('kept') is not None:
            qs['kept'] = int(merge_stats['kept'])
        if merge_stats.get('rejected') is not None:
            qs['rejected'] = int(merge_stats['rejected'])
        if merge_stats.get('scanned') is not None:
            qs['scanned'] = int(merge_stats['scanned'])
    domain_dist = gate.domain_balancer.distribution()
    lang_dist = gate.lang_balancer.distribution()
    recs: list[str] = []
    cal = gate.calibrator.distribution_stats() if hasattr(gate, 'calibrator') else {}
    kept = max(int(qs.get('kept', 0)), 1)
    evaluated = max(int(qs.get('evaluated', kept + int(qs.get('rejected', 0)))), 1)
    rej = qs.get('reject_reasons') or {}
    evidence_rej = qs.get('evidence_discard_reasons') or {}

    if rej:
        top_reason, top_n = max(rej.items(), key=lambda item: item[1])
        if top_n / evaluated > 0.12:
            recs.append(
                f'Dominant gate rejection: {top_reason} ({top_n}/{evaluated}) — review source mix for this failure mode'
            )
    if evidence_rej:
        ev_reason, ev_n = max(evidence_rej.items(), key=lambda item: item[1])
        if ev_n / max(int(qs.get('evidence_evaluated', evaluated)), 1) > 0.10:
            recs.append(
                f'Dominant semantic discard: {ev_reason} ({ev_n}) — corpus semantic profile skewed toward low utility'
            )

    if cal.get('ready'):
        p50 = float(cal.get('composite_p50', 0.0))
        p10 = float(cal.get('composite_p10', 0.0))
        score_mean = float(qs.get('score_mean', 0.0))
        if score_mean < p10:
            recs.append(
                f'Kept-doc score mean ({score_mean:.3f}) below corpus p10 ({p10:.3f}) — quality regression vs reservoir'
            )
        elif score_mean > p50 * 1.15 and qs.get('rejected', 0) > kept * 0.4:
            recs.append('High reject rate despite above-median scores — review adaptive calibration warmup')

    utility_mean = float(qs.get('utility_mean', 0.0))
    preserve_rate = float(qs.get('preserve_rate', 0.0))
    if qs.get('evidence_evaluated', 0) > 50 and utility_mean < 0.22:
        recs.append(f'Mean semantic utility low ({utility_mean:.3f}) — prioritize higher-information sources')
    if qs.get('evidence_evaluated', 0) > 50 and preserve_rate < 0.35:
        recs.append(f'Semantic preserve rate low ({preserve_rate:.1%}) — corpus dominated by low-value content classes')

    tok = qs.get('tokenizer_telemetry') or {}
    if float(tok.get('token_inflation_mean', 0.0)) > 0.45:
        recs.append('Tokenizer inflation elevated — review multilingual/script mix or tokenizer config')
    if float(tok.get('replay_stability_mean', 1.0)) < 0.82:
        recs.append('Tokenizer replay instability — encoding drift may reduce training efficiency')

    if qs.get('synthetic_score_mean', 0) > max(0.40, float(cal.get('composite_p50', 0.35))):
        recs.append('Synthetic signal above corpus baseline — review synthetic defense policy')
    if domain_dist.get('code', 0) > 0.25:
        recs.append('Code domain share above 25% — verify domain balancer caps')
    if lang_dist.get('other', 0) > 0.15:
        recs.append("Language 'other' bucket elevated — expand language policy coverage")
    kept = int(merge_stats.get('kept', qs.get('kept', 0)) if merge_stats else qs.get('kept', 0))
    rej = int(merge_stats.get('rejected', qs.get('rejected', 0)) if merge_stats else qs.get('rejected', 0))
    scanned = int(merge_stats.get('scanned', qs.get('scanned', kept + rej)) if merge_stats else kept + rej)
    token_savings = 0.0
    if dedup_stats:
        dupes = (
            int(dedup_stats.get('exact_duplicates', 0))
            + int(dedup_stats.get('fuzzy_duplicates', 0) or dedup_stats.get('duplicates', 0))
            + int(dedup_stats.get('semantic_duplicates', 0))
        )
        if kept + dupes > 0:
            token_savings = dupes / (kept + dupes)
    return CorpusQualityReport(
        quality=qs,
        dedup=dedup_stats or {},
        balance=        {
            'domain_distribution': domain_dist,
            'language_distribution': lang_dist,
            'merge_stats': merge_stats or {},
            'estimated_duplicate_token_savings': round(token_savings, 4)
        },
        tokenizer=tokenizer_stats or {},
        throughput={
            'elapsed_sec': elapsed_sec,
            'docs_per_sec': scanned / max(elapsed_sec, 0.001),
            'scan_docs_per_sec': scanned / max(elapsed_sec, 0.001),
            'keep_docs_per_sec': kept / max(elapsed_sec, 0.001),
        },
        recommendations=recs
    )

def append_quality_history(work_dir: Path, report: CorpusQualityReport) -> Path:
    hist_dir = Path(work_dir) / 'quality' / 'history'
    hist_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(hist_dir.glob('run_*.json'))) + 1
    out = hist_dir / f'run_{n:04d}.json'
    report.save(out)
    (Path(work_dir) / 'quality' / 'latest.json').write_text(json.dumps({'report': str(out), 'run': n}, indent=2), encoding='utf-8')
    return out
