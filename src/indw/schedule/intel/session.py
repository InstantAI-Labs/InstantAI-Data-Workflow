from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from indw.config.defaults import (
    ACIM_ENABLED,
    ACIM_FAST_PATH,
    ACIM_OBSERVE_ONLY,
    ACIM_STORE_VERSION,
    LCI_ENABLED,
    LCI_NUM_SHARDS,
    LCI_OBSERVE_ONLY,
    LCI_STORE_VERSION,
)
from indw.schedule.intel.fingerprints import IntelligenceBundle, build_intelligence_bundle
from indw.schedule.intel.genome import extract_genes, extract_genes_from_line, resolve_source_domain
from indw.schedule.intel.coordination import IntelligenceCoordinator
from indw.schedule.intel.hardware import collect_hardware_snapshot
from indw.schedule.intel.lci_graph import LCIContext, LivingCorpusGraph
from indw.schedule.intel.lci_router import lci_route_dict, route_with_lci
from indw.schedule.intel.promotion import (
    build_cleaned_intel,
    line_quality_ok,
    structural_hash_for_intel,
)
from indw.schedule.intel.router import ProcessingDepth, RouteDecision, route_document
from indw.schedule.intel.store import IntelligenceStore
from indw.schedule.config.resolve import env_str
from indw.schedule.config.resolve import env_flag as _env_flag
from indw.schedule.intel.pci import FingerprintBundle, _load_snapshot, fingerprint_from_line, fingerprint_from_raw


_tls = threading.local()
_BOUND: ACIMSession | None = None


class ACIMSession:
    def __init__(self, merge_work: Path, *, readonly: bool = False) -> None:
        self.merge_work = Path(merge_work)
        self.readonly = readonly
        self.enabled = _env_flag('INSTANT_ACIM_ENABLED', ACIM_ENABLED)
        self.observe_only = _env_flag('INSTANT_ACIM_OBSERVE_ONLY', ACIM_OBSERVE_ONLY)
        self.fast_path = _env_flag('INSTANT_ACIM_FAST_PATH', ACIM_FAST_PATH) and not self.observe_only
        self.lci_enabled = _env_flag('INSTANT_LCI_ENABLED', LCI_ENABLED)
        self.lci_observe_only = _env_flag('INSTANT_LCI_OBSERVE_ONLY', LCI_OBSERVE_ONLY)
        self.version = env_str('INSTANT_ACIM_VERSION', ACIM_STORE_VERSION)
        self.lci_version = env_str('INSTANT_LCI_VERSION', LCI_STORE_VERSION)
        self._lock = threading.Lock()
        self._docs = 0
        self._reused = 0
        self._verify_capable = 0
        self._fallbacks = 0
        self._lci_reuse = 0
        self._gene_known = 0
        self._gene_novel = 0
        self._promotions = 0
        self._promotion_fails = 0
        self._incremental_hits = 0
        self._hw_snapshot: Any = None
        self._depth_counts: dict[str, int] = {}
        self._started = time.perf_counter()
        self._dir = self.merge_work / 'acim'
        db_path = self._dir / 'intelligence.sqlite'
        if self.readonly:
            self._store = (
                IntelligenceStore.open_readonly(db_path, version=self.lci_version)
                if db_path.is_file()
                else IntelligenceStore(db_path, version=self.lci_version)
            )
        else:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._store = IntelligenceStore(db_path, version=self.lci_version)
        self._graph = (
            LivingCorpusGraph(self._store, num_shards=LCI_NUM_SHARDS)
            if self.lci_enabled
            else None
        )
        self._coordinator = (
            IntelligenceCoordinator(self.merge_work, num_shards=LCI_NUM_SHARDS)
            if self.lci_enabled and not self.readonly
            else None
        )
        self._pci_snapshot = _load_snapshot(self.merge_work / 'pci' / 'pci_snapshot.json')
        self._events_enabled = False
        if self.enabled:
            from indw.schedule.monitor.obs import is_debug
            self._events_enabled = is_debug() or _env_flag('INSTANT_ACIM_EVENTS', False)
        self._events_fp = (
            (self._dir / 'acim_events.jsonl').open('a', encoding='utf-8')
            if self.enabled and self._events_enabled and not self.readonly
            else None
        )

    @classmethod
    def for_worker(cls, merge_work: Path) -> ACIMSession | None:
        if not _env_flag('INSTANT_ACIM_ENABLED', ACIM_ENABLED):
            return None
        sess = cls(merge_work, readonly=True)
        if not sess.enabled:
            sess.close()
            return None
        return sess

    def preview_for_cleaning(
        self,
        text: str,
        *,
        source: str = '',
        fp: FingerprintBundle | None = None,
        scan: Any | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        if not self.enabled:
            return None, None, None
        intel, lci = self.classify(text, source=source, observe_lci=False, fp=fp, scan=scan)
        route = self.route(intel, lci)
        self.apply_cache_boost(route.cache_boost)
        acim_route = (
            lci_route_dict(route, lci) if lci is not None else route.to_dict()
        )
        lci_payload = lci.to_dict() if lci is not None else None
        return intel.to_dict(), acim_route, lci_payload

    def classify(
        self,
        text: str,
        *,
        source: str = '',
        observe_lci: bool = False,
        fp: FingerprintBundle | None = None,
        scan: Any | None = None,
    ) -> tuple[IntelligenceBundle, LCIContext | None]:
        base = build_intelligence_bundle(text, fp=fp, scan=scan)
        rec = self._store.lookup_family(base.family_key)
        pci_label = self._pci_snapshot.get(base.family_key, '')
        if rec is None:
            fid = f'fam_{pci_label}' if pci_label else base.family_id
            intel = base if fid == base.family_id else base.with_family(family_id=fid)
        else:
            intel = base.with_family(
                family_id=rec.family_id,
                observation_count=rec.observation_count,
                verified_count=rec.verified_count,
            )
        lci_ctx = None
        if self._graph is not None:
            domain = resolve_source_domain(source)
            genome = extract_genes(text, domain_id=domain)
            if observe_lci:
                lci_ctx = self._graph.observe_genome(
                    genome,
                    family_id=intel.family_id,
                    family_key=intel.family_key,
                )
            else:
                lci_ctx = self._graph.preview_genome(
                    genome,
                    family_id=intel.family_id,
                    family_key=intel.family_key,
                )
        return intel, lci_ctx

    def route(
        self,
        intel: IntelligenceBundle,
        lci: LCIContext | None = None,
        *,
        hw: Any = None,
    ) -> RouteDecision:
        if not self.enabled:
            return RouteDecision(depth=ProcessingDepth.FULL, reason='acim_disabled')
        if self._graph is not None and lci is not None:
            decision = route_with_lci(
                intel,
                self._store,
                lci,
                fast_path_enabled=self.fast_path and not self.lci_observe_only,
                hw=hw or collect_hardware_snapshot(),
            )
        else:
            decision = route_document(
                intel,
                self._store,
                fast_path_enabled=self.fast_path,
            )
        if decision.depth == ProcessingDepth.FULL and decision.reason not in (
            'high_complexity_or_entropy', 'high_novelty', 'acim_disabled',
            'high_novel_gene_ratio',
        ):
            with self._lock:
                self._fallbacks += 1
        return decision

    def update_hardware(self, **kwargs: Any) -> None:
        self._hw_snapshot = collect_hardware_snapshot(
            queue_depth=int(kwargs.get('queue_depth', 0)),
            docs_per_sec=float(kwargs.get('docs_per_sec', 0.0)),
            cpu_pct=kwargs.get('cpu_pct'),
            rss_mb=kwargs.get('rss_mb'),
            cache_hit_rate=kwargs.get('cache_hit_rate'),
        )

    def _hardware(self) -> Any:
        return self._hw_snapshot or collect_hardware_snapshot()

    def observe_preprocessed(self, line: dict[str, Any]) -> RouteDecision | None:
        if not self.enabled or self.readonly:
            return None
        text = str(line.get('raw_text') or '')
        source = str(line.get('src_name') or '')
        intel_raw = line.get('acim_intel')
        intel = IntelligenceBundle.from_dict(intel_raw) if isinstance(intel_raw, dict) else None
        fp = fingerprint_from_line(line)
        if intel is None:
            if not text:
                return None
            intel, _ = self.classify(text, source=source, observe_lci=False, fp=fp)
        label = self._pci_snapshot.get(intel.family_key, intel.family_id)
        rec = self._store.observe(intel, label=label)
        intel = intel.with_store_record(rec)
        lci = None
        promo = None
        if self._graph is not None:
            raw_genome, cleaned_genome = extract_genes_from_line(line, source=source)
            genome = cleaned_genome or raw_genome
            if genome is not None:
                lci = self._graph.observe_genome(
                    genome,
                    family_id=intel.family_id,
                    family_key=intel.family_key,
                )
                lci = self._graph.attach_incremental(
                    lci, line, family_key=intel.family_key,
                )
                if lci.incremental is not None and lci.incremental.reuse_eligible:
                    with self._lock:
                        self._incremental_hits += 1
                if (
                    cleaned_genome is not None
                    and not self.lci_observe_only
                    and line_quality_ok(line)
                ):
                    cleaned_text = '\n\n'.join(
                        str(c.get('chunk_text') or '')
                        for c in line.get('chunks') or []
                        if isinstance(c, dict) and c.get('chunk_text')
                    )
                    cleaned_intel = build_cleaned_intel(cleaned_text, intel)
                    vhash = structural_hash_for_intel(cleaned_intel)
                    promo = self._graph.attempt_promotions(
                        cleaned=cleaned_genome,
                        raw=raw_genome,
                        family_id=intel.family_id,
                        structural_hash=vhash,
                        quality_ok=line_quality_ok(line),
                        intel_observation_count=rec.observation_count,
                        intel_family_confidence=intel.family_confidence,
                        coordinator=self._coordinator,
                    )
                    with self._lock:
                        if promo.genes_promoted > 0 or promo.family_promoted:
                            self._promotions += 1
                        else:
                            self._promotion_fails += 1
        decision = self.route(intel, lci, hw=self._hardware())
        reused = decision.depth in (ProcessingDepth.VERIFY, ProcessingDepth.FAST)
        with self._lock:
            self._docs += 1
            if reused:
                self._reused += 1
                if self._graph is not None:
                    self._lci_reuse += 1
            if lci is not None:
                self._gene_known += lci.known_gene_count
                self._gene_novel += len(lci.genome.novel_gene_keys)
            if decision.verified_nodes:
                self._verify_capable += 1
            self._depth_counts[decision.depth.value] = (
                self._depth_counts.get(decision.depth.value, 0) + 1
            )
        self._store.record_observation_event(
            family_id=intel.family_id,
            seq=int(line.get('seq') or 0),
            source=str(line.get('src_name') or ''),
            depth=decision.depth.value,
            reused=reused,
        )
        if self._events_fp is not None:
            self._events_fp.write(json.dumps({
                'version': self.version,
                'observe_only': self.observe_only,
                'seq': line.get('seq'),
                'source': line.get('src_name', ''),
                'intel': intel.to_dict(),
                'route': decision.to_dict(),
            }, ensure_ascii=False) + '\n')
            if self._docs % 200 == 0:
                self._events_fp.flush()
        line['acim_intel'] = intel.to_dict()
        line['acim_route'] = (
            lci_route_dict(decision, lci) if lci is not None else decision.to_dict()
        )
        if lci is not None:
            line['lci'] = lci.to_dict()
        if promo is not None:
            line['lci_promotion'] = promo.to_dict()
        return decision

    def apply_cache_boost(self, boost: int) -> None:
        if boost <= 1:
            return
        try:
            from indw.clean.artifact.evidence_cache import set_session_cache_boost
            set_session_cache_boost(boost)
        except Exception:
            pass

    def stats(self) -> dict[str, Any]:
        with self._lock:
            docs = self._docs
            elapsed = max(time.perf_counter() - self._started, 1e-9)
            store_stats = self._store.stats()
            lci_stats = self._graph.stats() if self._graph is not None else {}
            gene_total = self._gene_known + self._gene_novel
            return {
                'enabled': self.enabled,
                'observe_only': self.observe_only,
                'fast_path': self.fast_path,
                'lci_enabled': self.lci_enabled,
                'lci_observe_only': self.lci_observe_only,
                'version': self.version,
                'lci_version': self.lci_version,
                'docs_classified': docs,
                'reuse_rate': round(self._reused / docs, 4) if docs else 0.0,
                'lci_reuse_rate': round(self._lci_reuse / docs, 4) if docs else 0.0,
                'gene_reuse_rate': round(self._gene_known / gene_total, 4) if gene_total else 0.0,
                'verify_capable_rate': round(self._verify_capable / docs, 4) if docs else 0.0,
                'fallback_rate': round(self._fallbacks / docs, 4) if docs else 0.0,
                'promotion_success_rate': round(
                    self._promotions / max(self._promotions + self._promotion_fails, 1), 4,
                ),
                'incremental_reuse_rate': round(self._incremental_hits / docs, 4) if docs else 0.0,
                'depth_counts': dict(self._depth_counts),
                'docs_per_sec': round(docs / elapsed, 3),
                'hardware': self._hardware().to_dict(),
                'store': store_stats,
                'lci': lci_stats,
                'coordination': (
                    self._coordinator.stats() if self._coordinator is not None else {}
                ),
            }

    def close(self) -> None:
        if self._events_fp is not None:
            self._events_fp.flush()
            self._events_fp.close()
        if self.readonly:
            self._store.close()
            return
        if not self.enabled:
            self._store.close()
            return
        stats = self.stats()
        (self._dir / 'acim_run_stats.json').write_text(
            json.dumps(stats, indent=2),
            encoding='utf-8',
        )
        if self._graph is not None:
            retired = self._graph.retire_stale_genes()
            stats = self.stats()
            lci_payload = dict(stats.get('lci', {}))
            lci_payload['genes_retired'] = retired
            (self._dir / 'lci_run_stats.json').write_text(
                json.dumps(lci_payload, indent=2),
                encoding='utf-8',
            )
        if self._coordinator is not None:
            self._coordinator.write_snapshot(self._dir / 'intelligence.sqlite')
        snap_path = self._dir / 'pci_snapshot_promoted.json'
        snap_path.write_text(
            json.dumps(self._store.export_pci_snapshot(), indent=2),
            encoding='utf-8',
        )
        self._store.close()

    def pci_compat_stats(self) -> dict[str, Any]:
        s = self.stats()
        return {
            'enabled': s['enabled'],
            'observe_only': s['observe_only'],
            'snapshot_id': self.version,
            'docs_observed': s['docs_classified'],
            'template_matches': int(s['reuse_rate'] * s['docs_classified']),
            'template_match_rate': s['reuse_rate'],
            'events_written': s['docs_classified'] if self._events_enabled else 0,
            'families': {
                row['family_id']: row['observations']
                for row in s.get('store', {}).get('top_families', [])
            },
            'acim': s,
        }


def bind_acim_session(session: ACIMSession | None) -> None:
    global _BOUND
    _BOUND = session
    _tls.session = session


def get_acim_session() -> ACIMSession | None:
    sess = getattr(_tls, 'session', None)
    if sess is not None:
        return sess
    return _BOUND
