from __future__ import annotations

import logging
import os
import socketserver
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import Any, Optional

logger = logging.getLogger(__name__)
_SERVER_STARTED = False
_BACKGROUND_STARTED = False
_PROMETHEUS_AVAILABLE = False
_METRICS_CACHE = b''
_METRICS_CACHE_LOCK = threading.Lock()
_PIPELINE_REGISTRY: Any = None
_PUSH_BACKOFF_UNTIL = 0.0
_PUSH_WARNED = False
_LAST_PUSH_SUCCESS = 0.0
_LAST_PUBLISH_WALL = 0.0
_PUSH_GATEWAY_UP = 0.0
_HEALTH_METRICS: dict[str, Any] = {}
_HEALTH_PROBE_AT = 0.0
_HEALTH_PREFIX = 'instant_monitoring'
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        delete_from_gateway,
        generate_latest,
        push_to_gateway,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    CONTENT_TYPE_LATEST = 'text/plain; version=0.0.4; charset=utf-8'
    CollectorRegistry = Counter = Gauge = None

    def generate_latest(*args, **kwargs):
        raise RuntimeError('prometheus_client not installed')

    def push_to_gateway(*args, **kwargs):
        raise RuntimeError('prometheus_client not installed')

    def delete_from_gateway(*args, **kwargs):
        raise RuntimeError('prometheus_client not installed')


class _PipelineMetricsHandler(BaseHTTPRequestHandler):
    """Serve a pre-rendered metrics snapshot so scrapes never block on merge work."""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = self.path.split('?', 1)[0]
        if path == '/metrics':
            with _METRICS_CACHE_LOCK:
                payload = _METRICS_CACHE
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(payload)
            return
        if path in ('/', '/health'):
            body = b'ok\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


class _ThreadingMetricsServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _active_registry() -> Any:
    global _PIPELINE_REGISTRY
    if _PIPELINE_REGISTRY is None:
        _PIPELINE_REGISTRY = CollectorRegistry()
    return _PIPELINE_REGISTRY


def _ensure_health_metrics(registry: Any) -> dict[str, Any]:
    global _HEALTH_METRICS
    if _HEALTH_METRICS or not _PROMETHEUS_AVAILABLE:
        return _HEALTH_METRICS
    p = _HEALTH_PREFIX
    _HEALTH_METRICS = {
        'monitoring_up': Gauge(f'{p}_up', 'Pipeline monitoring path healthy', registry=registry),
        'exporter_up': Gauge(f'{p}_exporter_up', 'Pipeline metrics HTTP server up', registry=registry),
        'pushgateway_up': Gauge(f'{p}_pushgateway_up', 'Last pushgateway push succeeded', registry=registry),
        'prometheus_up': Gauge(f'{p}_prometheus_up', 'Prometheus health probe', registry=registry),
        'grafana_connected': Gauge(f'{p}_grafana_connected', 'Grafana health probe', registry=registry),
        'metrics_age': Gauge(
            f'{p}_pipeline_metrics_age_seconds',
            'Seconds since last pipeline metrics publish',
            registry=registry,
        ),
        'last_push': Gauge(
            f'{p}_last_successful_push_timestamp',
            'Unix timestamp of last successful pushgateway push',
            registry=registry,
        ),
    }
    return _HEALTH_METRICS


def _probe_http(url: str, *, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False


def _update_health_probes(*, exporter: Optional['PipelineMetricsExporter'] = None) -> None:
    global _HEALTH_PROBE_AT, _LAST_PUBLISH_WALL, _PUSH_GATEWAY_UP
    health = _ensure_health_metrics(_active_registry())
    if not health:
        return
    now = time.monotonic()
    health['exporter_up'].set(1.0 if _SERVER_STARTED else 0.0)
    if _LAST_PUBLISH_WALL > 0:
        health['metrics_age'].set(max(time.time() - _LAST_PUBLISH_WALL, 0.0))
    else:
        health['metrics_age'].set(0.0)
    if _LAST_PUSH_SUCCESS > 0:
        health['last_push'].set(_LAST_PUSH_SUCCESS)
    push_target = PipelineMetricsExporter._pushgateway_target() if exporter else None
    if not push_target:
        health['pushgateway_up'].set(1.0)
    else:
        health['pushgateway_up'].set(_PUSH_GATEWAY_UP)
    if now >= _HEALTH_PROBE_AT:
        _HEALTH_PROBE_AT = now + 30.0
        skip_probe = os.environ.get('INSTANT_SKIP_METRICS_PROBE', '').strip().lower() in ('1', 'true', 'yes')
        if not skip_probe:
            prom_url = os.environ.get('INSTANT_PROMETHEUS_HEALTH_URL', 'http://127.0.0.1:9090/-/healthy')
            graf_url = os.environ.get('INSTANT_GRAFANA_HEALTH_URL', 'http://127.0.0.1:3001/api/health')
            health['prometheus_up'].set(1.0 if _probe_http(prom_url) else 0.0)
            health['grafana_connected'].set(1.0 if _probe_http(graf_url) else 0.0)
    exporter_ok = _SERVER_STARTED
    push_ok = not push_target or _PUSH_GATEWAY_UP >= 1.0
    health['monitoring_up'].set(1.0 if exporter_ok and push_ok else 0.0)


def _refresh_metrics_cache() -> None:
    global _METRICS_CACHE
    data = generate_latest(_active_registry())
    with _METRICS_CACHE_LOCK:
        _METRICS_CACHE = data


def _background_metrics_loop() -> None:
    cache_interval = float(os.environ.get('INSTANT_PIPELINE_METRICS_CACHE_SEC', '2'))
    push_interval = float(os.environ.get('INSTANT_PIPELINE_PUSH_SEC', '5'))
    next_cache = 0.0
    next_push = 0.0
    while True:
        now = time.monotonic()
        if now >= next_cache:
            try:
                inst = PipelineMetricsExporter._instance
                _update_health_probes(exporter=inst)
                _refresh_metrics_cache()
            except Exception as exc:
                logger.debug('Metrics cache refresh failed: %s', exc)
            next_cache = now + cache_interval
        if now >= next_push:
            inst = PipelineMetricsExporter._instance
            if inst is not None and inst._snapshot_ready:
                inst._push()
            next_push = now + push_interval
        time.sleep(0.25)


def _start_cached_http_server(port: int, addr: str = '0.0.0.0') -> None:
    global _SERVER_STARTED, _BACKGROUND_STARTED
    httpd = _ThreadingMetricsServer((addr, port), _PipelineMetricsHandler)
    threading.Thread(
        target=httpd.serve_forever,
        daemon=True,
        name='pipeline-metrics-http',
    ).start()
    _ensure_health_metrics(_active_registry())
    try:
        _update_health_probes()
        _refresh_metrics_cache()
    except Exception as exc:
        logger.debug('Initial metrics cache warmup failed: %s', exc)
    if not _BACKGROUND_STARTED:
        _BACKGROUND_STARTED = True
        threading.Thread(
            target=_background_metrics_loop,
            daemon=True,
            name='pipeline-metrics-bg',
        ).start()
    _SERVER_STARTED = True


class PipelineMetricsExporter:
    """Streams dataset quality pipeline metrics to Prometheus (port 9093 by default)."""

    _instance: Optional['PipelineMetricsExporter'] = None

    def __init__(self, *, corpus_id: str = 'default', prefix: str = 'instant_pipeline'):
        self.corpus_id = corpus_id
        self.prefix = prefix
        self._enabled = _PROMETHEUS_AVAILABLE
        self._merge_started_at: float | None = None
        self._scan_baseline: int = 0
        self._last_publish_scanned: int = 0
        self._last_publish_at: float | None = None
        self._snapshot_ready = False
        self._registry = _active_registry()
        self._metrics: dict[str, Any] = {}
        if not self._enabled:
            return
        lbl = ('corpus_id',)
        p = prefix
        reg = self._registry
        self._metrics['docs'] = Counter(
            f'{p}_documents_processed_total', 'Documents evaluated in merge', lbl, registry=reg
        )
        self._metrics['docs_kept'] = Gauge(
            f'{p}_documents_kept', 'Documents kept after quality gate', lbl, registry=reg
        )
        self._metrics['docs_scanned'] = Gauge(
            f'{p}_documents_scanned_total', 'Raw lines scanned in merge', lbl, registry=reg
        )
        self._metrics['keep_rate'] = Gauge(
            f'{p}_keep_rate', 'Fraction of scanned lines kept', lbl, registry=reg
        )
        self._metrics['duplicate_rate'] = Gauge(
            f'{p}_duplicate_rate', 'Duplicate reject rate', lbl, registry=reg
        )
        self._metrics['toxicity_rate'] = Gauge(f'{p}_toxicity_rate', 'Toxicity flag rate', lbl, registry=reg)
        self._metrics['pii_rate'] = Gauge(f'{p}_pii_rate', 'PII flag rate', lbl, registry=reg)
        self._metrics['language_early_rejected'] = Gauge(
            f'{p}_language_early_rejected', 'Documents rejected at early language gate', lbl, registry=reg
        )
        self._metrics['language_detection_cpu_sec'] = Gauge(
            f'{p}_language_detection_cpu_sec', 'Cumulative language detection CPU seconds', lbl, registry=reg
        )
        self._metrics['language_skipped_post_clean'] = Gauge(
            f'{p}_language_skipped_post_clean', 'Post-clean language detections skipped', lbl, registry=reg
        )
        self._metrics['corpus_score'] = Gauge(
            f'{p}_corpus_score_mean', 'Mean kept document quality score', lbl, registry=reg
        )
        self._metrics['docs_per_sec'] = Gauge(
            f'{p}_documents_per_second', 'Documents scanned per second during merge', lbl, registry=reg
        )
        self._metrics['language_share'] = Gauge(
            f'{p}_language_share', 'Language share of kept docs', lbl + ('language',), registry=reg
        )
        self._metrics['source_share'] = Gauge(
            f'{p}_source_share', 'Source share of kept docs', lbl + ('source',), registry=reg
        )
        self._metrics['docs_rejected'] = Gauge(
            f'{p}_documents_rejected', 'Documents rejected by quality gate', lbl, registry=reg
        )
        self._metrics['reject_rate'] = Gauge(
            f'{p}_reject_rate', 'Fraction of scanned lines rejected', lbl, registry=reg
        )
        self._metrics['semantic_reject_rate'] = Gauge(
            f'{p}_semantic_reject_rate', 'Semantic/near-dup reject rate', lbl, registry=reg
        )
        self._metrics['curator_reject_rate'] = Gauge(
            f'{p}_curator_reject_rate', 'Curriculum balance reject rate', lbl, registry=reg
        )
        self._metrics['embedding_candidates'] = Gauge(
            f'{p}_embedding_candidates_checked', 'Embedding dedup candidate pairs checked', lbl, registry=reg
        )
        self._metrics['embedding_clusters'] = Gauge(
            f'{p}_embedding_clusters', 'Embedding dedup cluster representatives', lbl, registry=reg
        )
        self._metrics['embedding_dup_removed'] = Gauge(
            f'{p}_embedding_duplicates_removed', 'Embedding semantic duplicates removed', lbl, registry=reg
        )
        self._metrics['embedding_keep_rate'] = Gauge(
            f'{p}_embedding_keep_rate', 'Embedding dedup keep rate', lbl, registry=reg
        )
        self._metrics['embedding_reject_rate'] = Gauge(
            f'{p}_embedding_reject_rate', 'Embedding semantic reject rate', lbl, registry=reg
        )
        self._metrics['embedding_threshold'] = Gauge(
            f'{p}_embedding_similarity_threshold', 'Adaptive embedding similarity threshold', lbl, registry=reg
        )
        self._metrics['embedding_provider_up'] = Gauge(
            f'{p}_embedding_provider_up', 'Embedding provider health', lbl, registry=reg
        )
        self._metrics['embedding_batches'] = Gauge(
            f'{p}_embedding_batches', 'Embedding inference batches executed', lbl, registry=reg
        )
        self._metrics['embedding_documents'] = Gauge(
            f'{p}_embedding_documents_embedded', 'Documents embedded by provider', lbl, registry=reg
        )
        self._metrics['embedding_latency_ms'] = Gauge(
            f'{p}_embedding_latency_ms', 'Cumulative embedding provider latency ms', lbl, registry=reg
        )
        self._metrics['embedding_cache_hits'] = Gauge(
            f'{p}_embedding_cache_hits', 'Embedding cache hits', lbl, registry=reg
        )
        self._metrics['embedding_cache_misses'] = Gauge(
            f'{p}_embedding_cache_misses', 'Embedding cache misses', lbl, registry=reg
        )
        self._metrics['embedding_failures'] = Gauge(
            f'{p}_embedding_failures', 'Embedding provider failures', lbl, registry=reg
        )
        self._metrics['embedding_memory_mb'] = Gauge(
            f'{p}_embedding_memory_mb', 'Embedding provider GPU memory MB', lbl, registry=reg
        )
        self._metrics['calibration_p10'] = Gauge(
            f'{p}_calibration_composite_p10', 'Calibration composite p10', lbl, registry=reg
        )
        self._metrics['calibration_p50'] = Gauge(
            f'{p}_calibration_composite_p50', 'Calibration composite p50', lbl, registry=reg
        )
        self._metrics['calibration_p90'] = Gauge(
            f'{p}_calibration_composite_p90', 'Calibration composite p90', lbl, registry=reg
        )
        self._metrics['score_variance'] = Gauge(
            f'{p}_score_variance', 'Calibration composite score variance', lbl, registry=reg
        )
        self._metrics['calibration_samples'] = Gauge(
            f'{p}_calibration_samples', 'Calibration reservoir sample count', lbl, registry=reg
        )
        self._metrics['worker_count'] = Gauge(
            f'{p}_worker_count', 'Configured merge worker count', lbl, registry=reg
        )
        self._metrics['active_workers'] = Gauge(
            f'{p}_active_workers', 'Active merge workers', lbl, registry=reg
        )
        self._metrics['queue_depth'] = Gauge(
            f'{p}_queue_depth', 'Merge read/pending queue depth', lbl, registry=reg
        )
        self._metrics['memory_rss_mb'] = Gauge(
            f'{p}_memory_rss_mb', 'Process peak RSS megabytes', lbl, registry=reg
        )
        self._metrics['cpu_utilization'] = Gauge(
            f'{p}_cpu_utilization', 'Host CPU utilization fraction', lbl, registry=reg
        )
        self._metrics['checkpoint_scanned'] = Gauge(
            f'{p}_checkpoint_scanned', 'Checkpoint cumulative scanned count', lbl, registry=reg
        )
        _ensure_health_metrics(reg)

    @classmethod
    def get(cls, *, corpus_id: str = 'default') -> 'PipelineMetricsExporter':
        if cls._instance is None or cls._instance.corpus_id != corpus_id:
            cls._instance = cls(corpus_id=corpus_id)
        return cls._instance

    @classmethod
    def begin_merge(cls, *, corpus_id: str, fresh: bool = False) -> 'PipelineMetricsExporter':
        global _PIPELINE_REGISTRY, _METRICS_CACHE
        if fresh:
            cls.clear_pushgateway(corpus_id)
        if fresh or cls._instance is None or cls._instance.corpus_id != corpus_id:
            _PIPELINE_REGISTRY = CollectorRegistry()
            _METRICS_CACHE = b''
            cls._instance = cls(corpus_id=corpus_id)
            cls._instance._seed_gauges()
        cls._instance._merge_started_at = time.perf_counter()
        cls.ensure_server()
        return cls._instance

    @classmethod
    def clear_pushgateway(cls, corpus_id: str) -> None:
        target = cls._pushgateway_target()
        if not target or not _PROMETHEUS_AVAILABLE:
            return
        try:
            delete_from_gateway(
                target,
                job='instant-pipeline',
                grouping_key={'corpus_id': corpus_id},
            )
            logger.info('Cleared stale pushgateway metrics for corpus_id=%s', corpus_id)
        except Exception as exc:
            logger.debug('Pushgateway clear failed (%s): %s', target, exc)

    @classmethod
    def ensure_server(cls, port: Optional[int] = None, addr: str = '0.0.0.0') -> None:
        if _SERVER_STARTED or not _PROMETHEUS_AVAILABLE:
            return
        if os.environ.get('INSTANT_PIPELINE_METRICS', '1').strip().lower() in ('0', 'false', 'no'):
            return
        port = port or int(os.environ.get('INSTANT_PIPELINE_METRICS_PORT', '9093'))
        try:
            _start_cached_http_server(port, addr=addr)
            logger.info('Pipeline Prometheus metrics at http://127.0.0.1:%s/metrics', port)
            push_target = PipelineMetricsExporter._pushgateway_target()
            if push_target:
                logger.info('Pipeline metrics pushgateway target: %s', push_target)
        except OSError as exc:
            logger.warning('Pipeline metrics server bind failed: %s', exc)

    def _cid(self) -> dict[str, str]:
        return {'corpus_id': self.corpus_id}

    @staticmethod
    def _pushgateway_target() -> str | None:
        raw = os.environ.get('INSTANT_PIPELINE_PUSHGATEWAY', '127.0.0.1:9094').strip()
        if raw.lower() in ('0', 'false', 'no', 'off'):
            return None
        return raw

    def _seed_gauges(self) -> None:
        if not self._enabled:
            return
        cid = self._cid()
        self._metrics['docs_kept'].labels(**cid).set(0)
        self._metrics['docs_scanned'].labels(**cid).set(0)
        self._metrics['keep_rate'].labels(**cid).set(0)
        self._metrics['duplicate_rate'].labels(**cid).set(0)
        self._metrics['toxicity_rate'].labels(**cid).set(0)
        self._metrics['pii_rate'].labels(**cid).set(0)
        self._metrics['corpus_score'].labels(**cid).set(0)
        self._metrics['docs_per_sec'].labels(**cid).set(0)
        self._metrics['docs_rejected'].labels(**cid).set(0)
        self._metrics['reject_rate'].labels(**cid).set(0)
        self._metrics['semantic_reject_rate'].labels(**cid).set(0)
        self._metrics['curator_reject_rate'].labels(**cid).set(0)
        self._metrics['embedding_candidates'].labels(**cid).set(0)
        self._metrics['embedding_clusters'].labels(**cid).set(0)
        self._metrics['embedding_dup_removed'].labels(**cid).set(0)
        self._metrics['embedding_keep_rate'].labels(**cid).set(0)
        self._metrics['embedding_reject_rate'].labels(**cid).set(0)
        self._metrics['embedding_threshold'].labels(**cid).set(0)
        self._metrics['embedding_provider_up'].labels(**cid).set(0)
        self._metrics['embedding_batches'].labels(**cid).set(0)
        self._metrics['embedding_documents'].labels(**cid).set(0)
        self._metrics['embedding_latency_ms'].labels(**cid).set(0)
        self._metrics['embedding_cache_hits'].labels(**cid).set(0)
        self._metrics['embedding_cache_misses'].labels(**cid).set(0)
        self._metrics['embedding_failures'].labels(**cid).set(0)
        self._metrics['embedding_memory_mb'].labels(**cid).set(0)
        self._metrics['calibration_p10'].labels(**cid).set(0)
        self._metrics['calibration_p50'].labels(**cid).set(0)
        self._metrics['calibration_p90'].labels(**cid).set(0)
        self._metrics['score_variance'].labels(**cid).set(0)
        self._metrics['calibration_samples'].labels(**cid).set(0)
        self._metrics['worker_count'].labels(**cid).set(0)
        self._metrics['active_workers'].labels(**cid).set(0)
        self._metrics['queue_depth'].labels(**cid).set(0)
        self._metrics['memory_rss_mb'].labels(**cid).set(0)
        self._metrics['cpu_utilization'].labels(**cid).set(0)
        self._metrics['checkpoint_scanned'].labels(**cid).set(0)
        self._snapshot_ready = True
        try:
            _refresh_metrics_cache()
        except Exception as exc:
            logger.debug('Initial metrics cache refresh failed: %s', exc)

    def _push(self) -> None:
        global _PUSH_BACKOFF_UNTIL, _PUSH_WARNED, _LAST_PUSH_SUCCESS, _PUSH_GATEWAY_UP
        target = self._pushgateway_target()
        if not target or not self._enabled:
            return
        now = time.monotonic()
        if now < _PUSH_BACKOFF_UNTIL:
            return
        try:
            push_to_gateway(
                target,
                job='instant-pipeline',
                registry=self._registry,
                grouping_key={'corpus_id': self.corpus_id},
            )
            _LAST_PUSH_SUCCESS = time.time()
            _PUSH_GATEWAY_UP = 1.0
            _PUSH_BACKOFF_UNTIL = 0.0
        except Exception as exc:
            _PUSH_GATEWAY_UP = 0.0
            backoff = float(os.environ.get('INSTANT_PIPELINE_PUSH_BACKOFF_SEC', '30'))
            _PUSH_BACKOFF_UNTIL = now + max(backoff, 1.0)
            if not _PUSH_WARNED:
                _PUSH_WARNED = True
                logger.warning(
                    'Pushgateway unavailable at %s (%s); local /metrics still served; '
                    'set INSTANT_PIPELINE_PUSHGATEWAY=off to silence',
                    target,
                    exc,
                )
            else:
                logger.debug('Pushgateway push failed (%s): %s', target, exc)

    def set_scan_baseline(self, scanned: int) -> None:
        baseline = max(int(scanned), 0)
        self._scan_baseline = baseline
        self._last_publish_scanned = baseline
        self._last_publish_at = time.perf_counter()

    def inc_document(self, n: int = 1) -> None:
        if self._enabled:
            if self._merge_started_at is None:
                self._merge_started_at = time.perf_counter()
            self._metrics['docs'].labels(**self._cid()).inc(n)

    def publish_gate_snapshot(
        self,
        gate: Any,
        *,
        merge_kept: int | None = None,
        merge_rejected: int | None = None,
        total_scanned: int | None = None,
        exact_duplicates: int | None = None,
        score_mean: float | None = None,
        reject_reasons: dict[str, int] | None = None,
        flush_every: int = 0,
        workers: int | None = None,
        active_workers: int | None = None,
        queue_depth: int | None = None,
        cpu_utilization_pct: float | None = None,
    ) -> None:
        if not self._enabled:
            return
        global _LAST_PUBLISH_WALL
        stats = gate.stats
        session_kept = max(int(stats.kept), 0)
        session_rejected = max(int(stats.rejected), 0)
        kept = max(int(merge_kept), 0) if merge_kept is not None else session_kept
        rejected = max(int(merge_rejected), 0) if merge_rejected is not None else session_rejected
        gate_total = max(kept + rejected, 1)
        qs = stats.to_dict()
        merged_rejects = dict(qs.get('reject_reasons') or {})
        if reject_reasons:
            for key, count in reject_reasons.items():
                merged_rejects[key] = int(merged_rejects.get(key, 0)) + int(count)
        session_dup = sum(
            int(merged_rejects.get(k, 0))
            for k in ('exact_dup', 'near_dup_fuzzy', 'near_dup_semantic', 'fuzzy_dup', 'semantic_dup', 'duplicate')
        ) + int(qs.get('downranked', 0))
        dup = max(int(exact_duplicates), 0) if exact_duplicates is not None else session_dup
        if dup <= 0 and merged_rejects.get('exact_dup'):
            dup = int(merged_rejects['exact_dup'])
        scanned = max(int(total_scanned), 0) if total_scanned is not None else 0
        dup_total = max(scanned, gate_total + dup, 1)
        session_scanned = max(scanned - self._scan_baseline, 0)
        keep_rate = kept / max(scanned, 1)
        reject_rate = rejected / max(scanned, 1)
        self._metrics['docs_kept'].labels(**self._cid()).set(kept)
        self._metrics['docs_scanned'].labels(**self._cid()).set(scanned)
        self._metrics['docs_rejected'].labels(**self._cid()).set(rejected)
        self._metrics['keep_rate'].labels(**self._cid()).set(keep_rate)
        self._metrics['reject_rate'].labels(**self._cid()).set(reject_rate)
        self._metrics['duplicate_rate'].labels(**self._cid()).set(dup / dup_total)
        self._metrics['checkpoint_scanned'].labels(**self._cid()).set(scanned)
        semantic_rejects = sum(
            int(merged_rejects.get(k, 0))
            for k in ('near_dup_semantic', 'semantic_dup', 'semantic')
        )
        curator_rejects = int(merged_rejects.get('curriculum_balance', 0))
        self._metrics['semantic_reject_rate'].labels(**self._cid()).set(
            semantic_rejects / max(scanned, 1)
        )
        embed_rejects = int(merged_rejects.get('near_dup_embed', 0))
        self._metrics['embedding_reject_rate'].labels(**self._cid()).set(
            embed_rejects / max(scanned, 1)
        )
        self._metrics['curator_reject_rate'].labels(**self._cid()).set(
            curator_rejects / max(scanned, 1)
        )
        if hasattr(gate, 'calibrator') and gate.calibrator is not None:
            cal = gate.calibrator.distribution_stats()
            self._metrics['calibration_p10'].labels(**self._cid()).set(
                float(cal.get('composite_p10', 0.0))
            )
            self._metrics['calibration_p50'].labels(**self._cid()).set(
                float(cal.get('composite_p50', 0.0))
            )
            self._metrics['calibration_p90'].labels(**self._cid()).set(
                float(cal.get('composite_p90', 0.0))
            )
            self._metrics['score_variance'].labels(**self._cid()).set(
                float(cal.get('score_variance', 0.0))
            )
            self._metrics['calibration_samples'].labels(**self._cid()).set(
                float(cal.get('reservoir_size', 0))
            )
        if workers is not None:
            self._metrics['worker_count'].labels(**self._cid()).set(max(int(workers), 0))
        if active_workers is not None:
            self._metrics['active_workers'].labels(**self._cid()).set(max(int(active_workers), 0))
        if queue_depth is not None:
            self._metrics['queue_depth'].labels(**self._cid()).set(max(int(queue_depth), 0))
        if cpu_utilization_pct is not None:
            self._metrics['cpu_utilization'].labels(**self._cid()).set(
                max(float(cpu_utilization_pct), 0.0) / 100.0
            )
        try:
            from indw.tools.reports.benchmark import peak_rss_mb

            self._metrics['memory_rss_mb'].labels(**self._cid()).set(peak_rss_mb())
        except Exception:
            pass
        kept_score = float(score_mean) if score_mean is not None else float(qs.get('score_mean', 0.0))
        if kept > 0 and kept_score <= 0.0 and score_mean is None:
            kept_score = float(qs.get('pre_filter_score_mean', 0.0))
        self._metrics['corpus_score'].labels(**self._cid()).set(kept_score if kept > 0 else 0.0)
        if self._merge_started_at is None:
            self._merge_started_at = time.perf_counter()
        elapsed = max(time.perf_counter() - self._merge_started_at, 1e-6)
        now = time.perf_counter()
        instant_dps = 0.0
        if self._last_publish_at is not None and scanned >= self._last_publish_scanned:
            window = max(now - self._last_publish_at, 1e-6)
            instant_dps = (scanned - self._last_publish_scanned) / window
        session_dps = session_scanned / elapsed if session_scanned > 0 else 0.0
        throughput_docs = session_scanned if session_scanned > 0 else max(session_kept + session_rejected, 0)
        if throughput_docs <= 0 and scanned > self._scan_baseline:
            throughput_docs = 1
        docs_per_sec = instant_dps if instant_dps > 0 else session_dps
        if docs_per_sec <= 0.0 and throughput_docs > 0:
            docs_per_sec = throughput_docs / elapsed
        self._metrics['docs_per_sec'].labels(**self._cid()).set(docs_per_sec)
        self._last_publish_scanned = scanned
        self._last_publish_at = now
        tox = gate.toxicity_stats
        if tox.documents_scanned > 0:
            rate = float(tox.rejected + tox.hard_rejected) / max(tox.documents_scanned, 1)
            self._metrics['toxicity_rate'].labels(**self._cid()).set(rate)
        pii = gate.pii_stats
        if pii.documents_scanned > 0:
            rate = float(pii.rejected + pii.hard_rejected) / max(pii.documents_scanned, 1)
            self._metrics['pii_rate'].labels(**self._cid()).set(rate)
        lang_stats = gate.language_stats
        self._metrics['language_early_rejected'].labels(**self._cid()).set(float(lang_stats.early_rejected))
        self._metrics['language_detection_cpu_sec'].labels(**self._cid()).set(float(lang_stats.detection_cpu_sec))
        self._metrics['language_skipped_post_clean'].labels(**self._cid()).set(float(lang_stats.skipped_post_clean))
        lang_dist = gate.lang_balancer.distribution()
        for lang, share in lang_dist.items():
            self._metrics['language_share'].labels(
                **self._cid(), language=str(lang)
            ).set(float(share))
        domain_dist = gate.domain_balancer.distribution()
        for source, share in domain_dist.items():
            self._metrics['source_share'].labels(
                **self._cid(), source=str(source)
            ).set(float(share))
        self._snapshot_ready = True
        _LAST_PUBLISH_WALL = time.time()
        if flush_every and kept % flush_every == 0:
            self.inc_document(0)
        try:
            _refresh_metrics_cache()
        except Exception as exc:
            logger.debug('Metrics cache refresh failed: %s', exc)

    def publish_final(
        self,
        gate: Any,
        report: Any,
        *,
        dedup_stats: Optional[dict[str, Any]] = None,
        merge_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self._enabled:
            return
        self.ensure_server()
        merge_kept = None
        merge_rejected = None
        total_scanned = None
        exact_duplicates = None
        if merge_stats:
            merge_kept = int(merge_stats.get('kept', 0))
            merge_rejected = int(merge_stats.get('rejected', 0))
            total_scanned = int(merge_stats.get('scanned', 0))
        if dedup_stats:
            exact_duplicates = int(dedup_stats.get('exact_duplicates', 0))
        self.publish_gate_snapshot(
            gate,
            merge_kept=merge_kept,
            merge_rejected=merge_rejected,
            total_scanned=total_scanned,
            exact_duplicates=exact_duplicates,
        )
        kept = max(int(merge_kept if merge_kept is not None else gate.stats.kept), 0)
        self._metrics['docs_kept'].labels(**self._cid()).set(kept)
        if dedup_stats:
            dupes = (
                int(dedup_stats.get('exact_duplicates', 0))
                + int(dedup_stats.get('fuzzy_duplicates', 0))
                + int(dedup_stats.get('semantic_duplicates', 0))
                + int(dedup_stats.get('embedding_duplicates', 0))
            )
            cid = self._cid()
            self._metrics['embedding_candidates'].labels(**cid).set(
                float(dedup_stats.get('embedding_candidates_checked', 0))
            )
            self._metrics['embedding_clusters'].labels(**cid).set(
                float(dedup_stats.get('embedding_clusters', 0))
            )
            self._metrics['embedding_dup_removed'].labels(**cid).set(
                float(dedup_stats.get('embedding_duplicates_removed', dedup_stats.get('embedding_duplicates', 0)))
            )
            self._metrics['embedding_keep_rate'].labels(**cid).set(
                float(dedup_stats.get('embedding_keep_rate', 0.0))
            )
            self._metrics['embedding_reject_rate'].labels(**cid).set(
                float(dedup_stats.get('embedding_reject_rate', 0.0))
            )
            self._metrics['embedding_threshold'].labels(**cid).set(
                float(dedup_stats.get('embedding_threshold', 0.0))
            )
            for key, metric in (
                ('embedding_provider_up', 'embedding_provider_up'),
                ('embedding_batches', 'embedding_batches'),
                ('embedding_documents', 'embedding_documents'),
                ('embedding_latency_ms', 'embedding_latency_ms'),
                ('embedding_cache_hits', 'embedding_cache_hits'),
                ('embedding_cache_misses', 'embedding_cache_misses'),
                ('embedding_failures', 'embedding_failures'),
                ('embedding_memory_mb', 'embedding_memory_mb'),
            ):
                if key in dedup_stats and key in self._metrics:
                    self._metrics[key].labels(**cid).set(float(dedup_stats[key]))
            rejected = max(int(merge_rejected if merge_rejected is not None else gate.stats.rejected), 0)
            total = max(kept + rejected + dupes, 1)
            self._metrics['duplicate_rate'].labels(**self._cid()).set(dupes / total)
        quality = report.to_dict().get('quality', {})
        self._metrics['corpus_score'].labels(**self._cid()).set(
            float(quality.get('score_mean', 0.0))
        )
        try:
            _refresh_metrics_cache()
        except Exception as exc:
            logger.debug('Final metrics cache refresh failed: %s', exc)
        self._push()
