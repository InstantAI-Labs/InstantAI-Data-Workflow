from __future__ import annotations

import re
from dataclasses import dataclass

from indw.clean.artifact.decompose import compute_layout
from indw.clean.semantic.classifier import SemanticChunkClassifier
from indw.clean.semantic.config import EmbeddedHeuristics, SemanticCleaningConfig
from indw.clean.semantic.routing import SectionRouter
from indw.clean.semantic.scoring import compute_section_signals
from indw.clean.semantic.thresholds import get_threshold_calibrator
from indw.clean.artifact.evidence import DocumentFeatureExtractor

_PIPE = re.compile(r'\|')
_URL = re.compile(r'https?://|www\.', re.I)

@dataclass
class EmbeddedArtifactStats:
    lines_removed: int = 0
    prefix_lines_removed: int = 0
    suffix_lines_removed: int = 0
    inline_blocks_removed: int = 0
    chars_removed: int = 0

def _is_pipe_nav_line(line: str, emb: EmbeddedHeuristics) -> bool:
    s = line.strip()
    if not s or len(s) > emb.pipe_nav_max_len:
        return False
    return s.count('|') >= 2 and len(s.split('|')) >= 3

def _is_url_heavy_line(line: str, emb: EmbeddedHeuristics) -> bool:
    s = line.strip()
    if not s or len(s) > emb.url_line_max_len:
        return False
    url_chars = sum(len(m.group(0)) for m in _URL.finditer(s))
    return url_chars / max(len(s), 1) >= emb.url_line_ratio

def _line_artifact_score(
    line: str,
    router: SectionRouter,
    classifier: SemanticChunkClassifier,
    emb: EmbeddedHeuristics,
) -> float:
    if not line.strip():
        return 0.0
    cls = classifier.classify(line.strip(), position_ratio=emb.position_mid)
    signals = compute_section_signals(cls, section_role='body')
    raw = DocumentFeatureExtractor().extract(line)
    noise = (
        signals.navigation_likelihood * emb.line_nav_weight
        + signals.metadata_likelihood * emb.line_meta_weight
        + signals.boilerplate_likelihood * emb.line_boilerplate_weight
        + signals.promotional_likelihood * emb.line_promo_weight
        + signals.noise_level * emb.line_noise_weight
    )
    if _is_pipe_nav_line(line, emb):
        noise = min(1.0, noise + emb.pipe_nav_boost)
    if _is_url_heavy_line(line, emb):
        noise = min(1.0, noise + emb.url_line_boost)
    if raw.contact_token_ratio > emb.contact_token_ratio:
        noise = min(1.0, noise + raw.contact_token_ratio)
    knowledge = signals.knowledge_value + signals.educational_value + signals.technical_value
    return max(0.0, noise - knowledge * emb.knowledge_dampen_line)

def _paragraph_noise_score(
    text: str,
    classifier: SemanticChunkClassifier,
    router: SectionRouter,
    emb: EmbeddedHeuristics,
) -> float:
    if not text.strip():
        return 0.0
    cls = classifier.classify(text.strip())
    signals = compute_section_signals(cls, section_role='body')
    raw = DocumentFeatureExtractor().extract(text)
    noise = (
        signals.navigation_likelihood
        + signals.metadata_likelihood
        + signals.boilerplate_likelihood
        + signals.promotional_likelihood
        + signals.noise_level
    ) / emb.paragraph_signal_divisor
    if raw.anchor_density > emb.anchor_density:
        noise = min(1.0, noise + raw.anchor_density)
    if _is_url_heavy_line(text, emb) or _URL.search(text):
        noise = min(1.0, noise + emb.url_noise_boost)
    rep = cls.evidence.representation if cls.evidence else None
    if rep:
        noise = min(1.0, noise + rep.promotional * emb.promo_boost + rep.transactional * emb.transactional_boost)
    if cls.evidence and not cls.evidence.preserve:
        noise = min(1.0, noise + emb.no_preserve_boost)
    if cls.utility < emb.utility_floor:
        noise = min(1.0, noise + (emb.utility_floor - cls.utility))
    knowledge = signals.knowledge_value + signals.educational_value + signals.technical_value
    return max(0.0, noise - knowledge * emb.knowledge_dampen)

def strip_leading_paragraphs(
    text: str,
    *,
    config: SemanticCleaningConfig | None = None,
) -> tuple[str, int]:
    if not text or not text.strip():
        return text, 0
    cfg = config or SemanticCleaningConfig()
    emb = cfg.embedded
    classifier = SemanticChunkClassifier()
    router = SectionRouter(cfg, get_threshold_calibrator())
    paras = [p.strip() for p in re.split(r'\n\s*\n+', text.strip()) if p.strip()]
    if len(paras) < 2:
        return text, 0

    removed = 0
    while len(paras) > 1:
        lead = paras[0]
        rest = '\n\n'.join(paras[1:])
        lead_cls = classifier.classify(lead, position_ratio=emb.position_lead)
        rest_cls = classifier.classify(rest, position_ratio=emb.position_mid)
        lead_raw = DocumentFeatureExtractor().extract(lead)
        edu_markers = lead_raw.copula_def_hits + lead_raw.fact_relation_hits + lead_raw.step_line_hits
        if edu_markers >= emb.edu_marker_min or (lead_cls.evidence and lead_cls.evidence.preserve):
            break
        if lead_raw.word_count >= emb.lead_word_min and lead_raw.avg_line_len >= emb.lead_avg_line_min:
            break
        if lead_cls.utility >= emb.lead_utility_keep and lead_raw.word_count >= emb.lead_keep_word_min:
            break
        noise = _paragraph_noise_score(lead, classifier, router, emb)
        decision = router.route(
            lead,
            lead_cls,
            section_role='metadata' if noise > emb.metadata_noise_floor else 'body',
            position_ratio=emb.position_lead,
        )
        rest_better = rest_cls.utility > lead_cls.utility + emb.rest_utility_delta
        if decision.action == 'REMOVE' or noise >= emb.lead_noise_remove:
            paras.pop(0)
            removed += 1
            continue
        if rest_better and lead_cls.utility < emb.lead_utility_floor and noise >= emb.lead_noise_rest_better:
            paras.pop(0)
            removed += 1
            continue
        if noise >= emb.lead_short_noise and len(lead) < emb.lead_short_chars and lead_cls.utility < emb.lead_utility_short:
            paras.pop(0)
            removed += 1
            continue
        break
    return '\n\n'.join(paras), removed

def strip_prefix_sentences(
    text: str,
    *,
    config: SemanticCleaningConfig | None = None,
    max_sentences: int | None = None,
) -> tuple[str, int]:
    cfg = config or SemanticCleaningConfig()
    emb = cfg.embedded
    limit = max_sentences if max_sentences is not None else emb.prefix_max_sentences
    if not text or len(text) < emb.prefix_min_text:
        return text, 0
    classifier = SemanticChunkClassifier()
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(parts) < 3:
        return text, 0

    removed = 0
    while len(parts) > 2 and removed < limit:
        lead = parts[0]
        if len(lead) < emb.prefix_min_sent_len:
            parts.pop(0)
            removed += 1
            continue
        noise = _paragraph_noise_score(
            lead,
            classifier,
            SectionRouter(cfg, get_threshold_calibrator()),
            emb,
        )
        if noise < emb.prefix_noise_min:
            break
        parts.pop(0)
        removed += 1
    if removed == 0:
        return text, 0
    return ' '.join(parts).strip(), removed

def _prose_edge_lines(text: str) -> tuple[list[str], int, int]:
    lines = text.splitlines()
    if not lines:
        return [], 0, 0
    if '```' not in text:
        return lines, 0, len(lines)

    prose_lines: list[str] = []
    start_idx = 0
    end_idx = len(lines)
    in_fence = False
    first_fence = None
    last_fence_end = len(lines)

    for i, line in enumerate(lines):
        if line.strip().startswith('```'):
            if not in_fence:
                if first_fence is None:
                    first_fence = i
                in_fence = True
            else:
                in_fence = False
                last_fence_end = i + 1
            continue
        if not in_fence:
            if first_fence is None:
                prose_lines.append(line)
                start_idx = i + 1
            elif i >= last_fence_end:
                prose_lines.append(line)

    prefix_n = first_fence if first_fence is not None else len(lines)
    suffix_start = last_fence_end if '```' in text else len(lines)
    return lines, prefix_n, suffix_start

def strip_edge_artifact_lines(
    text: str,
    *,
    config: SemanticCleaningConfig | None = None,
    max_edge_scan: int = 16,
) -> tuple[str, EmbeddedArtifactStats]:
    if not text or not text.strip():
        return text, EmbeddedArtifactStats()

    cfg = config or SemanticCleaningConfig()
    emb = cfg.embedded
    router = SectionRouter(cfg, get_threshold_calibrator())
    classifier = SemanticChunkClassifier()
    lines = text.splitlines()
    if len(lines) < 2:
        return text, EmbeddedArtifactStats()

    has_fence = '```' in text
    _, prefix_end, suffix_start = _prose_edge_lines(text)
    stats = EmbeddedArtifactStats()
    start = 0
    for i in range(min(max_edge_scan, prefix_end)):
        line = lines[i]
        if not line.strip():
            start = i + 1
            continue
        score = _line_artifact_score(line, router, classifier, emb)
        cls = classifier.classify(line, position_ratio=i / max(len(lines), 1))
        if cls.evidence and cls.evidence.preserve:
            break
        decision = router.route(
            line,
            cls,
            section_role='navigation' if _is_pipe_nav_line(line, emb) else 'body',
            position_ratio=i / max(len(lines), 1),
        )
        if decision.action == 'REMOVE' or score > emb.edge_line_remove:
            start = i + 1
            continue
        if score > emb.edge_line_early and i < emb.edge_early_scan and not has_fence:
            start = i + 1
            continue
        break

    end = len(lines)
    suffix_lo = max(start, suffix_start)
    for j in range(1, min(max_edge_scan, len(lines) - suffix_lo) + 1):
        i = len(lines) - j
        if i < suffix_lo:
            break
        line = lines[i]
        if not line.strip():
            end = i
            continue
        score = _line_artifact_score(line, router, classifier, emb)
        cls = classifier.classify(line, position_ratio=i / max(len(lines), 1))
        if has_fence and (cls.evidence and cls.evidence.preserve):
            break
        decision = router.route(
            line,
            cls,
            section_role='footer' if i / max(len(lines), 1) > emb.position_footer else 'body',
            position_ratio=i / max(len(lines), 1),
        )
        suffix_thr = emb.suffix_threshold_fence if has_fence else emb.suffix_threshold_plain
        if (decision.action == 'REMOVE' or score > suffix_thr) and not (
            has_fence and cls.utility > emb.suffix_utility_preserve
        ):
            end = i
            continue
        break

    if start == 0 and end == len(lines):
        return text, stats

    kept = lines[start:end]
    out = '\n'.join(kept).strip()
    stats.prefix_lines_removed = start
    stats.suffix_lines_removed = len(lines) - end
    stats.lines_removed = start + (len(lines) - end)
    stats.chars_removed = len(text) - len(out)
    return out, stats

def split_inline_artifact_blocks(
    text: str,
    *,
    config: SemanticCleaningConfig | None = None,
) -> list[str]:
    if not text or '\n' not in text:
        return [text] if text else []

    emb = (config or SemanticCleaningConfig()).embedded
    lines = text.splitlines()
    blocks: list[str] = []
    buf: list[str] = []

    for line in lines:
        nav = _is_pipe_nav_line(line, emb)
        url_heavy = _is_url_heavy_line(line, emb)
        if (nav or url_heavy) and buf:
            blocks.append('\n'.join(buf).strip())
            buf = [line] if line.strip() else []
        elif nav or url_heavy:
            if line.strip():
                buf = [line]
        else:
            if buf and (_is_pipe_nav_line(buf[0], emb) or _is_url_heavy_line(buf[0], emb)):
                blocks.append('\n'.join(buf).strip())
                buf = []
            buf.append(line)

    if buf:
        blocks.append('\n'.join(buf).strip())
    return [b for b in blocks if b]

def strip_embedded_artifacts(
    text: str,
    *,
    config: SemanticCleaningConfig | None = None,
) -> tuple[str, EmbeddedArtifactStats]:
    cfg = config or SemanticCleaningConfig()
    emb = cfg.embedded
    router = SectionRouter(cfg, get_threshold_calibrator())
    classifier = SemanticChunkClassifier()

    working, para_removed = strip_leading_paragraphs(text, config=cfg)
    stats = EmbeddedArtifactStats()
    stats.prefix_lines_removed = para_removed

    if para_removed == 0 and len(working) > cfg.embedded.strip_prefix_min_len:
        working, sent_removed = strip_prefix_sentences(working, config=cfg)
        stats.prefix_lines_removed += sent_removed

    working, edge_stats = strip_edge_artifact_lines(working, config=cfg)
    stats.prefix_lines_removed += edge_stats.prefix_lines_removed
    stats.suffix_lines_removed = edge_stats.suffix_lines_removed
    stats.lines_removed += edge_stats.lines_removed + para_removed
    stats.chars_removed += max(0, len(text) - len(working))

    parts = split_inline_artifact_blocks(working, config=cfg)
    if len(parts) <= 1:
        return working, stats

    kept: list[str] = []
    for part in parts:
        layout = compute_layout(part)
        role = 'navigation' if _is_pipe_nav_line(part.splitlines()[0], emb) else 'body'
        cls = classifier.classify(part)
        decision = router.route(part, cls, section_role=role, position_ratio=cfg.embedded.position_mid)
        if decision.action == 'REMOVE':
            stats.inline_blocks_removed += 1
            stats.chars_removed += len(part)
            continue
        if decision.action == 'KEEP_AFTER_CLEANING':
            from indw.clean.semantic.clean import clean_section_text
            part, _ = clean_section_text(part, role=role)
            if not part.strip():
                stats.inline_blocks_removed += 1
                continue
        kept.append(part.strip())

    out = '\n\n'.join(kept).strip() if kept else working
    if not out:
        out = working
    struct_re = re.compile(
        rf'(?is)^(?:'
        rf'table of contents\s*|'
        rf'enter a word for the dictionary definition\.?\s*|'
        rf'from the collaborative international dictionary[^:]{0,{emb.struct_prefix_max_chars}}:\s*'
        rf')+',
    )
    trimmed = struct_re.sub('', out).strip()
    if trimmed:
        stats.chars_removed += len(out) - len(trimmed)
        out = trimmed
    return out, stats
