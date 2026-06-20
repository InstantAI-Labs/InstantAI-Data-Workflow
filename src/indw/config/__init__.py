from indw.config.defaults import (
    DEFAULT_MERGE_CHUNK_SIZE,
    DEFAULT_PIPELINE_SPEC,
    DEFAULT_QUALITY_SPEC,
    DEFAULT_WRITE_BUFFER_BYTES,
    MERGE_CHECKPOINT_INTERVAL,
    MIN_CHARS_AFTER_CLEAN,
    MIN_CHARS_AFTER_REPAIR,
    MIN_CHARS_FINAL,
    MIN_CHARS_GATE,
)
from indw.config.validation import (
    ConfigValidationError,
    validate_pipeline_policy,
    validate_quality_config,
)

__all__ = [
    'ConfigValidationError',
    'DEFAULT_MERGE_CHUNK_SIZE',
    'DEFAULT_PIPELINE_SPEC',
    'DEFAULT_QUALITY_SPEC',
    'DEFAULT_WRITE_BUFFER_BYTES',
    'MERGE_CHECKPOINT_INTERVAL',
    'MIN_CHARS_AFTER_CLEAN',
    'MIN_CHARS_AFTER_REPAIR',
    'MIN_CHARS_FINAL',
    'MIN_CHARS_GATE',
    'PipelineConfigContext',
    'resolve_quality_config',
    'validate_pipeline_policy',
    'validate_quality_config',
]

def __getattr__(name: str):
    if name == 'PipelineConfigContext':
        from indw.config.resolve import PipelineConfigContext

        return PipelineConfigContext
    if name == 'resolve_quality_config':
        from indw.config.resolve import resolve_quality_config

        return resolve_quality_config
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
