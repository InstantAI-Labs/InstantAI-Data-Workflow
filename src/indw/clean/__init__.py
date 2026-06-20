__all__ = [
    'CorpusCleaningPipeline',
    'CleaningResult',
    'extract_row_text',
    'final_pass_jsonl_row',
    'process_jsonl_row',
    'row_text_key',
]

_LAZY = {
    'CorpusCleaningPipeline': ('indw.clean.corpus', 'CorpusCleaningPipeline'),
    'CleaningResult': ('indw.clean.corpus', 'CleaningResult'),
    'extract_row_text': ('indw.clean.corpus', 'extract_row_text'),
    'final_pass_jsonl_row': ('indw.clean.corpus', 'final_pass_jsonl_row'),
    'process_jsonl_row': ('indw.clean.corpus', 'process_jsonl_row'),
    'row_text_key': ('indw.clean.corpus', 'row_text_key'),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
