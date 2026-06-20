from __future__ import annotations

import os


def configure_hf_fast(*, disable_progress_bars: bool = False) -> None:
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if disable_progress_bars:
        os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'
    else:
        os.environ.pop('HF_DATASETS_DISABLE_PROGRESS_BARS', None)
