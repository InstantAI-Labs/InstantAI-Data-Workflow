"""Minimal merge example — requires: pip install -e ."""
from pathlib import Path

from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality

raw_dir = Path("./examples/raw")
out_path = Path("./examples/out/filtered.jsonl")
work_dir = Path("./examples/work")

cfg = QualityPipelineConfig()
merge_with_quality(
    raw_dir,
    out_path,
    quality_config=cfg,
    work_dir=work_dir,
    fresh=True,
    resume=False,
    workers=1,
)
