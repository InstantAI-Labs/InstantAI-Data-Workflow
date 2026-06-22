"""Load quality config from YAML instead of defaults."""
import yaml
from pathlib import Path

from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality

cfg_path = Path("configs/filtering/quality_smoke_5mb.yaml")
cfg = QualityPipelineConfig.from_dict(yaml.safe_load(cfg_path.read_text(encoding="utf-8")))

merge_with_quality(
    Path("./examples/raw"),
    Path("./examples/out/filtered.jsonl"),
    quality_config=cfg,
    work_dir=Path("./examples/work-custom"),
    fresh=True,
    workers=2,
)
