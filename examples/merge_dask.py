"""Dask backend example — requires: pip install -e ".[distributed]" """
import os
from pathlib import Path

from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality

os.environ["INSTANT_PIPELINE_BACKEND"] = "dask"
# os.environ["DASK_SCHEDULER_ADDRESS"] = "tcp://scheduler:8786"

merge_with_quality(
    Path("./examples/raw"),
    Path("./examples/out/filtered.jsonl"),
    quality_config=QualityPipelineConfig(),
    work_dir=Path("./examples/work-dask"),
    fresh=True,
    workers=4,
)
