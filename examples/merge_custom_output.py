"""Write filtered output to a custom path and separate work directory."""
from pathlib import Path

from indw.filter.spec.quality import QualityPipelineConfig
from indw.schedule import merge_with_quality

raw = Path("./examples/raw")
out = Path("./examples/out/custom/filtered.jsonl")
work = Path("./examples/work/custom-run")

out.parent.mkdir(parents=True, exist_ok=True)
work.mkdir(parents=True, exist_ok=True)

merge_with_quality(
    raw,
    out,
    quality_config=QualityPipelineConfig(),
    work_dir=work,
    fresh=True,
    workers=1,
)

print(f"output: {out}")
print(f"artifacts: {work}")
