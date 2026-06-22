#!/usr/bin/env python3
"""Export docs/images/*.svg to matching PNG files."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs" / "images"


def main() -> int:
    svgs = sorted(ROOT.glob("*.svg"))
    if not svgs:
        print("no svg files", file=sys.stderr)
        return 1
    for svg in svgs:
        out = svg.with_suffix(".png")
        try:
            subprocess.run(
                ["rsvg-convert", "-w", "1200", str(svg), "-o", str(out)],
                check=True,
                capture_output=True,
            )
            print(out)
        except (FileNotFoundError, subprocess.CalledProcessError):
            # fallback: copy svg path documented; PNG optional for local dev
            print(f"skip {svg.name} (install rsvg-convert or use SVG in docs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
