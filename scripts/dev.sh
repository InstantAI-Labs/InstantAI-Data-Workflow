#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${root}/src:${root}${PYTHONPATH:+:${PYTHONPATH}}"
exec "$@"
