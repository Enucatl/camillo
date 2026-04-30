#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports
uv run --extra dev scalene run \
  --outfile reports/scalene-performance.json \
  .venv/bin/pytest --- -m performance
uv run --extra dev scalene view --cli reports/scalene-performance.json \
  > reports/scalene-performance.txt
