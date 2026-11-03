#!/bin/bash
# Численные CSV/рисунки подготовить tools/build_report_data.py.
# Артефакты XeLaTeX сохраняются отдельно от исходников, PDF — в output/pdf.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
work=${SKIPOD_REPORT_BUILD_DIR:-${TMPDIR:-/tmp}/skipod-latex-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$work" "$root/output/pdf"
uv run --quiet "$root/tools/report_tables.py"
(
    cd "$root/report"
    xelatex -interaction=nonstopmode -halt-on-error -output-directory="$work" report.tex
    xelatex -interaction=nonstopmode -halt-on-error -output-directory="$work" report.tex
)
cp "$work/report.pdf" "$root/output/pdf/skipod-task1-report.pdf"
printf '%s\n' "$root/output/pdf/skipod-task1-report.pdf"
