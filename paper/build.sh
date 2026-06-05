#!/usr/bin/env bash
# Build Dynagraph CoRL paper PDF.
#
# From repo root:  ./paper/build.sh [--clean]
# From paper/:    ./build.sh [--clean]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

clean=0
if [[ "${1:-}" == "--clean" ]]; then
  clean=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--clean]" >&2
  exit 1
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk not found. Install a TeX distribution, for example:" >&2
  echo "  sudo apt install texlive-latex-extra texlive-bibtex-extra latexmk" >&2
  exit 1
fi

if [[ "$clean" -eq 1 ]]; then
  latexmk -C main.tex
  echo "Cleaned build artifacts in $SCRIPT_DIR"
  exit 0
fi

latexmk -pdf -interaction=nonstopmode -file-line-error main.tex

pdf="$SCRIPT_DIR/main.pdf"
if [[ -f "$pdf" ]]; then
  echo "Wrote $pdf"
else
  echo "Build finished but $pdf not found (check latexmk output above)." >&2
  exit 1
fi
