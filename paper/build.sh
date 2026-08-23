#!/usr/bin/env bash
# Build Dynagraph CoRL paper PDF.
#
# From repo root:  ./paper/build.sh [--clean]
# From paper/:    ./build.sh [--clean]
#
# Uses local latexmk when available; otherwise Docker (texlive/texlive:latest).
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

run_latexmk() {
    latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
}

run_latexmk_clean() {
    latexmk -C main.tex
}

run_docker_latexmk() {
    local uid gid
    uid="$(id -u)"
    gid="$(id -g)"
    docker run --rm \
        --user "${uid}:${gid}" \
        -v "$SCRIPT_DIR:/work" \
        -w /work \
        texlive/texlive:latest \
        latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
}

run_docker_clean() {
    local uid gid
    uid="$(id -u)"
    gid="$(id -g)"
    docker run --rm \
        --user "${uid}:${gid}" \
        -v "$SCRIPT_DIR:/work" \
        -w /work \
        texlive/texlive:latest \
        latexmk -C main.tex
}

if [[ "$clean" -eq 1 ]]; then
    if command -v latexmk >/dev/null 2>&1; then
        run_latexmk_clean
    elif command -v docker >/dev/null 2>&1; then
        run_docker_clean
    else
        echo "Neither latexmk nor docker found; remove *.aux, *.log, *.pdf manually." >&2
        exit 1
    fi
    echo "Cleaned build artifacts in $SCRIPT_DIR"
    exit 0
fi

if command -v latexmk >/dev/null 2>&1; then
    run_latexmk
elif command -v docker >/dev/null 2>&1; then
    echo "latexmk not found locally; building via Docker (texlive/texlive:latest)…" >&2
    run_docker_latexmk
else
    echo "latexmk not found. Install TeX or Docker:" >&2
    echo "  sudo apt install texlive-latex-extra texlive-bibtex-extra latexmk" >&2
    echo "  # or use Docker: ./paper/build.sh (auto-detects docker)" >&2
    exit 1
fi

pdf="$SCRIPT_DIR/main.pdf"
if [[ -f "$pdf" ]]; then
    echo "Wrote $pdf"
else
    echo "Build finished but $pdf not found (check latexmk output above)." >&2
    exit 1
fi
