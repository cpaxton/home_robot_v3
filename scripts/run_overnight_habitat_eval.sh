#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Compat wrapper. The HM-EQA overnight ladder is ``emet hmeqa overnight``:
# holdout-8 → gate → balanced-32, classic vs agentic Dynagraph H2H.
#
# This is not the old method-comparison overnight (canonical-8 / paper-20 /
# annotated-37 × static_graph + dynagraph via ``run_habitat_iter_subset.sh``).
# Those slices:
#   paper-20     IDS=$(seq -s, 0 19) METHOD=dynagraph ./scripts/run_habitat_iter_subset.sh
#   annotated-37 ./scripts/run_hmeqa_annotated37_h2h.sh
#   paper-113    ./scripts/run_hmeqa_paper113_h2h.sh
#
# SKIP_PHASES / OVERNIGHT_DEADLINE_HOURS / RUN_ANNOTATED37 are ignored.
# Resume: uv run emet hmeqa overnight --base ~/runs/emet/hmeqa_overnight_…
#
# Usage:
#   uv run emet jobs run --name hmeqa-overnight --need-mib 12000 --gpu-exclusive -- \
#     uv run emet hmeqa overnight
#   ./scripts/run_overnight_habitat_eval.sh
#   ./scripts/run_overnight_habitat_eval.sh --skip-bal32

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "${SKIP_PHASES:-}" || -n "${OVERNIGHT_DEADLINE_HOURS:-}" || -n "${RUN_ANNOTATED37:-}" ]]; then
    echo "WARNING: SKIP_PHASES / OVERNIGHT_DEADLINE_HOURS / RUN_ANNOTATED37 are ignored." >&2
    echo "This script now execs emet hmeqa overnight (classic vs agentic H2H)." >&2
    echo "Method-comparison slices: run_habitat_iter_subset.sh / run_hmeqa_annotated37_h2h.sh / run_hmeqa_paper113_h2h.sh" >&2
fi

echo "run_overnight_habitat_eval.sh → uv run emet hmeqa overnight $*" >&2
echo "Extra slices: run_hmeqa_annotated37_h2h.sh / run_hmeqa_paper113_h2h.sh / run_habitat_iter_subset.sh" >&2
exec uv run emet hmeqa overnight "$@"
