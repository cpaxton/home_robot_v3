#!/usr/bin/env bash
# Sequential head-to-head on the fixed 8-question HM-EQA subset:
#   1) dynagraph  (SigLIP-grounded CONFIRMED_MEMORY + SigLIP-guided exploration override)
#   2) graph_eqa  (clean baseline: no SigLIP grounding, no exploration override)
# Resumes each (skips already-done question_ids) so it is safe to re-run after a kill.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IDS="${IDS:-3,14,17,28,31,35,81,94}"
TIMEOUT="${TIMEOUT:-3000}"

echo "############ PHASE 1: dynagraph (SigLIP) ############"
TAG="${DG_TAG:-cmp_dynagraph}" IDS="$IDS" METHOD=dynagraph TIMEOUT="$TIMEOUT" \
    ./scripts/run_habitat_iter_subset.sh || true

echo "############ PHASE 2: graph_eqa (baseline) ############"
TAG="${BL_TAG:-cmp_graph_eqa}" IDS="$IDS" METHOD=graph_eqa TIMEOUT="$TIMEOUT" \
    ./scripts/run_habitat_iter_subset.sh || true

echo "############ DONE ############"
