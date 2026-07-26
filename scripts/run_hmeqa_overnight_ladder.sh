#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Thin shim — prefer the dogfood CLI:
#   uv run emet hmeqa overnight
#
# Kept so old STATUS / jobs ``cmd`` lines still work. When already inside
# ``emet jobs`` (EMET_JOB_ID set), ``emet hmeqa overnight`` runs the
# orchestrator in-process and does not nest another job.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run emet hmeqa overnight "$@"
