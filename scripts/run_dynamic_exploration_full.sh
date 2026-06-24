#!/usr/bin/env bash
# Dynamic exploration only (shortcut). For the full paper queue use run_large_paper_eval.sh.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
exec ./scripts/run_large_paper_eval.sh dynamic-explore
