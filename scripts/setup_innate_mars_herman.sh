#!/usr/bin/env bash
# Herman (jetson1@herman) innate Mars bridge — wrapper around emet mars.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run emet mars start --ip herman --username jetson1 --deploy --preview
