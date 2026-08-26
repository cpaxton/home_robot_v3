#!/usr/bin/env bash
# Run 5-qid HM-EQA count/clock canary at a specific git commit (git worktree).
#
# Usage (from any checkout):
#   BISECT_SHA=23efa534 ./scripts/run_hmeqa_bisect_canary.sh
#   BISECT_SHA=290e54e5 QUESTION_IDS=12,47,48,86,93 ./scripts/run_hmeqa_bisect_canary.sh
#
# Env: same as run_hmeqa_countclock_slice.sh plus BISECT_SHA (required).

set -euo pipefail

SHA="${BISECT_SHA:?set BISECT_SHA to the commit under test}"
SHORT="${SHA:0:8}"
ROOT="$(git rev-parse --show-toplevel)"
WT="${BISECT_WORKTREE:-${ROOT}/../home_robot_bisect_${SHORT}}"

if [[ ! -d "${WT}/.git" ]]; then
  echo "Creating worktree ${WT} @ ${SHA}"
  git worktree add -f "${WT}" "${SHA}"
fi

cd "${WT}"
echo "Bisect canary: commit=$(git rev-parse --short HEAD) worktree=${WT}"

export EMET_ALLOW_SDPA_ATTN="${EMET_ALLOW_SDPA_ATTN:-1}"
export RESUME="${RESUME:-0}"
export OUTPUT_PROFILE="${OUTPUT_PROFILE:-lean}"
export QUESTION_IDS="${QUESTION_IDS:-12,47,48,86,93}"
export RUN_ID="${RUN_ID:-bisect_${SHORT}}"

exec ./scripts/run_hmeqa_countclock_slice.sh
