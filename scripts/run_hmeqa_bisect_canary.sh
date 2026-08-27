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
PARENT_ROOT="${BISECT_PARENT:-${ROOT}}"
WT="${BISECT_WORKTREE:-${ROOT}/../home_robot_bisect_${SHORT}}"

if [[ ! -e "${WT}/.git" ]]; then
  echo "Creating worktree ${WT} @ ${SHA}"
  git worktree add -f "${WT}" "${SHA}"
fi

# Worktrees must not uv-sync in isolation (submodules + duplicate venvs fail).
# Share parent envs and reinstall editable emet at the bisect commit.
unset VIRTUAL_ENV UV_PROJECT
for d in .venv .venv-habitat; do
  parent="${PARENT_ROOT}/${d}"
  target="${WT}/${d}"
  if [[ ! -e "${parent}" ]]; then
    continue
  fi
  if [[ -L "${target}" ]]; then
    continue
  fi
  if [[ -e "${target}" ]]; then
    # A prior failed bisect may have left a partial uv-created .venv in the worktree.
    rm -rf "${target}"
  fi
  ln -s "${parent}" "${target}"
done
# Git worktrees do not init submodules; symlink populated third_party from parent.
for sub in segment-anything-2 robocasa robosuite ok-robot; do
  src="${PARENT_ROOT}/third_party/${sub}"
  dst="${WT}/third_party/${sub}"
  if [[ -d "${src}" && ! -e "${dst}/pyproject.toml" && ! -e "${dst}/setup.py" ]]; then
    rm -rf "${dst}"
    ln -s "${src}" "${dst}"
  fi
done
PY_HAB="${WT}/.venv-habitat/bin/python"
if [[ ! -x "${PY_HAB}" ]]; then
  echo "FATAL: missing ${PY_HAB}; run ./scripts/install_habitat.sh in ${PARENT_ROOT}" >&2
  exit 2
fi
echo "Reinstalling editable emet @ $(git -C "${WT}" rev-parse --short HEAD) into habitat venv"
"${PY_HAB}" -m pip install -q --no-deps -e "${WT}"
"${PY_HAB}" -m pip install -q --no-deps -e "${WT}/packages/emet_habitat"
# pip install overwrites emet-habitat with a plain entrypoint; restore libstdc++ wrapper.
HAB_ENV="$(cd "$(dirname "${PY_HAB}")/.." && pwd)"
cat > "${HAB_ENV}/bin/emet-habitat" <<'WRAP'
#!/usr/bin/env bash
# Restored by run_hmeqa_bisect_canary.sh (install_habitat.sh sets LD_LIBRARY_PATH).
ENV_PREFIX="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
exec "$ENV_PREFIX/bin/python" -m emet_habitat.cli "$@"
WRAP
chmod +x "${HAB_ENV}/bin/emet-habitat"

cd "${WT}"
echo "Bisect canary: commit=$(git rev-parse --short HEAD) worktree=${WT}"

# Older slice scripts gate VRAM via `uv run`, which tries to build a fresh worktree venv.
PY_GATE="${WT}/.venv/bin/python"
SLICE="${WT}/scripts/run_hmeqa_countclock_slice.sh"
if [[ -x "${PY_GATE}" ]] && grep -q 'VRAM_OK="$(uv run python' "${SLICE}" 2>/dev/null; then
  sed -i "s|VRAM_OK=\"\$(uv run python -c \"|VRAM_OK=\"\$(\"${PY_GATE}\" -c \"|" "${SLICE}"
fi

export EMET_ALLOW_SDPA_ATTN="${EMET_ALLOW_SDPA_ATTN:-1}"
export RESUME="${RESUME:-0}"
export OUTPUT_PROFILE="${OUTPUT_PROFILE:-lean}"
export QUESTION_IDS="${QUESTION_IDS:-12,47,48,86,93}"
export RUN_ID="${RUN_ID:-bisect_${SHORT}}"
export HMEQA_SEED="${HMEQA_SEED:-42}"

if [[ -n "${HMEQA_SEED}" ]]; then
  echo "Seeding HM-EQA RNGs: HMEQA_SEED=${HMEQA_SEED}"
  "${PY_HAB}" -c "import os; from emet.eval.ovmm_find_phase import set_find_phase_run_seed; set_find_phase_run_seed(int(os.environ['HMEQA_SEED']))"
fi

exec ./scripts/run_hmeqa_countclock_slice.sh
