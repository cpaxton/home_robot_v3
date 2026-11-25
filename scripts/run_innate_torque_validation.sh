#!/usr/bin/env bash
# Run the external innate-os MuJoCo torque-adapter smoke test.
#
# The innate-os checkout is NOT vendored in this repository. Set INNATE_OS_ROOT
# to an existing checkout (default /tmp/opencode/innate-os).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INNATE_OS_ROOT="${INNATE_OS_ROOT:-/tmp/opencode/innate-os}"
SIM_DIR="${INNATE_OS_ROOT}/sim"
OUTPUT="${1:-${ROOT}/logs/innate_torque_validation/report.json}"

if [[ ! -d "${SIM_DIR}" ]]; then
    cat >&2 <<EOF
innate-os not found at ${INNATE_OS_ROOT}

This test intentionally does not vendor innate-os. Clone it outside the Emet
checkout, then install its sim assets:

  git clone https://github.com/innate-inc/innate-os.git ${INNATE_OS_ROOT}
  cd ${INNATE_OS_ROOT}
  ./innate-sim assets
  cd sim && uv sync --group notebook

Then rerun:
  INNATE_OS_ROOT=${INNATE_OS_ROOT} ${BASH_SOURCE[0]}
EOF
    exit 2
fi

if [[ ! -d "${SIM_DIR}/assets" ]]; then
    echo "sim assets missing; running the external innate-os asset download..." >&2
    (cd "${INNATE_OS_ROOT}" && ./innate-sim assets)
fi

if [[ ! -x "${SIM_DIR}/.venv/bin/python" ]]; then
    echo "sim Python environment missing; running uv sync in ${SIM_DIR}..." >&2
    (cd "${SIM_DIR}" && uv sync --group notebook)
fi

mkdir -p "$(dirname "${OUTPUT}")"
cd "${SIM_DIR}"
MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python \
    "${ROOT}/scripts/validate_innate_torque_adapter.py" \
    --innate-sim "${SIM_DIR}" \
    --output "${OUTPUT}"

echo "wrote ${OUTPUT}"
