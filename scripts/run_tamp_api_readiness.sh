#!/usr/bin/env bash
# Managed, immutable-source readiness gate. Preserve all failed admission/replay outcomes.
set -euo pipefail
: "${EMET_JOB_ID:?Run through emet jobs --cpu-safe --gpu-exclusive}"
trial_root="${1:?Output directory required}"
source_registry="${2:?Original frozen registry required}"
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export EMET_UV_RUN=1 UV_NO_SYNC=1 EMET_ZMQ_STARTUP_TIMEOUT=120 EMET_MUJOCO_CTRL_DEBUG=1
export MPLCONFIGDIR=/tmp/emet-tamp-mpl
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$trial_root"
.venv/bin/python - "$source_registry" "$trial_root" <<'PY'
import json, subprocess, sys
from pathlib import Path
import yaml
source, root = Path(sys.argv[1]), Path(sys.argv[2])
ids = [f'tamp_v2_rby1_scene{s:02d}_cleanup_0' for s in (0, 2, 12)]
by_id = {row['id']: row for row in yaml.safe_load(source.read_text())['episodes']}
rows = []
for case in ids:
    row = dict(by_id[case])
    for key in ('fixture', 'clutter', 'robot_start_xy', 'goal_xy', 'episode_valid'):
        row.pop(key, None)
    rows.append(row)
(root/'candidate.yaml').write_text(yaml.safe_dump({'schema': 1, 'episodes': rows}, sort_keys=False))
(root/'protocol.json').write_text(json.dumps({'source': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'requested_cases': ids, 'repeats': 3, 'mode': 'GT/MCTS latch; separate live CHAT API smoke'},indent=2)+'\n')
PY
smoke_status=0
OUT_DIR="$trial_root/agent-tools" PROFILE=smoke TIMEOUT=900 bash scripts/run_tamp_agent_tools_gate.sh || smoke_status=$?
printf '%s\n' "$smoke_status" > "$trial_root/agent-tools-exit.txt"
admission_status=0
.venv/bin/python scripts/run_tamp_experiments.py --suite full --validate-fixtures \
    --registry "$trial_root/candidate.yaml" --output-dir "$trial_root/admission" || admission_status=$?
if [[ "$admission_status" -eq 0 && -f "$trial_root/admission/validated_registry.yaml" ]]; then
    cp "$trial_root/admission/validated_registry.yaml" "$trial_root/frozen_registry.yaml"
    for repeat in 1 2 3; do
        .venv/bin/python scripts/run_tamp_experiments.py --suite full \
            --registry "$trial_root/frozen_registry.yaml" --output-dir "$trial_root/repeat$repeat" || break
    done
fi
.venv/bin/python - "$trial_root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {'agent_tools_exit': int((root/'agent-tools-exit.txt').read_text()), 'phases': {}}
for phase in ('admission','repeat1','repeat2','repeat3'):
    path = root/phase/'summary.json'
    summary['phases'][phase] = json.loads(path.read_text()) if path.exists() else None
admission = summary['phases']['admission'] or {}
summary['ready'] = summary['agent_tools_exit'] == 0 and admission.get('status_counts',{}).get('fixture_admitted') == 3 and all(
    (summary['phases'][f'repeat{i}'] or {}).get('successful_cases') == 3 for i in (1,2,3))
(root/'readiness.json').write_text(json.dumps(summary,indent=2)+'\n')
raise SystemExit(0 if summary['ready'] else 1)
PY
