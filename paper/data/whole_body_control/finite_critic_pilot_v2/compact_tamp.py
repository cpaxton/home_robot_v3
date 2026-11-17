"""Preserve real prefix identities and verify matching initial simulator plants."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path.cwd()
    output = root / 'paper/data/whole_body_control/finite_critic_pilot_v2'
    raw = root / 'logs/manipulation/finite-critic-pilot-v2'
    report = json.loads((raw / 'physical-comparison.json').read_text())
    compact = json.loads((output / 'results.json').read_text())
    fields = (
        'id', 'primitive', 'planning_scope', 'code', 'success', 'stop_confirmed',
        'safety_flags', 'inference_counts', 'full_pick_place_success',
        'initial_state_sha256', 'model_fingerprint', 'compiled_robot_identity',
        'terminal_pose', 'terminal_pose_in_contract', 'gripper_open', 'control_ticks',
    )
    baseline = report['cases']['tamp']['initial']
    for name, rows in report['cases']['tamp'].items():
        if len(rows) != len(baseline):
            raise RuntimeError('TAMP case count mismatch')
        for first, current in zip(baseline, rows, strict=True):
            for key in ('id', 'initial_state_sha256', 'model_fingerprint', 'compiled_robot_identity'):
                if first[key] != current[key]:
                    raise RuntimeError(f'TAMP comparison differs in {key}')
        compact['case_rows']['tamp'][name] = [{k: row[k] for k in fields} for row in rows]
    compact['tamp_initial_states_plants_and_ids_matched'] = True
    (output / 'results.json').write_text(json.dumps(compact, indent=2, allow_nan=False) + '\n')
    (output / 'sha256.json').write_text(json.dumps({
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != 'sha256.json'
    }, indent=2) + '\n')
    print('All 18 TAMP records preserve real IDs and matched initial states/plants')


if __name__ == '__main__':
    main()
