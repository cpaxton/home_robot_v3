"""Compact complete physical readouts; preserve IDs without geometry/trace blobs."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path.cwd()
    out = root / 'paper/data/whole_body_control/quiet_reference_pilot_v1'
    raw = root / 'logs/manipulation/quiet-reference-pilot-v1'
    report = json.loads((raw / 'physical-comparison.json').read_text())
    compact = {k: v for k, v in report.items() if k != 'cases'}
    compact['raw_report_sha256'] = hashlib.sha256((raw / 'physical-comparison.json').read_bytes()).hexdigest()
    compact['case_rows'] = {}
    metrics = ('ee_position_error', 'ee_orientation_error', 'settled_seconds', 'time_to_first_settle',
               'peak_hold_base_drift_m', 'peak_base_linear_jerk_m_s3', 'peak_base_yaw_jerk_rad_s3', 'peak_contact_force')
    for cohort, controllers in report['cases'].items():
        compact['case_rows'][cohort] = {}
        for name, rows in controllers.items():
            if cohort == 'tamp':
                for a, b in zip(controllers['initial'], rows, strict=True):
                    for key in ('id', 'initial_state_sha256', 'model_fingerprint', 'compiled_robot_identity'):
                        if a[key] != b[key]:
                            raise RuntimeError('unmatched TAMP ' + key)
                keys = ('id', 'primitive', 'planning_scope', 'code', 'success', 'stop_confirmed', 'safety_flags',
                        'inference_counts', 'initial_state_sha256', 'model_fingerprint', 'compiled_robot_identity',
                        'full_pick_place_success', 'terminal_pose_in_contract', 'gripper_open')
                compact['case_rows'][cohort][name] = [{k: r[k] for k in keys} for r in rows]
            else:
                keys = ('id', 'seed', 'success', 'code', 'stop_confirmed', 'safety_flags', 'initial_state_sha256',
                        'model_fingerprint', 'reference_sha256')
                compact['case_rows'][cohort][name] = [{k: r[k] for k in keys} | {
                    'hand_target_active': r['metrics']['hand_target_active'],
                    'metrics': {k: r['metrics'][k] for k in metrics}} for r in rows]
    compact['runs'] = {}
    for arm in report['protocol']['reward_arms']:
        metadata = json.loads((raw / arm / 'run.json').read_text())
        updates = [json.loads(line) for line in (raw / arm / 'updates.jsonl').read_text().splitlines()]
        episodes = [json.loads(line) for line in (raw / arm / 'episodes.jsonl').read_text().splitlines()]
        compact['runs'][arm] = metadata | {
            'actor_optimizer_steps': sum(x.get('optimizer_steps', 0) for x in updates),
            'actor_kl_stops': sum(x.get('actor_kl_early_stop', False) for x in updates),
            'reduced_noise_decisions': sum(x['quiet_reference_reduced_decisions'] for x in updates),
            'general_completed_episodes': sum(x.get('training_family') == 'general' for x in episodes),
            'general_successful_episodes': sum(x.get('training_family') == 'general' and x.get('code') == 'ok' for x in episodes),
            'training_wheel_lift_episodes': sum(x.get('code') == 'wheel_lift' for x in episodes),
        }
    old = json.loads((root / 'logs/manipulation/finite-critic-pilot-v2/finite_independent/run.json').read_text())
    compact['ordinary_control_weights_exactly_reproduce_previous_independent_run'] = (
        compact['runs']['ordinary_noise']['final_weights_sha256'] == old['final_weights_sha256'])
    (out / 'results.json').write_text(json.dumps(compact, indent=2, allow_nan=False) + '\n')
    receipt = json.loads((root / 'logs/eval/quiet-reference-regressions-v1/full.receipt.json').read_text())
    (out / 'status.json').write_text(json.dumps({
        'format': 'quiet-reference-status-v1', 'phase': 'complete', 'runtime_revision': report['protocol']['runtime_revision'],
        'full_regression_tests': len(receipt['test_ids']), 'full_gate_seconds': receipt['finished_unix'] - receipt['started_unix'],
        'full_xml_sha256': receipt['xml_sha256'], 'ppo_started': True, 'complete': True,
        'hardware_used': False, 'safety_promoted': False, 'decision': 'reject reduced-noise candidate: new contacts and accuracy regression',
    }, indent=2) + '\n')
    (out / 'sha256.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir()) if p.is_file() and p.name != 'sha256.json'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
