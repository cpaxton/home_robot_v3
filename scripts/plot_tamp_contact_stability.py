#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Plot measured loaded-lift rotation; preserve failed native diagnostic episodes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def lift_samples(rows, body):
    """Do not plot a later placement rejection as part of a completed lift.

    Early diagnostic traces used only the generic payload_retention phase.
    New captures also name the active motion phase explicitly.
    """
    object_rows = [r for r in rows if r['object_body'] == body]
    closure = next((r for r in object_rows if r['event']['phase'] == 'gripper_settled'), None)
    complete = any(r['event']['phase'] == 'lift_complete' for r in object_rows)
    lift = [r for r in object_rows if r['event']['phase'] in ('lift', 'lift_complete')
            or (r['event']['phase'] == 'payload_retention'
                and (r.get('motion_phase') == 'lift' or (r.get('motion_phase') is None and not complete)))]
    lift = [r for r in lift if 'rotation_residual_rad' in r]
    later_rejection = complete and any(r['event']['phase'] == 'payload_retention' and r not in lift for r in object_rows)
    return closure, lift, later_rejection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='Frozen batch root with episodes/ and frozen_batch.json')
    parser.add_argument('--output', type=Path, required=True, help='Output figure basename')
    args = parser.parse_args()
    batch_path = args.runs / 'frozen_batch.json'
    batch = json.loads(batch_path.read_text())
    inputs, episodes = [batch_path], {}
    for item in batch['episodes']:
        root = args.runs / 'episodes' / item['id']
        result = json.loads((root / 'summary.json').read_text())
        if result['run_status'] not in ('complete', 'infrastructure_error'):
            parser.error('All declared episodes must be terminal')
        trace = root / 'contact.jsonl'
        rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        episodes[item['id']] = (result, rows)
        inputs.append(root / 'summary.json')
        if trace.exists():
            inputs.append(trace)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), constrained_layout=True, sharey=True)
    colors = {'baseline': '#D55E00', 'fine_yaw': '#0072B2'}
    samples = []
    limit = float(batch['retention_rotation_limit_rad']) * 180 / np.pi
    for i, layout in enumerate(('control', 'shift')):
        for j, (body, label) in enumerate((('object2', 'Cylinder'), ('object1', 'Cube'))):
            ax = axes[i, j]
            for policy, suffix in (('baseline', 'baseline'), ('fine_yaw', 'fine-yaw')):
                result, rows = episodes[layout + '-' + suffix]
                closure, lift, later_rejection = lift_samples(rows, body)
                if not closure or not lift:
                    ax.text(.03, .85 - .12 * (policy == 'fine_yaw'), f'{policy}: no lift samples', color=colors[policy], transform=ax.transAxes)
                    continue
                height = [(r['object_position_m'][2] - closure['object_position_m'][2]) * 100 for r in lift]
                angle = [r['rotation_residual_rad'] * 180 / np.pi for r in lift]
                ax.plot(height, angle, '.-', lw=1.2, ms=3, color=colors[policy])
                for r, h, a in zip(lift, height, angle, strict=True):
                    rejected = r['event']['phase'] == 'payload_retention' and r['event'].get('accepted') is False
                    if rejected:
                        ax.plot(h, a, 'x', color=colors[policy], ms=9, mew=2)
                    samples.append({'layout': layout, 'policy': policy, 'body': body, 'sim_time': r['sim_time'],
                                    'lift_from_closure_cm': h, 'rotation_residual_deg': a,
                                    'translation_residual_mm': r['translation_residual_m'] * 1000, 'rejected': rejected})
                if later_rejection:
                    policy_label = 'Finer yaw' if policy == 'fine_yaw' else 'Existing yaw'
                    ax.text(.34, .82, policy_label + ': lift completed\nLater rejection: placement', color=colors[policy],
                            transform=ax.transAxes, fontsize=8)
            ax.axhline(limit, color='#777777', ls='--', lw=1)
            ax.set_title(f'{"Initial X unchanged" if layout == "control" else "Initial X +15 mm"} / {label}')
            ax.set(xlabel='Object lift from closure (cm)', ylabel='Relative rotation (degrees)', ylim=(0, None))
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=.15)
    handles = [Line2D([], [], color=colors[p], label=label) for p, label in
               (('baseline', 'Existing yaw search'), ('fine_yaw', 'Finer yaw search'))]
    handles += [Line2D([], [], color='#777777', ls='--', label='Unchanged rotation limit'),
                Line2D([], [], color='black', marker='x', ls='', label='Retention rejection')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Native task-tool diagnostics: sampled lift rotation\nOne live development replay per condition; no agent score', fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.pdf', '.png', '.svg'):
        fig.savefig(args.output.with_suffix(extension), dpi=180)
    with args.output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['layout', 'policy', 'body', 'sim_time', 'lift_from_closure_cm',
                               'rotation_residual_deg', 'translation_residual_mm', 'rejected'])
        writer.writeheader()
        writer.writerows(samples)
    provenance = {'scope': 'Measured lift samples, connected for readability; independent live resets; no agent rescore',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]}
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
