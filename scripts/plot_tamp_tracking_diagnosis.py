#!/usr/bin/env python3
"""Plot paired tracking diagnostics; these are not task-success measurements."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(before, after, output):
    old, new = json.loads(before.read_text()), json.loads(after.read_text())
    def values(report, contacts):
        rows = sorted((r for r in report['rows'] if r['contacts_enabled'] == contacts), key=lambda r: r['attempt'])
        return rows
    old_on, old_off, new_off = values(old, True), values(old, False), values(new, False)
    identities = [r['attempt'] for r in old_on]
    if not identities or any([r['attempt'] for r in rows] != identities for rows in (old_off, new_off)):
        raise ValueError('paired attempt IDs must match')
    for rows in (old_off, new_off):
        if any(a['base_xyt'] != b['base_xyt'] for a, b in zip(old_on, rows, strict=True)):
            raise ValueError('paired base poses must match')
    x = np.arange(len(identities))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    for ax, groups, title, scale, unit in (
        (axes[0], [(old_on, 'Scene contacts'), (old_off, 'Contacts disabled')],
         'Original base hold: contact ablation', 1, 'm'),
        (axes[1], [(old_off, 'Pose reset'), (new_off, 'Solver support')],
         'No-contact tracking: base support', 1000, 'mm'),
    ):
        for index, (rows, label) in enumerate(groups):
            y = [r['ee_tracking_error_m'] * scale for r in rows]
            bars = ax.bar(x + (index - .5) * .36, y, .36, label=label)
            ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
        ax.set_xticks(x, [f'Attempt {i + 1}' for i in identities])
        ax.set_ylabel(f'EE tracking error ({unit})')
        ax.set_title(title, fontsize=10)
        ax.margins(y=.22)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, loc='upper right')
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(output / f'tamp_contact_tracking.{suffix}', dpi=300)
    plt.close(fig)
    (output / 'caption.txt').write_text(
        'Reconstructed RBY1 potato, apple, and kettle pregrasp commands (attempts 1–3). '
        'Left: wall–base contact explains the large original tracking errors. '
        'Right: replacing per-step floating-base pose resets with solver-based stationary support '
        'reduces free-space tracking offset. Errors compare measured EE pose with FK of commanded '
        'joints, not task completion. Contacts-disabled trials are diagnostic ablations only. '
        'The first two approach endpoints intersect a wall and remain rejected, even with improved tracking. '
        'Each bar is one deterministic reconstruction, not a reliability estimate.\n'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.before, args.after, args.output)
