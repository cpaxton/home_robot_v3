#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Plot recorded initial/final footprints, never invented intermediate trajectories."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.data.read_text())
    cases = {case['condition']: case for case in raw['cases']}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ('two_goal_causal', 'expanded_forward_buffer'),
                             ('Two-goal transfer: complete', 'Swap: partial (1 of 3 transfers)'), strict=True):
        case = cases[key]
        for region in case['task_config']['regions']:
            low, high = region['bounds']
            ax.add_patch(Rectangle(low, high[0]-low[0], high[1]-low[1],
                                   fill=False, linestyle=':', edgecolor='#777777'))
            ax.text((low[0]+high[0])/2, low[1]-.015, region['name'], ha='center', va='top', fontsize=7)
        for body, color, label in [('object1', '#376daf', 'Cube'), ('object2', '#bc3f39', 'Cylinder')]:
            for phase in ('initial', 'final'):
                low, high = case[phase][body]['bounds_xy']
                ax.add_patch(Rectangle(low, high[0]-low[0], high[1]-low[1], color=color,
                                       fill=phase == 'final', alpha=.8 if phase == 'final' else .45,
                                       linestyle='--' if phase == 'initial' else '-'))
            final = case['final'][body]['bounds_xy']
            ax.text(-.12, (final[0][1]+final[1][1])/2, label, color=color, ha='right', va='center', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set(xlabel='World x (m)', ylabel='World y (m)', xlim=(-.28,.33), ylim=(-1.12,-.45), aspect='equal')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.12)
    fig.suptitle('Measured footprints: dashed = initial, filled = final', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
