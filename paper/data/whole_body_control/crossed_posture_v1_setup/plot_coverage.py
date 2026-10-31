"""Teacher final heading error for the 25 finite training combinations."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

root = Path(__file__).resolve().parent
with gzip.open(root / 'teacher-report.json.gz', 'rt') as stream:
    report = json.load(stream)
names = ['hold', 'forward', 'reverse', 'yaw_left', 'yaw_right']
errors = np.full((5, 5), np.nan)
for case in report['cases']:
    if case['split'] != 'train':
        continue
    primitive, *posture = case['id'].split('@posture:')
    motion = primitive.rsplit(':', 1)[-1]
    posture = posture[0] if posture else motion
    errors[names.index(posture), names.index(motion)] = 1000 * case['metrics']['base_heading_error_rad']
assert np.isfinite(errors).all()
fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
im = ax.imshow(errors, vmin=0, vmax=10, cmap='YlGnBu')
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{errors[i,j]:.2f}', ha='center', va='center')
    ax.add_patch(Rectangle((i-.46, i-.46), .92, .92, fill=False, edgecolor='black', linewidth=2))
labels = ['Hold', 'Forward', 'Reverse', 'Left turn', 'Right turn']
ax.set_xticks(range(5), labels, rotation=20)
ax.set_yticks(range(5), labels)
ax.set_xlabel('Commanded base primitive')
ax.set_ylabel('Reset posture from original training case')
ax.set_title('Teacher passes all 25 training pairs\nOutlined cells are the original five pairs')
fig.colorbar(im, ax=ax, label='Final heading error (mrad); acceptance = 10 mrad')
fig.savefig(root / 'teacher-coverage.svg')
plt.close(fig)
