# Spatially varied Mars tracking demonstrations

This collection extends the Mars simulator training bank for base, arm, and
coupled tracking. It uses the same measurement grids as the
[first workspace pilot](../mars_workspace_coverage_v1/README.md).

Base cases use the pilot's differential-drive wheel splines and a separately
seeded audit proposal pool. Both splits use the same base reset fixture.
Arm and coupled cases span four reset postures and several target joint
directions with independent training and audit seeds. Coupled trajectories
vary the direction and amount of base travel and yaw while the arm moves.

The collector uses the existing sampled geometry admission, task executor,
and stop check. It also checks base-only path error against a 35 mm position
bound and the existing 5° heading bound. Arm and coupled cases retain the
3 mm and 0.02 rad hand path criteria. The training arrays contain only moving
ticks from admitted training episodes. Audit trajectories contribute to
coverage and outcome reports.

## Results

| Family | Goal cells, archive → combined | Path cells, archive → combined | Train admitted | Audit passed |
| --- | ---: | ---: | ---: | ---: |
| Base | 4 → 22 of 52 | 7 → 35 of 52 | 41/41 | 45/45 |
| Arm | 7 → 16 | 24 → 27 | 36/40 | 14/16 |
| Coupled hand, moving base frame | 1 → 15 | 1 → 23 | 37/40 | 14/16 |

The 114 admitted training episodes contain 43,956 moving frames. The 11
rejections exhausted geometric sampling. Each admitted episode confirmed
a stop with zero execution safety flags. The arm grids still have
unclassified reachability. The audit measured teacher feasibility.

See the [case manifest](manifest.json) and [collection receipt](receipt.json)
for per-case outcomes and dataset digests.

## Reproduce the collection

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
python scripts/collect_mars_workspace.py \
  --workspace "$WORKSPACE_BASELINE" \
  --proposals paper/data/whole_body_control/mars_workspace_coverage_v1/base-spline-proposals.json \
  --output "$WORKSPACE_BANK" --workers 8
```

Use the `workspace.json` built by the first pilot as `WORKSPACE_BASELINE`.
The output contains a manifest, per-episode receipts, a combined report, and
SHA-256-bound training arrays for each family. Raw arrays are experiment
artifacts.

## Use the bank

```sh
python scripts/train_mars_distilled.py \
  --work "$DISTILL_INPUT" --output "$DISTILL_OUTPUT" \
  --workspace-bank "$WORKSPACE_BANK/report.json"
```

For each offline base, arm, or coupled minibatch, the trainer selects the
new bank with probability 0.5. Within that bank it draws an occupied goal
cell uniformly, then an episode and a moving tick uniformly. The other half of
offline tracking draws uses the original pool. Online corrections use
their existing probability. Carry, TAMP approach, lift,
and preplace use their existing banks.

A 40-update CPU smoke run exercised weighted replay on the collected arrays.
