# Mars expert workspace coverage

This pilot measures coverage of archived simulator training episodes on fixed
grids. Each episode contributes one endpoint vote. Each path contributes one
vote divided evenly among its distinct sampled cells, so a longer dwell
does not increase a cell's weight.

- **Goal and path occupancy** count the cells reached and traversed.
- **Effective cells** measure how broadly the samples are spread after accounting
  for imbalance. **Distance to uniform** compares that distribution with equal
  cell probability across the declared workspace.
- **Start-to-goal cell pairs**, path length, orientation travel and nearest
  goal-orientation gaps distinguish datasets with similar endpoint occupancy.
- **Goal weights** assign equal probability to each occupied cell and divide
  that probability across its episodes. They can guide future sampling.
  Empty cells require new proposals and teacher demonstrations.

The base grid covers cells whose centers are within 20 cm of the starting
base frame. Hand grids use the listed task frames and keep reachability
unclassified. The coupled report tracks the hand relative to the moving base,
the same hand path relative to the starting base, and the measured base path.
Carry uses the initial hand frame.

| Pilot input | Episodes | Goal cells | Path cells |
| --- | ---: | ---: | ---: |
| Archived base | 4 | 4 / 52 | 7 / 52 |
| Base spline proposals | 41 | 22 / 52 | 35 / 52 |
| Archived arm | 22 | 7 | 24 |
| Coupled hand, moving base frame | 21 | 1 | 1 |
| Coupled hand, starting base frame | 21 | 6 | 6 |
| Coupled base | 21 | 7 / 52 | 10 / 52 |
| Carry | 95 | 7 / 64 | 7 / 64 |
| TAMP approach | 4 | 1 | 1 |
| Lift | 10 | 2 | 2 |
| Preplace | 5 | 4 | 9 |

The base proposal pilot stratifies bearing and radius and generates
differential-drive wheel splines. It checks the retimed duration and the
radius at 201 samples. The accepted wheel knots and rejected proposal IDs
are recorded in [the proposal report](base-spline-proposals.json). This is
kinematic evidence. Physics execution, teacher admission, and training on
these proposals remain pending.

## Reproduce

Use the archived simulator receipts and arrays under the existing `logs/eval`
and `logs/recovery` trees. The [baseline report](report.json) records the
source scope and grid for each family. Raw arrays stay in the experiment
archive.

```sh
export PYTHONPATH=src
python scripts/build_mars_workspace_pilot.py --logs logs --output /tmp/mars-workspace.json
python scripts/measure_mars_workspace.py --workspace /tmp/mars-workspace.json --output /tmp/mars-coverage.json
python scripts/propose_mars_base_splines.py --workspace /tmp/mars-workspace.json --output /tmp/mars-base-proposals.json
```
