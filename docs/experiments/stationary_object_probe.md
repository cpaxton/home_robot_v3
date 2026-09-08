# Stationary object/perception sanity check

This is a deliberately easy, **kinematically held** asset test: the configured
robot camera faces a 16 cm blue box and red cylinder on a table. No navigation,
exploration, bridge, physics stepping, graph, or VLM router is involved.
The `rby1` backend currently resolves to a **Galaxea R1 proxy MJCF**, not native
RB-Y1 geometry. Passing this test is not validation of RB-Y1 hardware or dynamics.

The actual robot camera and production head-camera geometry mask render RGB-D.
Simulator segmentation checks target visibility, but is never an input to YOLOE
or SigLIP. Query-conditioned detection and RGB-D conversion reuse the lazy-query
controller's implementation. Dense SigLIP is measured on the same image; this
is one-frame representation testing, **not a full voxel-map retrieval test**.

## Run

On the development workstation, from `/tmp/emet-query-memory`:

```bash
export PYTHONPATH=/tmp/emet-query-memory/src
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/cpaxton/src/home_robot_v3/.venv/bin/python scripts/probe_stationary_objects.py \
  --config configs/ovmm/stationary_rby1.yaml \
  --output-dir /tmp/emet-stationary-demo --learned
```

Run only when other GPU experiments are stopped, or use the existing exclusive
`emet jobs run --need-mib 12000 --cpu-safe` wrapper. Omit `--learned` for an
asset/rendering-only check. The fixture fixes the base and joint coordinates;
it does not ask an agent to discover or navigate to these poses.

Outputs include `rgb.png`, `overview.png`, cached RGB-D/calibration/GT segmentation
in `frame.npz`, learned detector masks, dense features, and `result.json`.
Exit success requires both targets to be visibly rendered. With `--learned`, it
also requires admitted detections within the diagnostic 15 cm center-distance
tolerance and no admitted detection for the absent green mug. Dense retrieval
scores are diagnostics, not a separate pass claim. The usual 0.12 detector
admission threshold is unchanged.

## Recorded result: 2026-09-08

Job `20260908_155513_e5ce54`, output
`/tmp/emet-stationary-rby1-learned-20260908`: **passed**.

| Query | Visible GT pixels | Detector confidence | Position error | Dense SigLIP maximum |
| --- | ---: | ---: | ---: | ---: |
| Blue cube | 2623 | 0.948 | 0.083 m | 0.176 |
| Red cylinder | 2099 | 0.266 | 0.064 m | 0.214 |
| Green mug (absent) | — | No detection | — | 0.111 |

The scene isolates a working visual path. It does not explain all of the failed
household-search run: objects here are larger, separated, unoccluded, and viewed
from an explicitly configured pose. In particular, a SigLIP-only 0.21 presence
gate would miss this blue cube despite its strong detector hit. Predicted 3D
bounds also include table contamination even though median position is good;
do not interpret this as manipulation-ready object geometry.

The next isolation step is the same view through the live bridge/held posture,
then measured approach motion, before reintroducing search. The seven fixture
and optical-axis tests pass. No full sweep or learned OVMM success is claimed.
