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

## Incremental live-bridge ladder

`scripts/probe_live_object_route.py` uses the same fixture with the production
ZMQ bridge, physics stepping, and teleport navigation explicitly disabled:

```bash
/home/cpaxton/src/home_robot_v3/.venv/bin/python scripts/probe_live_object_route.py \
  --config configs/ovmm/stationary_rby1.yaml \
  --output-dir /tmp/emet-live-known-route
```

Use the same environment as above and run serially (the managed job wrapper can
use `--need-mib 3000 --cpu-safe`; this stage loads no learned models). The
`live_probe` config defines a ten-second hold, a +10-degree turn and return,
then a 20 cm approach and return. The probe stops at the first failed gate.
It records actual world-frame base poses, uprightness, optical-axis errors,
command receipts, and calibrated RGB-D at every hold sample and waypoint.
`result.json` reports stage gates; `observations.jsonl` retains measurements.

This is **not wheel-driven locomotion or balance validation**: production idle
control pins the free base and navigation drives holonomic base velocity.
Upper-body joints remain dynamic. The fixture explicitly requests 0.02 m /
0.03 rad arrival tolerances through the server constructor; ordinary bridge
defaults remain 0.07 m / 0.15 rad. Inspect actual measured angles, not just
the pass flag. No learned object-search success is implied by
passing this route. The next steps are learned replay of these saved views,
then one occlusion/distractor change, then autonomous approach/search, each
with fixed targets and budgets before trying a household scene.

The first live hold (`20260908_161740_f19c70`) stopped before navigation:
base drift was zero but optical tilt error reached 0.190 rad (10.9 degrees).
The position command had reported success. The asset compensated gravity on
torso/arm links but not attached cameras/fingers; MuJoCo gravity compensation
is per body. Completing that existing ideal-compensation model gives an
isolated ten-second, named-base-pinned physics-test tilt error below one degree. The new test
requires less than one degree, without pinning the upper-body joints. This
does not calibrate the model's inferred sensor masses against real hardware.

The first post-fix live route (`20260908_162446_2cb3fa`) passed the **old loose
gates**, but actually turned only 1.45 degrees for a 10-degree request and
approached 13 cm for a 20 cm request. It is not a precision-control pass.
Camera pitch error still reached 6.23 degrees. A separate initialization bug
was also fixed: updating the free-base reset pose must not copy articulated
joint angles into `qpos0`, which changes MuJoCo's kinematic references.

The stricter rerun (`20260908_162836_044d79`, output
`/tmp/emet-live-known-object-route-v5-20260908`) **failed hold** at 0.10873 rad
camera pitch error, with zero measured base drift; no precision route was
attempted. The reference-coordinate fix did not resolve this residual live
error. Remaining investigation is bridge initialization/base contact/holding
versus the isolated model, before learned replay or harder search scenes.
Do not treat the earlier loose-gate result as a known-good navigation baseline.

43 focused fixture, load-reference, navigation-clamp, and optical-axis tests
pass (one skip). No full sweep or learned OVMM success is claimed.
