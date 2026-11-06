# Fault stop holds the first target instead of following sag

`ManipulationControl.stop()` now latches its first measured arm target. Repeated
fault calls keep that target, zero motion requests and base bias, and preserve
the latched grip command. A later release reference cannot revise it. A first
fault during an authorized release freezes measured jaw position once; invalid
jaw feedback retains the previous finite command. No fault is cleared.

Previously, each call reset the target to current measured position. During
loaded stopping this let the actuator target follow the moving arm. All six new
behavior tests fail against `b3115331`: Mars/Stretch arm/grip latching, invalid
jaw feedback, and the three recorded loaded fault resets. The corrected behavior
passes the 52-test control/runtime subset plus all seven final fault tests,
including binding the reset-state bytes into the source regression receipt.

| Diagnostic loaded reset | Before | After measured stop | Hold target drift |
|---|---|---:|---:|
| left,20g,seed0 | Unconfirmed | 0.10s | 0rad |
| left,50g,seed0 | Unconfirmed | 0.25s | 0rad |
| right,20g,seed0 | Unconfirmed | 0.20s | 0rad |

The original 0.75s stop budget and measured speed thresholds stay unchanged.
All six floating-base velocity DOFs and arm velocities are checked; the closed-grip command remains
in force and release stays unauthorized. These are diagnostic qpos/qvel/ctrl resets
from the first faults in the matched Cartesian experiment, not complete
integration-state replays or successful transfers. The retention failures
remain open. [Compact evidence](../../paper/data/whole_body_control/mars_fault_stop_v1/results.json)
keeps all failures, measurements and artifact/source hashes.

All 1,113 fresh source-bound branch regressions pass at `e6401542`. Because control source
identity changes, old checkpoint execution evidence remains bound to its old
runtime; do not rewrite training provenance to admit it. The teacher reproduces all122 earlier cases exactly. A declared256-transition
source-fidelity smoke audit archives its preselected initialized actor; all56
saved input actions remain exactly equal. Its trained final is not selected.
The subsequent six complete public-tool replays confirm every failed stop: all
three analytic references still fail retention at lift/preplace, and the
initialized learned source fails tracking (56/250/56 calls). All remain0/3
transfers. Stopping is fixed; grasp retention and learned approach tracking
are not. The ongoing context-training comparison
continues in its immutable earlier checkout. No hardware is used or qualified.
