# Contact diagnosis and grasp reference selection, October 17

The six-episode agent baseline remains **2/6**. These subsequent scripted,
privileged CPU diagnostics investigate execution; they do not rescore that
baseline or test the LLM, perception, navigation, or hardware.

## Question

Both left-start agent trials failed payload retention during lift. Before
assigning the problem to learned low-level control, check whether scene physics
or the reference grasp is responsible.

All probes use the archived baseline XMLs at
`/home/cpaxton/runs/emet/tamp-agent-baseline-20261017/fixtures/`. The robot model,
contact settings, actuator gains, and retention thresholds (20 mm translation,
0.1 rad rotation) are unchanged. No object attachment or live object pose writes
are introduced. These diagnostic probes did not enable the independent actuation
audit and are not physical acceptance certificates.

## Findings

- An in-process actuator replay completed the familiar and left cylinder
  transfers. Both selected a 45-degree wrist approach after rejecting earlier
  shallow alternatives. This establishes a feasible contact trajectory for the
  left scene in that harness, not equivalence to live execution.
- A live ZMQ replay with the original shallow-first ordering completed lift but
  failed during placement: translation residual **0.005545 m**, rotation
  **0.100025 rad**. This is a changed payload orientation, not evidence of a
  dropped object. It still invalidates the rigid payload collision certificate.
- A live probe preferring the steeper alternatives completed grasp, lift,
  placement, release, and retreat. It included two simulator seconds of startup
  settling. A prior steeper probe without that settling was rejected before
  execution with `scene_changed_replan`; it is not a contact failure.

These outcomes do **not** establish a solver/friction defect. The live server
also ramps gripper commands, whereas the in-process facade sets their actuator
target directly. That timing difference is a hypothesis, not an isolated cause.

The matched left-start controls both waited two simulator seconds after startup:

| Live reference ordering | Completed transfers | Failure |
| --- | --- | --- |
| Shallow first | 0/1 | Placement rotation 0.100064 rad; translation 0.005747 m |
| Steeper first | 2/2 | None |

These are cylinder transfers, not two-object agent episodes. The failure phase
in these scripted probes differs from the earlier agent baseline's lift failures;
therefore the original baseline failure has not been reproduced identically.

## Change under test

Prefer the existing 45-degree grasp family before the 15-degree family, keeping
all original families and height/yaw alternatives. The same ordering applies to
all objects. IK, collision checks, measured retention, lift, bilateral hold, and
released support verification continue to gate success. This changes reference
selection, not simulated contact physics.

## Evidence and reproduction

Archived probe sources and traces are under
`/home/cpaxton/runs/emet/tamp-grasp-reference-20261017/`.
A compact [machine-readable result](tamp_grasp_reference_20261017.json) records
outcomes and the production patch hash. Original temporary paths:

- `/tmp/probe-tamp-lift.py` and `/tmp/tamp-lift-physics/{familiar,heldout_left}.json`
- `/tmp/probe-tamp-live-lift.py` and `/tmp/tamp-live-lift-physics/result.json`
- `/tmp/probe-tamp-live-lift-pitch-settled.py` and
  `/tmp/tamp-live-lift-pitch-settled/result.json`
- `/tmp/tamp-left-confirm.py`, `/tmp/tamp-left-shallow-control.py`, and
  `/tmp/tamp-familiar-confirm.py` for confirmation and matched controls.

Launches use `emet jobs run --no-gpu-exclusive --cpu-safe`, software rendering,
no cameras, and no local GPU or VLM. Each live facade cleans up its client and
server in `finally`.

## Regression checks

The native adapter, physical motion contract, and contact tracking suites pass:
**117 tests, zero failures or skips**. The initial run exposed one calibration
test that required exactly the old center-height grasp. It now checks horizontal
pad centering and that the pinch remains within the object's vertical bounds,
allowing the already-supported elevated grasp candidates. Ruff and whitespace
checks pass. The separate in-process cube and cylinder actuator regressions
also pass **2/2** (single transfer per object; no LLM or live IPC).
The familiar-start cylinder transfer also passes through the live ZMQ interface.

## Implication for RL references

A collision-free geometric plan is a candidate reference. Qualify it by replay
through the live actuator interface, with measured retention and task completion,
before treating it as a successful demonstration. Keep failed traces and their
failure phase/residuals as separate diagnostic data. Simulator object poses used
for these checks are privileged labels, not proposed hardware sensors.

Broader initial placements, both payloads, repeated contact trials, and the full
agent baseline remain necessary before claiming general grasp robustness. The
right-start planning failures are a separate unresolved issue.

## Subsequent full-agent result

The [six-episode rerun](tamp_agent_baseline_grasp_v2_20261017.md) remains **2/6**
full tasks, with **5/12** satisfied goals versus 4/12 previously. One left
cylinder transfer completes, but pregrasp revalidation and subsequent planning
failures prevent additional full tasks. The scripted success above does not
establish robust agent-level execution.
