# Herman blocked-waypoint rehearsal — 2026-11-22

## Question and setup

Can the current Mars whole-body controller handle a reference that requires
self-collision or environmental penetration, including one received while
already moving? This is a baseline for the proposed recovery expert in the
[WBC design guide](../rl/whole_body_control.md#handle-blocked-goals-and-contact-during-execution).

The MuJoCo audit sends timed hand schedules through the simulator's task-pose
command runtime. It compares the analytic feedback teacher and the shared
`workstation-12000/actor-012000` checkpoint using the same scene, seed
(`1900000`) and references. The two replacement cases begin with a reachable
schedule and replace it at 0.60 s with a hand target inside the chassis or
below the floor. Contacts with the robot are sampled at physics cadence.
Contact onset for this report means the summed robot contact force first
exceeded 0.1 N in MuJoCo. This threshold is a measurement convention.

## Paired results

Both controllers completed the reachable control with zero measured robot
contact. The four blocked schedules, including both replacements, ended in
simulator guard faults. All ten runs confirmed a hold.

| Blocked schedule | Guard outcome | Peak force, feedback / actor (N) | Contact impulse, feedback / actor (N·s) | Contact onset to confirmed hold, feedback / actor (ms) |
| --- | --- | ---: | ---: | ---: |
| Inside chassis | Unsafe contact | 35.94 / 34.98 | 4.31 / 9.49 | 254 / 398 |
| Below floor | Wheel lift | 11.83 / 16.31 | 2.27 / 1.47 | 472 / 316 |
| Retarget inside chassis | Unsafe contact | 39.70 / 42.90 | 6.29 / 12.29 | 320 / 400 |
| Retarget below floor | Wheel lift | 12.30 / 16.58 | 2.25 / 1.69 | 474 / 374 |

The learned actor accumulated nearly twice the contact impulse of feedback
in the chassis retarget. On the below-floor cases it reached a confirmed hold
sooner and accumulated less impulse, with a higher peak force. Record all of
these measures when evaluating recovery. This comparison covers one simulator
start and checkpoint. The guard stopped both controllers. Learned
blocked-goal handling and hardware contact limits still need measurements.

## Relation to the hardware wave

In a separate attended hardware trial, the position bridge moved shoulder
pitch, elbow, wrist pitch and wrist roll by 33.1°, 32.7°, 33.5° and 32.2°,
respectively. Each waypoint confirmed a hold. That trial exercised joint
position commands, with measurements of joint travel and hold receipts.

A later passive capture provided the open pose and scene for a simulator
rehearsal. The wave used a synthetic start with the yaw joint moved 0.04 rad
inward from the captured value. Feedback and both learned checkpoints
completed the base-held wave with peak hand position errors of 1.15–2.34 mm.
In the separate six-case tracking rehearsal, feedback completed 6/6 and the
shared actor 0/6, with confirmed holds in all cases. The cases include hold,
small arm motions, opposite base yaw requests and a coupled retarget.
Hardware WBC and browser waypoint submission remain open. See
[Mars hardware readiness](../rl/MARS_HARDWARE_READINESS.md).

## Reproduce the audit

Run from the WBC worktree with its existing environment and checkpoint. Use
the workstation's assigned simulator CPU if CPU affinity is needed.

```bash
export PYTHONPATH=src CUDA_VISIBLE_DEVICES= MUJOCO_GL=disable
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ACTOR=logs/recovery/expert-progression-v1/overnight-distill-v1/workstation-12000/actor-012000
REPORT=$(mktemp /tmp/herman-wbc-infeasible.XXXXXX.json)
.venv/bin/python scripts/audit_mars_wbc_infeasible.py \
  --actor "$ACTOR" --seed 1900000 --output "$REPORT"
```

The local report is `logs/herman-wave/wbc-infeasible-paired-v2.json`.
Companion traces are under `logs/herman-wave/wbc-open-normal-wave/`,
`logs/herman-wave/wbc-open-tracking/` and
`logs/herman-wave/attended-coordinated-large.UMKAuv/`. These records are
stored locally outside the tracked documentation.
