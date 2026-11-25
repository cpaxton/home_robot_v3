# Failure-to-control loop for whole-body control

Goal: a single pipeline that lets **any embodiment** improve from its own
recorded failures, including compliant behavior learned with the torque-based
WBC contract. This is not gain tuning: each stage is a measured, replayable
model with a held-out check.

## Stack

- `feat/wbc-failure-pipeline` (this layer) ← `feat/torque-contract` (#259) ←
  `feat/graded-conservatism` (#257) ← `main`

The torque contract defines the per-robot embodiment (joints, limits, control
modes, effective stiffness) and the policy action. This layer defines how
execution failures become data and how that data changes control.

## Loop

### 0. Capture

The execution sink writes one JSONL record per failed attempt:

- `context`: `op`, object/receptacle, failure mode, **robot id**, **job id**
- `attempt`: EE target, planned joint waypoints, observed EE/joint trace,
  final error and failure category

Sinks: `~/runs/emet/manip_failures/manip_failures.jsonl` (attempts) and
`~/runs/emet/case_library/cases.jsonl` (reconstructible scenarios).

### 1. Embodiment calibration (implemented here)

Steady-state tracking error under load is a measurable property of the
actuator/gravity model. For every tracking failure:

```text
bias[joint] = final_planned_q[joint] - median(last N observed q[joint])
```

A robust median across attempts gives a static feedforward in joint units.
`fit_wbc_feedforward.py` fails closed when a joint lacks enough valid
attempts, rejects implausible outliers, and writes:

```json
{"schema": 1, "embodiment": "rby1", "joint_bias": {"torso_joint2": -0.52, ...}}
```

Apply it to the kinematic executor with
`EMET_WBC_FEEDFORWARD=~/runs/emet/wbc_feedforward.json`; the final posture is
commanded as `final_q + bias`, clipped to the MuJoCo joint limits. For the
torque path, `FeedforwardModel.torque_mapping(stiffness)` converts the same
bias to `PositionServoAdapter.gravity_torques` (never inferred implicitly).

Commands:

```bash
uv run python scripts/fit_wbc_feedforward.py --embodiment rby1 --dry-run
uv run python scripts/fit_wbc_feedforward.py --embodiment rby1
EMET_WBC_FEEDFORWARD=~/runs/emet/wbc_feedforward.json \
  uv run python scripts/scripted_sim_pick_place.py ... --tool-calls-json ...
```

### 2. Residual / compliance policy (next)

The calibrated controller is the baseline. The failure dataset then supplies
hard cases to a torque policy:

- observation: WBC contract observation (base twist, joint positions/velocities,
  torso/arm groups, contacts when available)
- action: torques from the policy; `PositionServoAdapter` maps them to bounded
  position leads/current limits for position-servo robots, or a true torque
  adapter drives torque/current robots
- objective: tracking plus contact/force terms so the policy can yield instead
  of fighting the environment (compliant behavior)

The same scenario records are replayable: start state, initial map, goal and
embodiment are sufficient to rebuild the case per robot.

### 3. Evaluation and promotion

Every controller change is validated on **held-out** records for that robot:

- tracking: EE error, torso/joint steady-state error, arrival time
- contact/safety: penetration, force/current saturation, refusal behavior
- coverage: which failure modes and object/receptacle categories improved

Promote per embodiment only. A calibration or policy that helps rby1 is not
evidence for Stretch, Sourccey, or Mars unless their held-out records improve.

## Current evidence

The rby1 RoboCasa counter task (`robocasa_rby1_counter_to_cab_mcts`) reached
execution and failed `pregrasp_tracking_failed` with torso2/3 droop (~0.4 rad)
while torso1/4 tracked. Fitting the existing three records yields:

```text
torso_joint2  -0.5216
torso_joint3  +0.3737
arm joints    ~0.00
```

That is the system-ID answer to the failure. The next validation run applies
this model and compares the EE error against the uncalibrated 0.52 m baseline.

## Non-goals

- No implicit gain tuning or hard-coded per-task offsets.
- No cross-embodiment promotion from a single robot's records.
- No claim of physical compliance from kinematic/teleport data; those records
  provide geometry and tracking cases, not contact compliance.
