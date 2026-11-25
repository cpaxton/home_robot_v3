# External innate-os MuJoCo torque-adapter validation

Status: **smoke test passes against an external innate-os checkout.** This is a
validation harness, not a vendored simulator. The innate-os checkout and its
assets live outside this repository.

The purpose is to answer one question before touching the robot: can a torque
policy be realized through the proposed position-servo adapter on the actual
Mars plant model, without losing the controller's stability?

## Setup

The harness expects an innate-os checkout and its sim assets:

```bash
git clone https://github.com/innate-inc/innate-os.git /tmp/opencode/innate-os
cd /tmp/opencode/innate-os
./innate-sim assets
cd sim
uv sync --group notebook
```

It does **not** need the innate Docker runtime. It imports
`mars_sim_driver.core.VirtualMars` from the external checkout, which is the
same MuJoCo plant, URDF, contact model, and PD servo constants the Docker sim
uses. The Docker stack remains the place for ROS-level integration; this
harness is for physics/adapter checks.

Run:

```bash
cd /path/to/home_robot_v3
INNATE_OS_ROOT=/tmp/opencode/innate-os ./scripts/run_innate_torque_validation.sh
```

The report is written to `logs/innate_torque_validation/report.json`.

## What the harness does

The script is `scripts/validate_innate_torque_adapter.py`. It:

1. Builds the Mars `EmbodimentProfile` (5 arm joints + separate gripper) and a
   sim-matched actuator profile (`kp_eff=50`, `kd_eff=1`, `max_lead=0.2`,
   `torque_limit=50`) from the sim's own PD constants.
2. Applies a **0.5 N*m wrist torque** for one second through:
   - the Emet `PositionServoAdapter` → `VirtualMars.set_joint_target`, and
   - a direct-torque `VirtualMars` subclass that writes `qfrc_applied`.
   It compares final position, velocity, and applied torque.
3. Repeats the command with a hardware-like profile (`kp_eff=1000`,
   `kd_eff=1`, `max_lead=0.05`, `torque_limit=2.0`) to check torque and
   position-lead clipping.
4. Holds the home pose with a gravity feedforward and checks joint drift.
5. Drops a cube at the gripper, closes until contact, applies a wrist torque,
   and verifies that cube–pad contact remains measurable from
   `data.contact`/`mj_contactForce`.
6. Flattens and rebuilds a `WbcAction` to verify the Mars profile has one arm,
   five arm joints, one gripper, and an 8-value action vector.

## Result (2026-11-25)

| Check | Result |
|---|---|
| Adapter vs direct torque, final joint4 | `0.54398` vs `0.54864` rad; delta `0.0047` rad |
| Adapter vs direct torque, final joint4 velocity | `0.09493` vs `0.09715` rad/s; delta `0.0022` rad/s |
| Applied torque | `0.4606` vs `0.5` N*m (sim sag/backlash/PD residual) |
| Hardware-like torque clip | commanded `2.0` N*m at a `2.0` limit |
| Position lead at clip | `0.002` rad, within `max_lead=0.05` |
| Gravity-feedforward home drift | `0.0015` rad max |
| Contact after wrist torque | cube–`robot_link61_pad4` force `3.16` N, contact detected |
| Gripper position under contact | `-0.0857` rad (target `-0.6`; cube stops the jaw) |
| Mars action layout | 1 arm × 5 joints + 1 gripper = 8 values |
| Failures | none |

The adapter/direct difference is expected: the sim's PD uses a velocity term,
structural sag, and joint backlash, while the adapter is model-based
feedforward. The `kd_eff * qd` term brings the two within a few milliradians;
the remaining gap is the plant-model mismatch that a real bench identification
must reduce.

## What this establishes

- The **torque → position-target** mapping is stable on the Mars plant model
  and reproduces direct torque to within a few milliradians for a wrist joint.
- The **limits** (torque, position lead) behave as intended.
- The robot-side adapter has a credible path: it can run at the 200 Hz control
  loop, with `kp_eff`, `kd_eff`, gravity, and current/PWM limits identified per
  joint.
- The sim exposes enough contact information to evaluate contact-rich behavior;
  the test can measure a cube–pad contact force while a torque command is
  active.
- The Emet multi-arm schema maps cleanly to Mars: one 5-DOF arm plus a separate
  gripper.

## What it does not establish

- It is **not** full-arm torque control. The x430 shoulder/elbow joints still
  require position-offset impedance; the test uses the sim's PD plant, not a
  current loop.
- It does **not** validate hardware `kp_eff`, `kd_eff`, gravity, friction, or
  thermal limits. Those are bench-identification outputs.
- It is **not** a grasp/lift/carry result. The cube is on the floor and the
  gripper is stopped by contact; this is a contact-telemetry smoke test.
- It is **not** a dual-arm or multi-robot sim. The multi-arm schema is covered
  by the contract tests; the innate sim is Mars-only.
- The external checkout is not part of CI. CI would need a pinned innate-os
  image/checkout and MuJoCo/EGL; that is a separate integration decision.

## Relation to the interface plan

The robot-side adapter and typed command/state changes are specified in
[`innate-os-wbc-interface.md`](innate-os-wbc-interface.md). The sim harness is
the first half of that plan: before adding `ArmCommand`/`ArmState` to
innate-os, the same torque-to-position mapping is exercised against the plant
model. The next step is to put that adapter in `mars_arm` and expose the
resulting applied torque/current and contact state to Emet.

## Files

| Path | Purpose |
|---|---|
| `scripts/validate_innate_torque_adapter.py` | External MuJoCo smoke test |
| `scripts/run_innate_torque_validation.sh` | Checkout/assets/venv wrapper |
| `docs/plans/innate-os-wbc-interface.md` | Innate OS interface changes |
| `docs/plans/whole-body-rl-control.md` | Emet multi-arm torque contract |
