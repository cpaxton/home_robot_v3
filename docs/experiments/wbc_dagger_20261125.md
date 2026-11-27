# Whole-body DAgger on Mars: arm, base, and combined

Date: 2026-11-25

Robot: **Innate Mars** in MuJoCo (`mars_sim_profile()`): planar differential
base, one 5-DOF arm, one parallel-jaw gripper. The contract is multi-robot, but
this run is Mars-only. No Sourccey, RB-Y1, Stretch, or hardware results are
claimed here.

## Scope

This is a simulator-only, adapter-matched result:

- policy action: normalized per-arm joint torque + planar base velocity +
  gripper command;
- actuator path: `PositionServoAdapter` → bounded position targets → MuJoCo
  position servos (`kp=80`, `kv=4`, effort `2.0`);
- base: planar `MarsControl` `base_x/y/yaw` velocity actuators, not the weak
  physical-wheel contact model;
- observation: contract observation plus derived sensor-only base/EE error
  features (layout version 2);
- teacher: simulator-privileged staged controller (base first, then arm IK),
  used only for labels.

## Subtask suite

| Subtask | Base goal | Arm goal | Horizon |
|---|---|---|---|
| `arm` | held | 0.15 rad joint-space target range | 3 s |
| `base` | forward cone, 5–12 cm | held | 5 s |
| `combined` | forward cone, 5–12 cm | 0.15 rad target range | 5 s |

Base turning behind the robot, contact-rich manipulation, carrying, and
perturbations are not in this suite.

## Training protocol

- initial privileged-teacher rollouts: 4,096 steps
- DAgger iterations: 8 × 2,048 actor steps
- actor noise during DAgger: 0.05 normalized action units
- BC epochs per iteration: 40
- hidden width: 256
- evaluation: 20 episodes per subtask after each iteration; best checkpoint by
  aggregate held-out success
- final validation: 50 fresh episodes per subtask (seeds disjoint from
  training)
- device: local RTX 4070 Laptop (CUDA); Caliban was not reachable with the
  available credentials

Command:

```bash
python -m emet.rl.manipulation.wbc_dagger \
  --iterations 8 --teacher-steps 4096 --dagger-steps 2048 \
  --bc-epochs 40 --batch-size 64 --hidden 256 \
  --eval-episodes 20 --noise-std 0.05 \
  --device auto --output runs/wbc-dagger
```

## Results

Fresh held-out validation, 50 episodes per subtask:

| Subtask | Learned actor | Privileged teacher |
|---|---:|---:|
| Arm-only | 1.00 | 1.00 |
| Base-only | 1.00 | 1.00 |
| Combined base+arm | 0.92 | 0.94 |
| **Aggregate** | **0.973** | **0.98** |

DAgger progression (20 episodes per subtask):

| Stage | Arm | Base | Combined | Aggregate |
|---|---:|---:|---:|---:|
| Privileged teacher | 1.00 | 1.00 | 0.75 | 0.92 |
| Plain teacher BC | 0.35 | 1.00 | 0.00 | 0.45 |
| DAgger iter 1 | 1.00 | 1.00 | 0.75 | 0.92 |
| DAgger iter 3 | 1.00 | 1.00 | 0.80 | 0.93 |
| DAgger iter 5 | 1.00 | 1.00 | 0.85 | 0.95 |
| DAgger iter 8 | 1.00 | 1.00 | 0.75 | 0.92 |

The best checkpoint is iteration 5/6; later iterations drift and are rejected
by best-checkpoint selection.

## What mattered

1. **DAgger**, not plain BC. BC improved from 0.45 to 0.92 in the first
   actor-visited-label iteration.
2. **Derived error features.** Explicit base x/y/heading error and EE
   position/orientation error took the combined task from ~0.65 to ~0.85 in the
   iteration sweep. Combined failures were almost entirely arm orientation
   (quat error 0.045–0.157 against a 0.05 rad tolerance).
3. **Staged teacher.** Base first, arm hold, then arm IK against the settled
   base. This fixed the 0/20 plain-BC mixed result.
4. **Forward-cone base curriculum.** Avoids a differential base needing a
   near-π turn inside the horizon.
5. **Executor slew limits** for base, joint targets, and gripper; unchecked
   command jumps were being misreported as invalid physics.

## Artifacts

`logs/eval/wbc_dagger_features_20261125/`

- `best.pt` — selected checkpoint
- `final.pt`
- `policy_iter5.pt`, `policy_iter6.pt`
- `dagger_report.json`
- `dagger_metrics.jsonl`
- `heldout_50.json`

## Limitations

- Simulator-only, Mars-only; no hardware qualification.
- Base goals are forward-hemisphere; full turning and lateral targets are not
  yet covered.
- No carry/contact task beyond free-space reach.
- The teacher is privileged (sampled joint target, staged phase, exact target
  base); it is not part of the actor input or exported checkpoint.
- The policy is not promoted; the old velocity tracker remains the baseline for
  broader evaluation.
