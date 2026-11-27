# Whole-body torque control: training report

Date: 2026-11-26

Status: simulator-only, direct MuJoCo plus a first RoboCasa reach adapter. The
clean free-space suite is trained for all five robots; RoboCasa task success,
contact/carry, and the merged-scene adapters are not yet done. This report is
the narrative companion to the result tables in
[wbc_multirobot_20261125.md](wbc_multirobot_20261125.md) and the RoboCasa smoke
in [wbc_robocasa_20261126.md](wbc_robocasa_20261126.md).

## 1. Objective

Build one controller stack that generalizes across Mars, Stretch 3, Sourccey,
RB-Y1, and PandaOmron:

- **Normalized torque action.** The actor emits base velocity plus per-arm
  joint torques in `[-1, 1]`. A robot-specific adapter turns that into a
  position-servo lead, current command, or true torque.
- **Sensor-only actor.** The privileged teacher may use simulator targets and
  phase state, but the exported actor and checkpoint receive only contract
  observations: base pose/twist, joint encoders, EE pose, EE target, gripper
  state/target, measured contact, and derived target errors. Layout v4 adds the commanded
gripper target to the v3 contact channels.
- **MuJoCo first.** Every robot is trained and gated against its own direct
  plant. RoboCasa/Robosuite comes second, behind the same `WbcBackend` contract.
- **A benchmark path.** The long-term target is to run these policies as the
  low-level controller inside RoboCasa kitchen tasks, with a higher-level
  skill/TAMP layer supplying grasp, lift, and place targets.

“Good” here means a **held-out gate**, not teacher success: 50 episodes per
subtask on a seed not used for training, plus a family-partitioned coverage
gate. Teacher success is recorded only as a ceiling.

## 2. Contract and training loop

The contract is `EmbodimentProfile` + `WbcObservation` + `WbcAction`
(`wbc_contract.py`). Each profile declares the base, torso, arms, grippers,
joint order, torque limits, and identified servo gains. The observation layout
is versioned; v3 added measured contact channels (presence, flag, EE-frame
wrench). A v2 checkpoint is migrated on load by remapping the first-layer
observation columns and zeroing the newly added contact/gripper-target inputs.

Training has three stages:

1. **Privileged teacher rollouts.** A PD/P teacher reaches sampled joint
   postures with staged base-then-arm motion. It is used only to label states.
2. **BC warm start.** The actor regresses `tanh(mean)` to the teacher action.
   The default is 200 epochs; the original 30-epoch runs were badly undertrained.
3. **DAgger.** The current actor rolls out with small action noise, the teacher
   labels the visited states, the buffer is capped, and the actor is retrained.
   Checkpoints are ranked by aggregate or by a balance-aware score
   (`aggregate + 0.25 * weakest subtask`), with later checkpoints winning ties.

### Curriculum decisions that made the robots learnable

- **Home-posture resets.** Sourccey and RB-Y1 originally sampled every arm and
  torso joint from its full range. That made the 10–12-DOF imitation problem
  enormous. Resetting near a collision-free home pose with small noise took
  single-arm BC from roughly 0.2 to 1.0.
- **Ready poses for Stretch and PandaOmron.** Stretch’s zero pose folds the
  wrist into the chassis; PandaOmron’s zero pose folds it into link 5. Both
  plants reset to a collision-free ready posture, and Stretch/Panda combined
  targets are re-anchored to the measured base after the drive.
- **Single-arm subtasks.** Dual-arm robots train `arm_left`, `arm_right`,
  `combined_left`, `combined_right`, and `base`. The inactive arm holds and
  publishes an empty target mask.
- **Combined-heavy collection.** Combined reaches get 60% of the collection
  budget for Stretch/PandaOmron and 60% across the two combined variants for
  Sourccey/RB-Y1. Stretch/Panda use an 8 s combined window.
- **Warm starts and bounded replay.** `--init` loads a profile-checked
  checkpoint; `--max-bc-samples` caps the growing DAgger buffer.

### Contact/payload curriculum (layout v4)

Layout v4 adds `has_gripper_target` + `gripper_target` per gripper. This is the
commanded open/close/hold setpoint, not a measured object state; it sits beside
the v3 measured wrench/finger-force channels so the actor can tell a commanded
squeeze from an accidental contact.

The PandaOmron contact suite adds two subtasks to the existing arm/base/combined
set:

- `wide_arm`: ±0.55 rad arm targets with torso height sampled over 0.12–0.32 m.
- `carry`: arm + base goals with a 0.1–1.2 kg inertial payload attached to a
  body downstream of the wrist F/T sensor, and a closed-gripper teacher command.

The payload mass and the `gripper_close` flag are teacher-side only. The actor
sees the resulting measured wrist wrench, finger force, joint torque, and the
commanded gripper target. A separate Stretch suite uses a 0.05–0.8 kg payload
on the gripper slider; Stretch observes the load through joint torque/servo
effort and finger force rather than a 6-axis wrist sensor. This is the
two-archetype check: true-torque + F/T (PandaOmron) and position-servo +
joint-torque/finger-force (Stretch).

Pilot result: the Panda carry teacher succeeds **3/3** with 2.1 cm tracking.
The longer BC/DAgger run (4,096 teacher steps, 3 × 2,048 DAgger steps, 300 BC
epochs) reached **1/3** after teacher BC and **0/3** after each DAgger
iteration, but DAgger cut the mean tracking to 6.5–6.8 cm. A 10-episode
re-evaluation gives **2/10** for `policy_iter3` (0.196 m mean tracking).

Contact ablation on the same 10 episodes:

| Checkpoint | Full success | Masked success | Full tracking | Masked tracking |
|---|---:|---:|---:|---:|
| `policy_iter1` | 0.10 | 0.00 | 0.034 m | 0.053 m |
| `policy_iter3` | 0.20 | 0.20 | 0.196 m | 0.311 m |

Masking the measured wrist wrench, contact flag, and finger force measurably
worsens tracking, so the actor is using the load channels rather than only
averaging the payload. Masking joint torque as well drops success to **0/10**
for both checkpoints and worsens tracking further (iter1 **3.4 → 9.4 cm**;
iter3 **19.6 → 33.5 cm**). It is still far below the planned carry gate (0.75);
no contact-aware checkpoint is promoted.

To separate “hard carry” from “learn the load,” the suite now includes
`carry_easy`: a 0.4–0.6 kg payload, held base, ±0.10 rad arm targets, and torso
0.24–0.26 m. The PPO launcher is `scripts/finetune_wbc_contact_ppo.py`; the mask ablation is
`scripts/eval_wbc_contact_ablation.py`.

PPO fine-tune result: 8,192 steps raised carry success from **0.20 to 0.30**
(3/10), but mean tracking moved from **0.196 to 0.344 m** and the settled max
reached **1.20 m**. The reward is still optimizing a mix of position, action,
and base terms rather than the carry success gate. Do not promote this
checkpoint.

**Easy-carry curriculum (first contact-aware pass):** `carry_easy` (0.4–0.6 kg,
held base, ±0.10 rad, torso 0.24–0.26 m) reaches **1.00** after teacher BC and
remains **1.00** through two DAgger iterations on 5 eval episodes. Mean tracking
is **1.05 cm**, settled max **3.5 cm**, and BC loss is 0.616 → 0.0016 through
DAgger. This is the first contact/payload policy to pass a carry task, but it is
the narrow curriculum stage, not the full `carry` gate. The checkpoint is
`/tmp/opencode/wbc_contact_carry_easy/best.pt`. A 10-episode ablation still
passes **1.00/1.00** when wrist wrench, finger force, and joint torque are
masked, but tracking worsens from **1.26 to 1.73 cm**. The narrow payload range
is learnable by averaging; the full `carry` task remains the contact test.

Full-carry ramp from the easy checkpoint: teacher BC **0.20**, DAgger iter1
**0.20**, iter2 **0.40** on 5 episodes. The BC loss is already 0.005, so the
remaining gap is closed-loop tracking, not label fit. This is better than the
0.30 PPO result but still below 0.75. The suite now adds `carry_medium`
(0.2–1.0 kg, held base, ±0.15 rad, torso 0.20–0.30 m) as the next curriculum
stage. The easy checkpoint already evaluates **1.00** on medium, and teacher BC
and two DAgger iterations stay **1.00** on 5 episodes (BC loss 0.00065). The
0.2–1.0 kg load range is therefore learnable with a held base and a narrow
target; the remaining full-carry gap is the base+arm coordination and ±0.35 rad
targets, not the payload alone.

A 10-episode medium ablation still passes **1.00/1.00** when wrist wrench,
finger force, and joint torque are masked; tracking is essentially unchanged
(**1.76 vs 1.79 cm**). The easy/medium stages can be solved by averaging the
payload, so they prove the curriculum is trainable but not that the actor needs
contact. Full `carry` is the contact gate.

Full carry initialized from the medium checkpoint is **0/5** at init, after
teacher BC, and after both DAgger iterations. The medium stage overfits to the
held base and ±0.15 rad targets and does not retain the base+arm skill. The best
full-carry result remains the easy-to-full ramp at **0.40**. The suite now adds
`carry_wide` (0.2–1.0 kg, base+arm, ±0.20 rad, torso 0.18–0.30 m) as the bridge
between medium and full. Trained from the 0.40 full-carry checkpoint, it reaches
**0.80** after one DAgger iteration (5 episodes, **2.03 cm** mean tracking,
**3.56 cm** settled max, 0 collisions). The second iteration also succeeds 0.80
but its settled max is 7.14 cm, so `policy_iter1.pt` is the gate candidate. This
is the first contact-aware policy to pass a base+arm carry task on the 5-episode
screen. The 10-episode re-evaluation is the honest gate: success stays **0.80**,
but mean tracking is **18.2 cm**, settled max **1.04 m**, and there are **2
collisions**. That fails the carry gate (5 cm tracking, zero collisions). The
masked run is also 0.80 with 7.0 cm tracking and 2 collisions, so the 5-episode
screen was optimistic and contact use does not fix the instability. No
checkpoint is promoted.

The next run is a mixed suite (`arm`, `base`, `combined`, `carry_wide`) trained
from the wide checkpoint, to keep the free-space/base skills and reduce
collisions instead of overfitting a carry-only task. A full-carry ramp from the
wide checkpoint is also in flight.

RoboCasa task sweep with the wide contact checkpoint: **0/3**. The canned food
reached the retreat phase with 0.78 m displacement, but did not satisfy the
final placement/retreat checks; mango and dish brush were `grasp_failed`. This is
better than the free-space policy's no-transport 0/3, but it is still not task
success and is not a benchmark claim.

**Full carry from wide (5-episode screen):** init **0.60**, teacher BC **0.40**,
DAgger iter1 **1.00**, iter2 **0.40**. The iter1 checkpoint is
`/tmp/opencode/wbc_contact_carry_full_from_wide/policy_iter1.pt`: 5/5 success,
**2.84 cm** mean tracking, **7.08 cm** settled max, 0 collisions. This is the
first full-carry (±0.35 rad, 0.1–1.2 kg, base+arm) pass on the screen.

**10-episode gate:** full **0.80** success, **2.25 cm** mean tracking, **3.04 cm**
settled max, **0 collisions**, `contact_state_rate=1.0`. Masking wrist wrench,
finger force, and joint torque drops success to **0.50** with 2.46 cm tracking.
On seed 30000 the checkpoint passes the planned carry gate (0.75 success, 5 cm
tracking, zero collisions) and the contact channels are materially used. The
3-seed gate is **0.80 / 0.70 / 0.50** at seeds 30000/40000/50000, mean
**0.667**, min **0.50**, so it **fails** the 0.75 gate. The seed-30000 pass was
optimistic. Candidate: `/tmp/opencode/wbc_contact_carry_full_from_wide/policy_iter1.pt`;
the carry case stays `planned` in `configs/wbc/pandaomron_carry_cases.json`.

RoboCasa task sweep with the full-carry checkpoint: **0/3**, all
`grasp_failed`. The wide checkpoint was better (one canned food reached retreat
with 0.78 m displacement). Direct-plant carry is not yet a RoboCasa task policy;
promotion into the WBC carry gate is separate from the task bridge.

A fresh
paired sweep from seed 31000, after refreshing MuJoCo’s derived payload
constants, gave **10/10** on `carry_easy` (1.15 cm mean tracking) and **0/10**
on `carry` (15.49 cm) with the same checkpoint.

**Mixed-suite stability:** training `arm`, `base`, `combined`, and
`carry_wide` from the wide checkpoint reaches **0.75** aggregate after two
DAgger iterations with **0 collisions** on all tasks. Per task:
`arm` 1.00 (1.85 cm), `base` 1.00 (0.0 cm), `combined` **0.33** (3.88 cm),
`carry_wide` **0.67** (3.94 cm). The mixed suite removes the collisions and
preserves arm/base, but combined and carry_wide are still below their gates;
more iterations are needed before promotion.

A longer continuation (3 iterations, 8,192 teacher steps, 4,096 DAgger steps
per iteration) peaks at **0.83 aggregate** on iteration 2: arm 1.00, base 1.00,
combined **0.67** (2.58 cm, 0 collisions), carry_wide **0.67** (2.11 cm, 0
collisions). Iteration 3 is 0.75. The 10-episode evaluation of iteration 2 is
**0.85 aggregate, zero collisions**: arm 1.00 (1.56 cm), base 1.00,
combined **0.80** (2.04 cm, 0 collisions), carry_wide **0.60** (5.26 cm,
settled max 31 cm, 0 collisions). Combined is one episode short of 0.85 and
carry_wide is below 0.75, so no promotion. A focused `combined,carry_wide`
continuation from iteration 2 is the next run.

## 3. Robot matrix

| Robot | Plant | DoF summary | Base | Training notes |
|---|---|---|---|---|
| Innate Mars | direct MuJoCo, adapter-matched | planar 2 base, 1×5 arm, 1 gripper | planar | original contract-native suite; retained 3 s/5 s protocol |
| Stretch 3 | direct MuJoCo wheel/freejoint plant | planar 2, 1×8 lift/telescope/wrist, 1 gripper | planar | raised ready pose; collision-filtered targets; 5 s/8 s windows |
| Sourccey | direct MuJoCo packaged MJCF | planar 2, lift 1, 2×5 arms, 2 grippers | planar | home reset; single-arm subtasks |
| RB-Y1 | Galaxea R1 proxy, steer/wheel modules | holonomic 3, torso 4, 2×6 arms, 2 grippers | holonomic | home reset; swerve mapping; wheel proxies |
| PandaOmron | composed Robosuite Omron+Panda | holonomic 3, torso 1, 1×7 arm, 1 gripper | holonomic | Robosuite ready pose; 300/30 torque gains at 20 Hz |

All policies are simulator-only and profile-checked. A checkpoint loads only
with the exact profile fingerprint and (after migration) the observation layout.

## 4. Results

Held-out clean suite: 50 episodes per subtask, three fresh seeds. All rows are
from the strict re-gate: v2/v3 checkpoints load into layout v4 by copying matching fields and zeroing
new inputs. Migration resizes gripper-force weights only for v2, collisions are measured, and tracking
is the settled-window mean. The previously reported Panda clean 0.996 row was
not reproducible from the shipped checkpoint; it re-gates to **0.942**.

| Robot | 30000 | 40000 | 50000 | Mean | Weak task |
|---|---|---|---|---|---|
| Mars | 0.973 | 0.920 | 0.973 | **0.955** | combined |
| Stretch | 0.953 | 0.940 | 0.960 | **0.951** | arm |
| Sourccey | 0.952 | 0.960 | 0.968 | **0.960** | combined_left |
| RB-Y1 | 1.000 | 1.000 | 1.000 | **1.000** | combined_right |
| PandaOmron (clean) | 0.947 | 0.973 | 0.907 | **0.942** | combined |
| PandaOmron (robust) | 0.973 | 0.980 | 0.980 | **0.978** | combined |

Strict re-gate provenance (150 episodes per subtask, measured collisions,
settled tracking):

| Robot | Mean success | Collisions | Settled tracking mean | Gate |
|---|---|---|---|---|
| Mars | 0.956 | 0 | 0.008 m | multirobot pass |
| Stretch | 0.951 | 1 | 0.012 m | multirobot pass (combined allows 1/150) |
| Sourccey | 0.960 | 0 | 0.010 m | multirobot pass |
| RB-Y1 | 1.000 | 0 | 0.009 m | multirobot pass |
| PandaOmron (clean) | 0.942 | 0 | 0.014 m | fails combined `base_arm` (0.827 < 0.85) |
| PandaOmron (robust) | 0.978 | 0 | 0.012 m | Panda gate pass |

`configs/wbc/multirobot_cases.json` gates Mars, Stretch, Sourccey, and RB-Y1;
`configs/wbc/pandaomron_cases.json` gates PandaOmron. `free_space_reach` and
`base_arm` pass for the promoted policies; `hard_posture`, `shelf_entry`,
`carry_payload`, and `contact_recovery` are planned/not gated. The clean Panda
checkpoint is not promoted. Gate results are generated with
`scripts/emit_wbc_coverage.py` and consumed by
`python -m emet.rl.manipulation.wbc_coverage`.

## 5. RoboCasa reach adapter

The first RoboCasa adapter (`RoboCasaPandaBackend`) drives the native
PandaOmron `PickPlaceCounterToCabinet` sim by writing contract
torques/velocities directly to the Omron+Panda actuators. It uses a
reset-anchored robot frame, backs the chassis 0.45 m from the counter, and
holds the torso height with a PD/bias servo. An 8 cm reach target tracks to
**1.3 cm / 0.029 rad** in the kitchen scene.

A privileged teacher now labels RoboCasa states: joint-space PD for sampled
postures (0.48 cm / 0.0115 rad on a sampled target) and a Cartesian PD fallback
through a finite-difference EE Jacobian for arbitrary task targets. A loaded
20-episode reach sweep (joint targets ±0.15 rad) is **20/20** with position
error mean **1.13 cm**, p95 **1.67 cm**, max **1.70 cm**; the 1.3 cm smoke is
typical, not a marginal one-off. The sweep also writes an
`observations`/`teacher_actions` dataset for BC/DAgger fine-tuning
(`/tmp/opencode/wbc_robocasa_sweep_policy/loaded_data.npz`). A Cartesian offset
sweep (±5 cm x/y, 0–8 cm up) is also **20/20**, mean **0.87 cm**, p95 **1.54
cm** (`/tmp/opencode/wbc_robocasa_sweep_cartesian/`). Both reach envelopes are
inside the 2 cm tolerance; DAgger is reserved for farther task targets.

This is a reach-tracking result, not task success. The benchmark task still
needs:

1. object/receptacle/grasp targets from a skill or TAMP layer,
2. base goals for counter approach and retreat,
3. carry/contact observations for the real manipulation phases,
4. merged-scene adapters for Stretch, Sourccey, and RB-Y1, whose RoboCasa
   configs use emet’s merged-scene path rather than a native Robosuite robot.

## 6. Review corrections (2026-11-26)

- **Hard cases were invalid.** Records 014–018 in the failure sink are
  `place_failed` attempts that carry the preceding successful grasp as their
  `attempt` (`reached=True`, `failure_mode=None`). Only record 019 is a genuine
  reach failure. The exporter now rejects anything without `reached=False`,
  normalizes robot aliases, and writes explicit train/test splits. The train
  split is empty, 019 is evaluation-only, and the earlier 5/5 train / 1/1 test
  claim is retracted. `dagger_train` rejects non-train and empty hard-case
  files.
- **Coverage gate schema.** The gate now rejects missing, non-scalar, or
  non-finite metrics (NaN/Inf), missing tracking/duration/collisions,
  list-valued episode counts, and invalid contact rates. There are no
  zero-collision or zero-duration defaults. `scripts/emit_wbc_coverage.py`
  converts `eval_wbc_policy.py` and `eval_wbc_hard_cases.py` reports into
  strict per-case rows; tracking is the mean error over the last 10 steps
  (0.5 s) after the first successful step (last 10 steps for a failed episode),
  with final-step and max fields reported separately. Every row must certify
  `collision_measured=True`.
- **Collision measurement.** Stretch, Sourccey, RB-Y1, and RoboCasa now read
  solved MuJoCo contacts instead of hardcoding `collision=False`. Wheel/caster
  floor contacts are support; intentional hand-object contacts are not
  collisions; chassis/arm contact and non-gripper self-collision are. The
  direct Mars and Panda plants already had measured contact paths.
- **Checkpoint compatibility.** Layout v3 added contact channels. `WbcPPO.load`
  migrates v2 checkpoints by remapping the first-layer observation columns and
  zeroing the newly added contact/gripper-target inputs; migrated checkpoints are
  marked in metadata.
- **Seeding and aliases.** `dagger_train` seeds before constructing the actor;
  `default_subtasks` canonicalizes `mars`, `rb_y1`, `panda_omron`, etc.
- **Resume and CLI checks.** Periodic checkpoints carry the task contract and a
  resume without it is refused. `--hold-base` requires base-goal probability 0.
  Odd `total_steps` are consumed instead of silently dropped.
- **Feedforward.** Multi-calibration tables are rejected until nearest-posture
  torque selection is implemented.

Remaining review follow-ups: recorded-scenario embodiment fingerprinting is a
cross-PR gap with #267; hard-case checkpoint selection should use the coverage
report once genuine cases exist; hard-case data must be regenerated after #267.

Landing order and old-benchmark regression gates are in
[robocasa_benchmark_landing_20261126.md](../plans/robocasa_benchmark_landing_20261126.md).

## 7. Reproduce

```bash
# Held-out clean gate (writes per-seed JSON + manifest)
uv run python scripts/eval_wbc_policy.py \
  --robot pandaomron \
  --checkpoint logs/eval/wbc_multirobot_20261126/pandaomron_robust/best.pt \
  --output /tmp/wbc_gate --episodes 50 --seeds 30000,40000,50000

# Convert to strict coverage rows and run the family gate
uv run python scripts/emit_wbc_coverage.py \
  --cases configs/wbc/pandaomron_cases.json \
  --policy-report /tmp/wbc_gate \
  --out /tmp/wbc_coverage.json
uv run python -m emet.rl.manipulation.wbc_coverage \
  --cases configs/wbc/pandaomron_cases.json \
  --results /tmp/wbc_coverage.json \
  --out /tmp/wbc_gate_report.json

# RoboCasa reach smoke
uv run python scripts/smoke_wbc_robocasa_panda.py \
  --checkpoint logs/eval/wbc_multirobot_20261126/pandaomron/best.pt \
  --backoff 0.45 --layout 1 --style 1
```

Artifacts: `logs/eval/wbc_multirobot_20261126/` (checkpoints, held-out JSON,
manifest), plus the per-robot training directories under `/tmp/opencode/`.
Training commands and per-robot hyperparameters are recorded in
[wbc_multirobot_20261125.md](wbc_multirobot_20261125.md).

## 8. What we are reaching toward

1. **RoboCasa task targets.** Use the WBC as the low-level controller; a
   skill/TAMP layer emits EE targets and gripper commands for grasp, lift, and
   place phases. PandaOmron is the native path; Stretch/Sourccey/RB-Y1 need the
   merged-scene adapter.
2. **Contact and carry.** The layout-v4 contact/gripper-target channels are measured plant
   state; shelf entry and payload carry still need occupancy and contact-rich
   training cases.
3. **Family gates.** Promote a policy only when free-space, base+arm,
   hard-posture, shelf, carry, and contact-recovery families pass their
   held-out gates; hard-posture is currently planned.
4. **One controller.** The current artifacts are one policy per robot. A
   single embodiment-conditioned policy is the longer-term target once the
   per-robot gates and RoboCasa adapters are stable.
