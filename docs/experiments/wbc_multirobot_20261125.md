# Multi-robot WBC MuJoCo backend matrix

Date: 2026-11-26 (started 2026-11-25)

Narrative training report (method, curriculum, gates, and roadmap):
[wbc_training_report_20261126.md](wbc_training_report_20261126.md). This matrix
is the result index.

Scope: direct-MuJoCo WBC plants for Mars, Stretch, Sourccey, RB-Y1, and
PandaOmron. This is simulator-only; none of these is a RoboCasa benchmark
result.

## Robot matrix

| Robot | Profile | Direct MuJoCo backend | Teacher | DAgger status |
|---|---|---|---|---|
| Innate Mars | `mars_sim_profile()` | `MuJoCoWbcBackend` | 1.00 | 0.955 mean (v2 gate) |
| Stretch 3 | `stretch_sim_profile()` | `StretchWbcBackend` | 0.97 | 0.951 held-out (v3) |
| Sourccey | `sourccey_sim_profile()` | `SourcceyWbcBackend` | 1.00 | 0.960 held-out (v3) |
| RB-Y1 | `rby1_sim_profile()` | `Rby1WbcBackend` | 1.00 | 1.000 held-out (v3) |
| PandaOmron | `pandaomron_sim_profile()` | composed Omron+Panda torque plant | 1.00 | 0.942 clean (v2, re-gated) / 0.978 robust (v3) |

Teacher success is a ceiling, not a policy result. Held-out numbers are the
strict 150-episode re-gate (50 episodes per subtask on seeds 30000/40000/50000),
with measured collisions and settled tracking.

| Robot | Arm subtask(s) | Base | Combined | Aggregate |
|---|---|---|---|---|
| Innate Mars | arm 1.00 | 1.00 | 0.92 | **0.973** |
| Stretch 3 | arm 0.94 | 1.00 | 0.92 | **0.951** |
| PandaOmron (clean) | right 1.00 | 1.00 | 0.84–0.92 | **0.942** (re-gated) |
| PandaOmron (robust) | right 1.00 | 1.00 | 0.94–1.00 | **0.978** (v3) |
| RB-Y1 | left 1.00, right 1.00 | 1.00 | left 1.00, right 1.00 | **1.000** |
| Sourccey | left 1.00, right 1.00 | 1.00 | left 0.86–0.88, right 0.90–0.96 | **0.960** |

The clean Panda checkpoint re-gates to 0.942 and fails the strict combined
base/arm gate (0.827 < 0.85); it is not promoted. The robust checkpoint is the
promoted policy at 0.978.

Multi-seed gate (50 episodes per subtask, 150 per robot; aggregate success):

| Robot | seed 30000 | seed 40000 | seed 50000 | Mean |
|---|---|---|---|---|
| Mars | 0.973 | 0.920 | 0.973 | 0.955 |
| Stretch | 0.953 | 0.940 | 0.960 | 0.951 |
| Sourccey | 0.952 | 0.960 | 0.968 | 0.960 |
| RB-Y1 | 1.000 | 1.000 | 1.000 | 1.000 |
| PandaOmron (clean) | 0.947 | 0.973 | 0.907 | 0.942 |
| PandaOmron (robust) | 0.973 | 0.980 | 0.980 | 0.978 |

A combined-weighted Mars retrain (6 s combined, 60% collection weight) reached
0.956 mean (0.967/0.933/0.967), statistically the same as the retained 0.955.
The original Mars 3 s/5 s and 40/20/40 protocol stays.

Strict coverage gates: `configs/wbc/multirobot_cases.json` passes
`free_space_reach` and `base_arm` for Mars, Stretch, Sourccey, and RB-Y1 with
measured collisions and settled tracking. Stretch has one measured
self-collision in 150 combined episodes (6.85 N; `max_collision_rate: 0.01`);
the other rows have zero. The robust Panda checkpoint passes its separate gate;
the clean checkpoint fails the combined base/arm gate (0.827 < 0.85).

Artifacts and the per-task breakdown are in
`logs/eval/wbc_multirobot_20261126/` (`manifest.json`, checkpoints, held-out
JSON) and the strict re-gate under `logs/eval/wbc_multirobot_20261126/strict/`.
The clean Panda row was re-gated from the shipped checkpoint: the old 0.996
row was not reproducible (the checkpoint gives 0.947/0.973/0.907 at the three
seeds). All rows load through the layout-v4 path with v2/v3 checkpoints migrated
(zero contact/gripper-target inputs, v2 gripper-force columns rescaled); the strict
re-gate also records measured collisions and settled tracking error. See the
narrative [training report](wbc_training_report_20261126.md).

## What landed

- `wbc_robots.py`: robot/profile registry and RoboCasa names.
- `wbc_backends.py`: direct-MuJoCo vs RoboCasa factory.
- `wbc_robocasa.py`: explicit registration hooks; PandaOmron now registers the
  native Omron+Panda adapter, while missing adapters still fail closed.
- `wbc_contact.py`: measured external/self contacts for Stretch, Sourccey,
  RB-Y1, and RoboCasa too. Support-floor contacts are excluded; intentional
  hand-object contacts are reported separately, not as collisions.
- `wbc_coverage.py`: no metric defaults; every gate row must carry measured
  collisions, and tracking is the mean over the last 10 steps with final/max
  fields reported.
- `wbc_dagger.py`: checkpoints store the full task contract; evaluators rebuild
  that suite instead of the current defaults. Legacy checkpoints can be
  migrated with `scripts/stamp_wbc_contract.py`.
- `wbc_stretch.py`: wheel/freejoint plant through `ManipulationControl`; five
  contract motion axes plus gripper. Resets to a raised, collision-free ready
  pose (the packaged zero pose folds the wrist into the chassis), resamples
  targets that start against the chassis/table, and re-anchors the combined
  reach goal to the measured base pose after the drive.
- `wbc_sourccey.py`: planar base, lift, two 5-DOF arms, two grippers, optional
  `active_arms` for single-arm subtasks. Resets at the raised, collision-free
  home posture with small joint noise; sampling the full arm range made the
  imitation target much harder.
- `wbc_rby1.py`: Galaxea R1 proxy; freejoint base + three steer/wheel modules,
  4-DOF torso, two 6-DOF arms, two grippers, optional `active_arms`. Also
  resets near the packaged home posture instead of sampling every joint range.
- `wbc_pandaomron.py`: composes the Robosuite Omron base + Panda arm + gripper
  into a direct MuJoCo model. The arm/torso are torque motors, so the plant uses
  `TorqueAdapter`; the base is velocity-controlled. It resets to Robosuite's
  Panda initial posture instead of the composed model's self-colliding zero
  pose, samples combined targets at the base goal, and re-anchors them to the
  measured base after the drive. The visual floor is collision-disabled because
  the Omron base has no wheels and otherwise pins against the floor.
- `wbc_dagger.py`: per-robot default suites. Dual-arm robots train single-arm
  variants (`arm_left`, `arm_right`, `combined_left`, `combined_right`, `base`)
  because a simultaneous 10–12-DOF reach is a much harder imitation target.
  BC defaults to 200 epochs; the first multi-arm runs used 30 and were
  undertrained. `--init` warm-starts from a profile-checked checkpoint,
  `--max-bc-samples` bounds the replay buffer, and `--selection balanced`
  ranks checkpoints by aggregate plus the weakest subtask. Combined tasks get
  60% of the collection budget for Stretch/PandaOmron and 60% for the two
  Sourccey/RB-Y1 combined variants; later checkpoints win ties.

## What fixed the plateau

The earlier Sourccey/RB-Y1 0.33 aggregate was not an adapter bug. Three
curriculum choices were compounding:

1. **Full-range joint resets.** Sourccey and RB-Y1 sampled every arm and torso
   joint uniformly across its limits. The actor had to learn IK from an
   enormous 10–12-DOF pose distribution with a few thousand labels. Resetting
   near the collision-free packaged home pose made single-arm BC go from
   ~0.2 to 1.0.
2. **Too few BC epochs.** 30 epochs left the actor underfit; 200–600 epochs
   lowered the arm action error enough for closed-loop success.
3. **Simultaneous dual-arm targets.** Asking for both arms at once doubled the
   effective action dimension. The default suite now trains one active arm at a
   time; the inactive arm is held and publishes an empty target mask.

The combined subtasks were the weakest gate before the curriculum change; the
combined-heavy collection budget and the longer Stretch/Panda window closed
most of the gap.

## Observed limitations

- **Stretch:** the position servos are underdamped and slow; the arm and
  combined subtasks need a 5 s / 8 s window. The promoted policy reaches 0.927
  arm and 0.927 combined. One real gripper/arm self-collision occurs in 150
  combined episodes (6.85 N); the combined gate allows 1/150.
- **Sourccey:** the promoted policy reaches 0.960 held-out (arms 1.00, base
  1.00, combined_left 0.86–0.88, combined_right 0.90–0.96). The checkpoint
  spread remains widest on combined_left.
- **RB-Y1:** the packaged wheels are visual-only; the backend adds wheel
  collision proxies and a steer/wheel swerve mapping. The direct plant is a
  training proxy, not RoboCasa benchmark parity.
- **PandaOmron:** the robust checkpoint reaches 0.978; the clean checkpoint is
  0.942 and fails the combined base/arm gate. The native RoboCasa reach adapter
  now exists, but it is not task success. The composed model's zero arm posture
  self-collides with the wrist; the ready-pose reset and 300/30 torque gains are
  required at 20 Hz.

## Hard-case data correction (PandaOmron)

The 2026-11-25 manipulation-failure sink contains 20 records. Records 014–018
are `place_failed` attempts whose `attempt` field is the preceding **successful
grasp** (`reached=True`, `failure_mode=None`); they are stale last-attempt
entries, not failure postures. Only record 019 is a genuine reach failure
(`reached=False`, `tracking_failed`, `error_m=0.42`).

The exporter now requires an explicit `reached=False` and canonicalizes robot
aliases (`panda_omron`, `rb_y1`, `mars`). The train split is empty and the one
genuine case is evaluation-only (`split='test'`). #267 is merged and future
records carry `attempt.op`, but a new genuine failure population is still
required; `hard_posture` stays planned until then. `dagger_train --hard-cases`
rejects non-train files and empty train files.

The earlier “5/5 train, 1/1 test” hard-case claim is retracted. The mixed
checkpoint's clean-suite result is reported separately; it is not evidence for
a hard-case-trained policy.

## Next experiments

1. **PandaOmron RoboCasa task-target bridge.** Extract object/receptacle/grasp
   poses, emit pre-grasp/grasp/lift/place EE targets plus base/gripper commands,
   and measure task success. The native reach adapter and teacher are ready; the
   policy is not yet trained on task-like target poses.
2. **Failure-data closure and contact-aware retrain.** Add episode/seed/scene
   correlation, collect new genuine `reached=False` failures, export a real
   hard-posture split, and retrain layout v4 with measured contact channels.
3. **Contact/carry/shelf coverage.** Add occupancy and payload cases; do not
   claim these from the current migrated v2 checkpoints.
4. **Merged-scene adapters** for Stretch/Sourccey/RB-Y1 after the PandaOmron
   task bridge.
5. Consider a single embodiment-conditioned policy once per-robot policies are
   stable.

Artifacts:

- Strict re-gate: `logs/eval/wbc_multirobot_20261126/strict/` (checkpoints,
  per-seed held-out JSON, manifests, coverage reports).
- Mars train: `logs/eval/wbc_dagger_features_20261125/`.
- PandaOmron robust train: `/tmp/opencode/wbc_panda_mixed_heldout/`.
- The earlier dual-arm plateau runs remain under `/tmp/opencode/wbc_sourccey_full/`
  and `/tmp/opencode/wbc_rby1_full/`; they are superseded.
