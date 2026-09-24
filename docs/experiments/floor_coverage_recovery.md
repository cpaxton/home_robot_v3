# Floor-coverage recovery pilot

The September 23 small-room controls both stopped before grasping. Molmo had
28 reachable cells, all outside the 0.709–0.85 m grasp workspace. RoboCasa had
five candidates in range, all rejected by the explored-footprint check, not
by observed obstacle overlap. These are Stretch-in-MuJoCo controls, not RBY1.
Artifacts: `~/runs/emet/grasp-approach-audit-20260923/`.

## Contract

The tool result distinguishes `insufficient_floor_coverage`,
`workspace_obstructed`, and `no_reachable_workspace`. The last code does not
prove either missing coverage or a physical obstacle. Results include radial
bounds, reachable-cell counts, and rejection counts; no simulator truth enters
the agent. The planner retains its existing footprint and clearance checks.

For a confirmed pre-grasp rejection with no pickup executed and a successful
post-action observation, the model may request `observe_floor`. The failed
batch is discarded. Only that observation is executable in the next round;
another action requires a subsequent round after the observation succeeds.
The existing three-round budget is unchanged. Uncertain payloads, stale frames,
unconfirmed head motion, and failed mapping do not authorize a retry.

The observation moves only the head, requires measured head pose and a newer
received frame after motion, then updates the map from calibrated RGB-D.
Frame receipt sequence is not an acquisition timestamp guarantee; stronger
bridge timestamp validation remains useful. No clearance is inferred from a
successful observation. Adapters lacking the required capabilities refuse it.
Lidar may provide obstacle evidence, but free scan rays are not floor support
or proof against drop-offs.

## Acceptance

Run the frozen small-room Molmo and RoboCasa controls serially, with the same
Qwen preset, seeds, budgets and independent physical scorer. Inspect whether
the model requests the observation, whether the map gains usable coverage,
whether replanning succeeds, and whether physical pick/place passes. Do not
call the change an OVMM improvement solely because the head moved.

EQA uses its existing separate episode tool pack; this pilot changes CHAT
manipulation recovery, not EQA policy. The low-level diagnostics are shared.
No robot-specific policy branches or weakened safety thresholds are added.

## September 24 results

Three sequential managed jobs, unchanged task prompts and physical scorers:

- `20260924_083453_7df611`, source `4cb00bf0`, Molmo + RoboCasa:
  both physical pick/place false. Molmo exposed a contradictory system prompt
  forbidding all recovery; RoboCasa happened to obtain a grasp workspace without
  the new observation, then failed fresh manipulation target verification.
- `20260924_084308_7c9cad`, source `9fc4c927`, Molmo: Qwen selected
  `observe_floor`, which succeeded, but the next turn returned object labels.
  Chat and perception share a VLM client; caption inference had reset the agent
  conversation. The system prompt repair alone was therefore insufficient.
- `20260924_084920_24fb06`, source `213b239f`, Molmo: conversation isolation
  across tool execution enabled the complete model-directed sequence: failed
  pickup, floor observation, and pickup retry. Reachable cells increased 28→37;
  nearest reachable target distance improved 1.027→0.927 m, still outside the
  unchanged 0.85 m maximum. The retry stopped safely; physical pick/place false.

Artifacts respectively live under `~/runs/emet/floor-recovery-20260924/`,
`~/runs/emet/floor-recovery-prompt-20260924/`, and
`~/runs/emet/floor-recovery-context-20260924/`. Inspect `process.log`,
`physical_result.json`, and `evidence/navigation/failed_approach_*.npz` inside
each room's `hybrid_learned_pick_place` directory. Chat transcripts are also
under the frozen worktree's `logs/chat/`. All agent processes exited zero;
managed jobs failed on the independent physical scorer, not task success.

203 focused tests pass on the final code, including navigation, view quality,
tool contracts, grasp handoff, payload guards, and conversation restoration on
both successful and exceptional perception calls. No full EQA run was made.

Next investigate why one floor view leaves the remaining approach unobserved:
inspect actual depth/map support and footprint coverage before adding retries
or enlarging budgets. Do not increase grasp reach or weaken unknown-space checks.
This establishes usable model-directed recovery, not completed OVMM acceptance.

## Follow-up: coverage versus clearance

The follow-up **corrects the pure missing-floor hypothesis above**. Along the
straight approach, the next cell at (-0.8, -0.3) is already observed but has
0.156 m obstacle clearance, below the configured 0.22 m requirement. Sensor
depth places the counter edge near x=-0.43 m; the 0.1 m grid pads obstacles by
two cells before A* applies clearance. These safeguards combine to constrain
the approach. This is not sufficient evidence to weaken either safeguard.

![Observed floor expands the reachable patch, but padded obstacles and clearance still separate the robot from grasp range.](figures/floor-approach-boundary-20260924.svg)

The figure replays the saved `floor-second-observation-20260924` planner maps
(0.1 m cells, grid origin 512,512). It uses no simulator geometry. Red is the
**padded map**, not the physical counter outline. Eight observed, center-clear
cells fall within the grasp-distance annulus beside the counter, but are
disconnected from the reachable component. Full footprint/path validity at
those cells has not been established. A lateral observation/route is a useful
next hypothesis; straight-ahead observations cannot remove an observed obstacle.

### Controlled second observation

The default three-round budget was not increased. The pilot driver accepts an
optional recorded `SIM_FOLLOWUP_COMMAND` for a second user instruction. Such a
run is **assisted diagnostic evidence**, not an autonomous policy score.

- `20260924_101650_2846e7` (`a35da35e`): conditional follow-up was not executed;
  the model emitted malformed farewell JSON. Do not count it as a second view.
- `20260924_102207_03e568` (`a35da35e`): direct follow-up attempted
  `observe_floor` with a missing JSON brace. The parser salvaged its inner
  arguments dictionary and silently treated it as a completed turn.
- `d5398a75` repairs that failure: malformed tool envelopes return
  `invalid_tool_call_json`; no action is dispatched, and correction consumes an
  existing tool round. No string-based action repair or extra budget is added.
- `20260924_103619_b798fd` (`d5398a75`): the model corrected its malformed reply,
  executed the second floor observation and retried. Reachable cells stayed
  **37→37**, nearest reachable target distance stayed **0.927→0.927 m**, and the
  measured base stayed approximately **(-0.8982, -0.2989)**. Observed-cell count
  changed 230→222; global observed counts are not a navigation-success metric.
  Physical pick/place remained false, and execution stopped at the budget.

Each output directory is under `~/runs/emet/`, named after its job (without the
timestamp ID). Floor captures now retain RGB, depth, calibration, head/base pose,
and before/after maps in `evidence/navigation/floor_observation_*.{png,npz}`.
Failed approaches also retain the clearance field and required clearance.

### RoboCasa and EQA rechecks

`20260924_101836_068a73`, source `a35da35e`, uses the unchanged RoboCasa task
prompt. Five in-range footprint checks initially fail on unknown floor. Qwen
requests a floor view, adds 51 observed cells, retries and reaches manipulation.
Thus coverage recovery works in this episode, but **physical pick/place is
still false**. The independent trace records 37 gripper-contact samples and a
minimum gripper-center/can-center separation of 0.00215 m. The can shifts about
10 cm and topples rather than lifting; the gripper subsequently rises about
13 cm without it. This is a grasp-retention failure, not navigation timeout.

Manual inspection also finds a distinct verifier error: the post-lift crop
accepts the nearby metal paper-towel holder as `can`. Initial grounding tracked
the actual red can. The near-gripper check rejects the final result, correctly
preventing placement, but same-object identity needs to survive manipulation.
Exact image: `floor-followup-robocasa-20260924/robocasa/hybrid_learned_pick_place/`
`evidence/grasp_lift_verification/grounding-40195db7d3cf4c79b0d8a01292ecd3db.png`.

EQA job `20260924_101816_1a1131` compares `91871b77` with `a35da35e`, seed 0,
q12/q16: **both 1/2**, identical answers (q12 B/wrong, q16 C/right), 14 planning
steps each, all four processes exit zero. Final-source recheck
`20260924_103655_e82504` on `d5398a75` reproduces those answers and step counts.
This is a small paired non-regression signal, not a broad EQA acceptance claim.
All heavy jobs ran serially under the exclusive GPU lock with CPU-safe limits.

### Review and remaining gates

- Standalone conversation isolation is [PR #176](https://github.com/cpaxton/home_robot_v3/pull/176),
  branch `fix/shared-vlm-conversation`, commit `138ac920`, based on main. Only
  dialogue isolation and its tests are included: no experimental recovery,
  navigation, benchmark or threshold changes. 35 targeted tests and hooks pass;
  no remote CI checks were reported at the time of testing. Main is unchanged.
- The experiment candidate passes 207 focused tests. Evidence capture and
  malformed-response handling remain on `experiment/eqa-inspection-progress`.
- Do not increase straight-ahead recovery retries: the second view plateaued.
  Test floor coverage along a lateral approach, retaining full footprint/path
  checks; audit padding/clearance semantics before any safety-margin change.
- Diagnose RoboCasa grasp retention from the contact/closure/lift trace, and
  preserve target identity in post-action verification. Neither small-room
  physical manipulation gate has passed.

### Lateral observation and closure diagnosis (September 24 follow-up)

`17ce63f1` exposes optional absolute `pan_rad` on `observe_floor`, bounded to
[-1, 1] radians. Omitting it preserves measured pan. Invalid values stop before
motion; measured head pose, fresh-frame and map-update checks remain unchanged.
No base motion or clearance relaxation is added. 227 focused tests pass.

Molmo job `20260924_111809_f49e11` (`floor-lateral-molmo-20260924`) completed
in 327 seconds. An explicit diagnostic follow-up requested pan -0.8, then
another pick/place attempt; this is not an autonomous policy score. The floor
capture measured pan -0.7994 / tilt -0.9992 and added 65 observed cells. The
planner found a lateral route toward (-0.8, -0.6), with minimum clearance
0.2383 m against the unchanged 0.22 m requirement, and the robot moved.
However, final grasp sampling still failed: 38 reachable cells, nearest target
0.9763 m versus 0.85 m maximum. Independent physical pick/place is false/false.
Do not equate more observed cells or a new route with successful approach.

The prior RoboCasa trace identifies excessive closure as a concrete hypothesis:
before lift, both tips apply roughly 19 N with 5–7 mm simulated penetration;
the can moves ~2 cm while lift stays at ~0.8684 m. The lift command does **not**
overwrite the gripper target: full joints are converted to the six manipulation
joints. Do not attribute this failure to a reopening command.

A private sampled-control checkpoint diagnostic starts at sim time 73.01 s
from the saved qpos/qvel/act/ctrl/warm-start state in the unchanged frozen scene.
Linearly replayed controls reproduce failure (maximum rise 0.0103 m; final
object/gripper separation 0.1858 m). Clamping only gripper closure to the existing
`GRIPPER_CLOSED_LOOSE` preset gives 0.1263 m rise and 0.00635 m final separation.
This is approximate sampled replay with evaluator state, **not learned-agent
evidence**, and supplies no GT to the live agent. Diagnostic script:
`/tmp/replay_closure_20260924.py`; source trace is the RoboCasa run above.

`query_geometry_loose_pilot.yaml` selects that existing preset through
`mapping.grasp.loose`, inheriting all tracked/narrow settings. Explicit operation
arguments override config; absent config preserves the old false default.
Non-boolean values are rejected. This is a closure-only ablation, not force
control, a physics change, or a proposed hardware default. Full learned RoboCasa
validation and same-object post-lift identity remain required.

Full learned RoboCasa job `20260924_112629_939c18`
(`closure-loose-robocasa-20260924`, source `1f25ddae`) completed in 308 seconds:
**independent physical pickup true, placement false**. The original task prompt
and scene are unchanged; no assisted follow-up was supplied. Qwen selected floor
recovery, retried, and the log confirms closure to 0.0 rather than -0.3. Both
lift/carry visual checks passed. At the last private trace sample (124.826 s),
the actual can remained in gripper contact, 0.02179 m from its center, after
navigation to the placement area. This supports retention on one live episode;
it does not establish generality across objects or real robots.

Placement stops on `target absent or ambiguous`, not a dropped object. Saved
grounding records show inconsistent localization of `countertop_right_of_stove`:
one abstention cites a missing stove in the current coffee-machine view.
Inspect relational destination context and final support verification next;
do not bypass abstention or release onto an unverified surface. The separate
same-object verifier weakness exposed by the earlier failed pickup remains open.
Final focused suite: **240 passed**, two existing SWIG warnings. No new EQA
episode was run for these CHAT/grasp-only changes; prior EQA numbers above are
not a new acceptance result.

The final Molmo lateral map also shows observed neighboring cells failing
clearance (0.067–0.176 m). Audit map semantics before further scans: the
`pad_obstacles: 2` dilation operates on 2D grid cells (0.1 m in this pilot),
then AStar computes its 0.22 m clearance on that padded map; oriented footprint
validation also consumes the padded obstacles. The YAML comment's `voxel_size`
wording is misleading. This is evidence of overlapping conservative margins,
not permission to remove them without geometry/contact acceptance tests.

### Repeat and carry-loss detection

Frozen `b40bf9b3` repeat `20260924_120339_d33727` completes in 578 s:
physical pickup true, placement false. Unlike the first loose run, the actual
can is lost during base travel: last contact at sim 109.730 s, falling by
109.832 s and on the floor by 110.240 s. Destination search continues until
`target absent or ambiguous`. The final object/gripper separation is 1.679 m.
Thus loose closure has two pickup successes on the same fixture, but **does not
establish reliable retention** and has zero complete placements.

The same-source tight control `20260924_120343_f5b315` completes in 65 s and
fails fresh can grounding before any closure. It is an end-to-end failure, but
not an isolated control for closure efficacy. Do not pool it as another
observed tight-grasp ejection.

The repeat's actuator trace keeps the gripper target at 0.00410 simulated slide
meters throughout travel. Relative object displacement gradually grows from
about 4 mm to 23 mm before loss; contact forces fall substantially. Wrist/lift
references are unchanged across the sampled loss interval. This is not an
explicit reopening command. Do not claim a torque/friction fix from these data.

New checks reuse fresh semantic RGB-D, gripper proximity and relative-position
verification before navigation and after completed trajectory chunks, including
the direct placement-workspace route. The head looks at the end effector and
restores its prior measured pose; no arm/base motion or GT is used by the check.
Confirmed grounded pickup installs the visual reference; confirmed release
clears it. Failed verification latches uncertainty, aborts the placement batch
with `payload_unverified`, and preserves the potentially-held-object interlock.
Missing visibility is **not** reported as a confirmed drop. These are stationary
boundary checks, not continuous sensing and not a repair for unstable grasping.
264 focused tests pass; live detection acceptance remains pending.

### Controlled carry battery

The live detection pilot `20260924_143622_768dff` stopped at initial can
grounding (68 s, physical false/false). It did not reach grasp or exercise the
new carry checks. This is not evidence that drop detection works in an episode.

Use `scripts/diagnose_carry_checkpoint.py` to separate that upstream variability
from grasp physics. It restores the same sampled MuJoCo state and compares:
stationary hold, straight travel, in-place turn, sharper braking, and interpolated
recorded controls. Non-recorded cases keep all non-wheel controls fixed. Motion
ramps up over 2 s, cruises for 4 s, then brakes over 2 s (0.25 s for the braking
ablation); the remaining duration tests retention. Wheel actuator gearing is
included, and references exceeding actuator limits are rejected.

This is **privileged diagnostic evidence**, not a learned-agent score or exact
replay. It uses existing physical-trace checkpoints, does not attach/teleport
objects, and leaves contact physics unchanged. Defaults explicitly describe the
Stretch fixture (wheel radius 0.0508 m, separation 0.3153 m), not a universal
robot model. Supply appropriate geometry/actuator names for other fixtures.

Example (from repo root, single-threaded CPU; run cases serially):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/diagnose_carry_checkpoint.py \
  --scene /path/to/frozen/scene.xml --trace /path/to/physical_trace.jsonl \
  --time 80 --duration 35 --out /tmp/carry-diagnostic-new
```

Output includes input/script hashes, checkpoint, per-case sampled JSONL and a
summary. Loss means object/gripper distance exceeds 0.12 m for at least 0.2 s;
this is a diagnostic criterion, not the benchmark scorer. Inspect forces and
relative displacement as well as the binary event. A no-loss 35 s result is not
proof of indefinitely stable retention. Profile tests: 12 passed; a 0.3 s real
MuJoCo hold smoke retained contact. Full battery pending.

#### Completed checkpoint results (September 24)

All conditions below use the same 80.048 s checkpoint from the loose-closure
repeat, 35 s evaluation duration, and unchanged contact physics. Jobs run
serially, single-threaded with CPU-safe affinity:

| Job | Conditions | Outcome |
| --- | --- | --- |
| `20260924_163132_b815bb` | hold; early straight/turn/brake at 0.05 m/s, 0.2 rad/s; full recorded replay | First four retain contact with 13.06–13.12 mm relative drift. Recorded replay loses the object at +29.48 s. |
| `20260924_163816_418c8a` | wheel-only recorded replay; same synthetic motions after 20 s hold | Wheel-only replay loses at +29.79 s. Delayed synthetic cases retain contact with 13.06–13.23 mm drift. |
| `20260924_164059_c5f7a3` | delayed straight/turn/brake at existing native speed limits, 0.09 m/s and 0.5 rad/s | Turn loses at +24.89 s. Straight and brake retain contact with ~13.06 mm drift. |

Artifacts are in `~/runs/emet/carry-checkpoint-{battery,aged,native-speed}-20260924/`.
The runner now accepts `--motion-delay` and `--modes recorded_wheels` to separate
grip aging and wheel motion from arm/head command changes. Recorded profiles
ignore the synthetic delay. Unit/regression suite: **277 passed**.

![Measured relative drift and contact force](figures/carry-checkpoint-20260924.svg)

Actual motion was checked: the early straight profile travels 0.228 m, turn
rotates 0.773 rad, and braking profile travels 0.197 m; the stationary base
drifts only 0.00033 m. These are not failed-motion "retention successes".
Numbers in the table are commanded speed limits, not measured speed claims.
The faster synthetic turn also covers a larger angle because profile duration
is fixed. **Rate and total turn angle are confounded**; use an equal-angle route
comparison before claiming a rate-only repair or promoting a carry speed limit.

Private model reconstruction further finds the actual can center initially
~1.3–1.4 mm from pad center along pad-local X, drifting to ~24–25 mm before the
original drop; pad half-width is 20 mm. No non-finger object contacts occur in
the source trace from 95 s until contact loss. The gripper command remains
unchanged. This supports outward slip followed by motion-sensitive loss, not
an initially obvious shallow grasp, reopening command, or object/environment
collision. It does not establish a purely numerical cause or hardware safety.

Do not blindly deepen the grasp: the current preset already inherits 25 mm
contact-depth calibration, and the initial pad alignment is near-centered.
Next isolate equal-angle turn profiles and, separately, a clearly labeled
solver-only control for contact creep. Older mirrored/tabletop experiments
already investigated NoSlip and depth corrections; see
[earlier carry diagnostics](shared_grounding_pilot.md). Any physics comparison
must remain explicit and apply consistently across benchmark rows. No new
production speed, squeeze, contact, or safety-margin defaults were changed by
this battery. These are privileged diagnostics, **not agent acceptance**.

#### Predeclared matched-angle / solver-control follow-up

The next eight conditions use the same checkpoint without changing the grasp:
native hold (45 s); measured 1.5-rad turns at command caps 0.5 and 0.2 rad/s
(20 s initial hold, 45 s total); native recorded-wheel replay (35 s); then
NoSlip=10 hold, fast matched turn, recorded-wheel replay, and an 8 s release
negative. The latter ramps the gripper actuator to the explicitly supplied
0.04 open target between seconds 2 and 3. It does not remove an attachment.

The matched turns share feedback and acceleration limit (0.25 rad/s²), command
actual wheel motion, and record measured yaw/target completion. Neither case
counts as matched unless final yaw is within 0.01 rad of the requested angle.
This compares complete rate-limited turn profiles, not constant traversal time.
The solver-only override is explicit in arguments and manifest, along with the
model's original value. It is not a production environment change. If it reduces
creep, that supports a numerical contribution; it does not validate real-world
grasping. Keep the original-physics results and release negative alongside it.

Runner options: `--turn-angle 1.5`, `--noslip-iterations 10`, and
`--modes release --open-control 0.04`. Omitting the solver flag keeps original
physics; omitting turn angle retains the earlier open-loop profiles. Command
profile/validation tests: 22 pass. A real MuJoCo 0.3 s NoSlip smoke retained
contact; the full eight-condition battery is pending.
