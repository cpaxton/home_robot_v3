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
