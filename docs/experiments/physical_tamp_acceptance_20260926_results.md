# Physical motion and full TAMP acceptance — September 26–27, 2026

**The full TAMP control experiments are complete. Physical task acceptance is not
established.** Both original fixtures now have independently verified physical
pickups. Molmo r32 reused its measured placement route and completed eleven carry
commands before rotational slip exceeded the unchanged retention bound. RoboCasa
has the same failure category. Retained transport and placement remain unsuccessful.

Scope: GT/MCTS, simulated robots, full control registries, and separately scored
physical diagnostics. No learned-model comparison, real-robot trial, or main push.
See the [requested plan](../plans/2026-09-26_physical_tamp_motion_acceptance.md),
[commands and contracts](physical_tamp_acceptance.md), and
[diagnostic history](physical_tamp_acceptance_20260926_diagnostics.md).
Implementation is on `feat/physical-tamp-motion-acceptance`; the latest frozen
runtime candidate is `f5c34fe4` (r32).

## Completed control experiments

| Suite | Source | Result | Artifact root |
| --- | --- | --- | --- |
| Protocol battery | `7b6ad51d` | 17/24 passed; all groups terminal | `/tmp/tamp-protocol-20260926-r5` |
| Small registry | `7b6ad51d` | 4/7 task successes; all rows terminal | `/tmp/tamp-small-20260926-r5` |
| Scripted tool controls | `7b6ad51d` | 3/3 passed; no learned model | `/tmp/tamp-tools-20260926-r5` |
| GT floor registry | `88f8c48e` | 2/3 executed tasks passed; one find-only row deferred under GT-only scope | `/tmp/tamp-floor-20260926-r8` |
| Full registry | `dbe0f4da` | 54 successes, 19 executed failures, 126 errors, 1 invalid fixture; all 200 terminal | `/tmp/tamp-full-20260926-r14` |
| Stretch recheck | `b1c4400c` | 16 successes, 14 fixture errors; all 30 terminal, matching original Stretch outcomes | `/tmp/tamp-stretch-recheck-20260926-r18` |
| Corrected 50-case subset | `fb817dcc` | 18 successes, 15 executed failures, 16 errors, 1 invalid fixture; all terminal | `/tmp/tamp-fixture-recheck-20260927-r22` |

The original full registry contains **110 kinematic-latch and 90 oracle-teleport
cases**. Neither mode establishes contact-based grasping. These are newly executed
results; historical 24/24 scores are not carried forward.

| Robot | Mode | Success | Executed failure | Error | Invalid | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RBY1 | Kinematic latch | 19 | 16 | 75 | 0 | 110 |
| Stretch | Oracle teleport | 16 | 0 | 14 | 0 | 30 |
| Innate Mars | Oracle teleport | 4 | 2 | 24 | 0 | 30 |
| Nori | Oracle teleport | 15 | 1 | 13 | 1 | 30 |

The original full run retained 50 scene-resolution failures for RBY1 scenes 12–21.
Their registry incorrectly requested the training split. The corrected subset
uses the same houses/tasks with `scene_split: val` and the exact installed assets;
all scene-resolution errors are gone. Its 16 remaining errors are missing
landmarks; executed failures comprise seven cleanup and eight blocked-navigation
cases. Two blocked cases cleared all eight selected objects, showing that clearing
the designated clutter does not itself certify a base route.

The corrected subset has a different source and **does not overwrite or merge
into the original 200-row denominator**. Other original failures include missing
receptacles, Mars spawn-search timeouts, cleanup failures, and blocked or missed
navigation goals. Detailed row accounting is retained in:

- [Original matrix](/tmp/tamp-acceptance-report-20260926/full_matrix.png) and [CSV](/tmp/tamp-acceptance-report-20260926/full_results.csv).
- [Corrected subset matrix](/tmp/tamp-fixture-recheck-report-20260927/full_matrix.png) and [CSV](/tmp/tamp-fixture-recheck-report-20260927/full_results.csv).

## Latest physical evidence

| Fixture / run | Source | Independently supported outcome |
| --- | --- | --- |
| RoboCasa can → counter, r26 | `013a8bf1` | Pickup verified; transport stopped on rotational slip. Failed at 362.1 s; no placement. |
| Molmo tomato → bowl, r29 | `82441047` | Approach passed without retries at 551.7 s, with 8.75 mm XY / 0.01496 rad yaw residuals. Grasp stopped before a predicted fingertip/egg collision; failed at 644.7 s. No forbidden contact recorded. |
| Molmo physical r32 | `f5c34fe4` | Pickup verified; 16 approach and 11 carry commands passed. Stopped on rotational slip at 1733.6 s; no forbidden actuation or robot contacts. No release. |
| Molmo physical r30 | `57df3d74` | Pickup independently verified; approach passed, measured placement route passed, redundant navigation search exhausted 60 s. Failed at 1247.6 s; no forbidden actuation or robot contacts. |
| Molmo corrected static r30 | `57df3d74` | Complete open-gripper grasp/lift and carried-route witness found in 64.8 s from the archived measured initial state. Static evidence only. |
| Molmo release-checked static r31 | `1440d034` | Complete witness found in 64.9 s, including open-finger release and retreat. Static evidence only. |
| RoboCasa release-checked static r31 | `1440d034` | Complete witness found in 5.3 s; static evidence only. |

Artifacts: `/tmp/physical-tamp-robocasa-20260927-r26`,
`/tmp/physical-tamp-molmo-20260927-r29`, `/tmp/physical-tamp-molmo-20260927-r30`,
`/tmp/physical-tamp-molmo-20260927-r32`,
`/tmp/static-tamp-molmo-20260927-r30`, `/tmp/static-tamp-molmo-20260927-r31`,
and `/tmp/static-tamp-robocasa-20260927-r31`.

RoboCasa r23, r24, r26 and Molmo r30/r32 independently passed pickup. None completed
physical placement. The 41 completed physical diagnostics span changing implementations;
they are **not a matched reliability sweep**. The machine-readable
[physical ledger](/home/cpaxton/runs/emet/physical-tamp-acceptance-20260926/physical_trials.csv)
retains each source, seed, stage outcome, timeout, and failure reason.

### Navigation and planning findings

Molmo r27 exhausted its 1800 s trial budget in placement search, with 28 route-budget
failures and no motion. Its profiler showed substantial Python work inspecting
irrelevant scene contacts. Equivalent vectorized contact filtering and opt-in
profiling let r28 find a complete tracking-qualified nominal witness in 154.7 s;
its second carried route passed in 47.0 s. The offline contact comparison returned
identical lists on 63 archived checks and retained the known stove-collision
negative. These timings are diagnostics, not a controlled speedup estimate.

R28 then stalled during its final approach turn. Measured turn-induced translation
repeatedly crossed the 20 mm bound, triggering position reacquisition and losing
yaw progress. The wheel midpoint was aligned with the base origin. The repair adds
a bounded forward-position correction during precision turns while retaining
arrival tolerances, configured speed/reverse limits, deadlines and stop checks.
Four disturbance rollouts covering both turn directions and ±4 mm/s creep passed;
r29 subsequently completed the same approach without retries.

[Measured r28 turn failure plot](/tmp/physical-tamp-molmo-20260927-r28/tamp-molmo-r28-turn.png).

### Gripper geometry and retention findings

R29 exposed a real planning-state mismatch: its grasp plan used nearly closed
fingers, but execution opened the gripper before descending. Full-open replay
rejects that grasp and its ±0.08 rad alternatives, with approximately 18 mm egg
overlap at the final target. Wider yaw probes fail IK at that base pose. Execution
stopped at the first invalid next waypoint; the collision was not waived.

R30 declares open-gripper coordinates in the robot profile, including dependent
finger joints. It checks opening and plans with open geometry before arm motion,
then rebuilds paths from measured state after the normal opening command. Static
search rejects the unsafe first approach and finds a different base pose with a
complete witness. Actual grasp width and carried geometry still require measured
lift/placement revalidation.

[Regenerated r29 review image](/tmp/physical-tamp-molmo-20260927-r29/tamp-molmo-r29-side-camera.png)
shows the robot, tomato and clutter. It is rendered from archived measured state;
original camera images and results remain unchanged. The new camera framing is
also used for subsequent stage captures.

RoboCasa r26 retained continuous gripper contact with no scene/support contact
through its pickup/transport interval. Translation drift was 4.505 mm, but rotation
reached 0.10609 rad, exceeding the declared 0.1 rad retention bound.
[Retention plot](/tmp/physical-tamp-robocasa-20260927-r26/robocasa-r26-retention.png).
Private 20-s dynamics replays with fixed arm commands and zero wheel commands
also exceeded that bound: 0.16196 rad with the recorded grip target, 0.14475 rad at
the existing actuator closed limit, and 0.18690/0.17464 rad with ±0.16 rad initial
relative yaw offsets. No stronger-close, alignment, friction, or scoring change
was adopted. A further contact-centering probe moved only the private initial
object state by half/full contact-centroid offsets. It also failed the same bound:
0.16259 and 0.11953 rad, with 7.130 and 7.285 mm translation drift. Job
`20260927_135218_af170f` and scripts/results are archived in `hold_center_replay/`.
No contact-centering change was adopted. These private restored-state probes are
not task trials.

R30's alternate approach completed, both measured grasp replans passed, and the
independent scorer verified pickup. A new carried route passed in 58.5 s after
lift. Execution then ignored this route and launched another 60-s search, which
failed before carry. R32 reuses the selected route only after a fresh full sweep
and goal-endpoint check; stale proposals fall back to bounded search. Per-command
checks and cancellation/replanning after divergence remain mandatory.

R31 also certifies opening at release and retreat with open fingers, repeating
those checks at the measured placement pose before the opening command. Both
fixtures still have static witnesses. Runtime release acceptance remains unproven.

## Final candidate execution

Molmo r32 (`f5c34fe4`, job `20260927_135954_c01dee`) is terminal. It declared
3600 s total / 60 s per route, with profiling off; the larger total budget is
not a matched-budget improvement claim. The approach route passed a fresh sweep,
all 16 approach commands passed without retries, and both measured grasp replans
passed. The independent scorer verified pickup.

Measured placement planning found a carried route in 35.5 s. Its fresh reuse
sweep passed in 14.8 s, eliminating r30's redundant-search failure. Eleven carry
commands passed before a later measurement rejected retention. The executor
recorded 9.195 mm / 0.10912 rad drift relative to its original grasp reference.
Across the independent scorer's 78-s pickup-to-stop interval, maximum drift was
9.561 mm / 0.11082 rad. Gripper contact was continuous; no other-object contact,
forbidden robot contact, or forbidden actuation was recorded. The robot stopped
before release. This is rotational slip, not an observed payload drop.

[Retention plot](/tmp/physical-tamp-molmo-20260927-r32/molmo-r32-retention.png),
[metrics and reference](/tmp/physical-tamp-molmo-20260927-r32/molmo-r32-retention.json),
and [final review image](/tmp/physical-tamp-molmo-20260927-r32/final_1_side.png).
The original trace, result, images and analysis scripts are archived together.

All jobs submitted for this continuation are terminal. No experiment is left
running or queued. The new route reuse and release checks are implemented and
unit-tested; complete physical release remains untested because retention fails
first. The next physical work is stable grasp selection/control under the frozen
physics and bounds, followed by the matched pilot. Broader layouts and partial-map
execution must not be claimed from these two seed-0 diagnostics.

## Validation and acceptance gates

The latest combined targeted suite passed **268 tests**, with Ruff and diff checks
clean. Coverage includes footprint/unknown-space rejection, alternative candidates,
pose/coupled IK, arm/base sweeps, payload contacts, measured arrival, controller
races/cancellation, gripper-state transitions, forbidden actuation, independent
scoring, and registry accounting. Tests do not replace live physical acceptance.

| Gate | State |
| --- | --- |
| A: exact fixtures and failure reproduction | Established for the two original Stretch fixtures, with model/config/state/scorer hashes and execution-mode labels. Initial diagnostics are not a matched baseline pilot. |
| B: complete GT static feasibility | Both fixtures have static witnesses including open-gripper grasp and release/retreat checks. Blocked/alternate/unknown negatives are covered; broader live layout coverage remains outstanding. |
| C: physical pickup, retained carry, release and stable support | Not established. Both fixtures have verified pickups, but no complete physical task has passed; three matched trials per fixture and baseline/candidate comparison remain outstanding. |
| D: partial-observation physical execution | Deferred until C. GT has not been inserted into observed maps. Learned comparisons remain outside the user's scope. |
| Full TAMP control accounting | Complete, including all 200 original rows and separate corrected/recheck ledgers. |

## Review summary and artifact preservation

The branch repairs continuous footprint rasterization and separates map inflation
from footprint/clearance checks. The physical GT driver reuses bounded candidate/MCTS
search, collision/IK paths, shared controllers and independent trace scoring. Shared
repairs address frame consistency, command races, bounded cancellation, precision
arrival, measured placement replanning, revalidated route reuse, and gripper
geometry through release. Physical mode
prohibits live pose-setting, teleports and attachments during scored execution.
Tracking-offset samples do not apply the nominal 22 cm center margin a second time.

The full multi-robot controls are frozen at their listed sources; they are not a
same-source physical non-regression claim for the latest candidate. Physical
controller support remains limited to the supported telescoping-arm interface.
Broader robot adapters, stable grasp dynamics, the matched pilot and partial-map
execution remain explicit work in [TODO](../../TODO.md).

Completed roots are archived under
`/home/cpaxton/runs/emet/physical-tamp-acceptance-20260926/`, with original `/tmp`
paths retained as symlinks. Physical and static indexes are separate. Trials retain
manifests, scene/model hashes, measured states, stage events, contacts, actuation
audits and server logs; control runs retain per-row simulator logs. Failed and
canceled diagnostics are preserved. Detailed pre-consolidation working notes are
in `reports/working_notes_before_consolidation_20260927.md` in that archive.
All heavy runs use `emet jobs --cpu-safe --gpu-exclusive` with OMP/OpenBLAS/MKL
threads set to one. Other agents' checkouts and running jobs are untouched.
