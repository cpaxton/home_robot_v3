# Physical motion and full TAMP acceptance — September 26, 2026

**Physical acceptance remains unestablished.** Earlier revisions produced static
witnesses for both matched Stretch fixtures,
but controller-aware replay exposed missing travel turns in Molmo’s transport
route. RoboCasa r23 passed independently scored pickup but lost the can during transport.
No complete contact-based pickup and placement has passed the independent scorer. Static
feasibility and controller acknowledgements are not task success.

Scope: GT/MCTS, simulated robots, the existing full TAMP registries, and a separately
scored physical track. No learned-model comparison, real-robot trial, or main push.
The requested plan is [the motion acceptance handoff](../plans/2026-09-26_physical_tamp_motion_acceptance.md).
Earlier raw findings and canceled submissions are preserved in
[the diagnostic history](physical_tamp_acceptance_20260926_diagnostics.md).

Completed physical and control roots are durably archived under
`/home/cpaxton/runs/emet/physical-tamp-acceptance-20260926/`, retaining their directory
basenames. The original `/tmp` paths below are symlinks for completed runs.
`physical_index.json` indexes the archived physical diagnostics; each includes
a copied `server.log`. Active runs are moved only after completion.

## Experiment results

| Suite | Source | Result | Artifact root |
| --- | --- | --- | --- |
| Protocol battery | `7b6ad51d` | 17/24 tests passed; all six groups terminal | `/tmp/tamp-protocol-20260926-r5` |
| Small registry | `7b6ad51d` | 4/7 task successes; all rows terminal | `/tmp/tamp-small-20260926-r5` |
| Scripted tool controls | `7b6ad51d` | 3/3 passed; scripted JSON, no learned model | `/tmp/tamp-tools-20260926-r5` |
| GT floor registry | `88f8c48e` | 2/3 executed tasks passed; one find-only row deferred under GT-only scope | `/tmp/tamp-floor-20260926-r8` |
| Full registry | `dbe0f4da` | 54/200 task successes; all 200 terminal | `/tmp/tamp-full-20260926-r14` |
| Corrected 50-case subset | `fb817dcc` | 18 successes, 15 executed failures, 16 errors, 1 invalid fixture; all terminal | `/tmp/tamp-fixture-recheck-20260927-r22` |
| Latest completed physical RoboCasa can → counter | `013a8bf1` | Verified pickup; rotational slip stopped transport, failed at 362.1 s | `/tmp/physical-tamp-robocasa-20260927-r26` |
| Latest completed physical Molmo tomato → bowl | `cc44da56` | Static placement search timed out at 1802.5 s; no execution | `/tmp/physical-tamp-molmo-20260927-r25` |

The full registry contains **110 kinematic-latch and 90 oracle-teleport cases**.
Neither mode establishes contact-based grasping. All results above are newly run;
the historical 24/24 battery is not carried forward as a current result.

Protocol failures: Nori scene 1 cleared 8/8 obstacles but failed final navigation;
Mars scene 0 passed manipulation and failed both navigation cases; Mars scene 1
had four startup timeouts while searching for a collision-free spawn. RBY1 passed
both scenes, 8/8. Small-registry failures: one latch cleanup placed 5/6 objects
(the potato failed attachment verification), and two named-navigation rows lacked
the requested landmark. The paired teleport cleanup passed 6/6. Floor: RBY1
picked but failed independently verified placement despite successful controller
returns; Sourccey and Stretch oracle controls passed.

Full job: `20260926_125932_8d0984`. Earlier r14 physical jobs: `20260926_125430_da88fd` (RoboCasa),
`20260926_125430_146377` (Molmo), frozen in
`/tmp/emet-physical-tamp-20260926-r14`. Full jobs r12 and r13 were canceled before
execution to repair the command race and protocol boundary; they produced no
registry rows. Submission is not completion. Read each terminal result with its
manifest, trace, audit, and stage events.

## September 27 completed registry analysis

| Robot | Mode | Success | Executed failure | Error | Invalid | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RBY1 | Kinematic latch | 19 | 16 | 75 | 0 | 110 |
| Stretch | Oracle teleport | 16 | 0 | 14 | 0 | 30 |
| Innate Mars | Oracle teleport | 4 | 2 | 24 | 0 | 30 |
| Nori | Oracle teleport | 15 | 1 | 13 | 1 | 30 |

The 127 error/invalid rows comprise 50 scene-resolution failures (RBY1
scenes 12–21; the requested training split was incorrect), 45 missing landmarks, 12 missing receptacles, and 20 Mars startup
timeouts. These remain in the denominator. The final matrix and row-level CSV
are in `/tmp/tamp-acceptance-report-20260926/{full_matrix.png,full_results.csv}`.

Of the 19 executed failures, seven were cleanup failures (RBY1), eleven were
navigation cases whose path remained blocked (nine RBY1, two Mars), and one
Nori navigation case had an open path but failed to reach the goal. Several
blocked cases had cleared all eight designated objects, so clearing the selected
clutter alone did not certify a usable base route. Mars startup logs stop in
collision-free spawn search after binding the transport ports; these are not
successful motion tests.

## Current physical candidate and fixture recheck

The latest completed pair, r22 (`fb817dcc`), failed before pickup. RoboCasa
stalled during its final approach turn after 426.6 s. Molmo found a complete
static witness and executed approach segments, then hit its wall deadline at
602.3 s. Neither result passes physical acceptance.

The current candidate, r23 (`1b3f57bf`), retains collision-checked differential-drive
RRT steering and returns to the original precision arrival bounds of 20 mm /
0.03 rad. It declares three grasp yaw alternatives (0 and ±0.08 rad), each subject
to the same IK, joint-margin, and fresh collision-path checks. Offline RoboCasa
replay passed all 16 sampled arrivals using these alternatives. This is bounded
sampling, not a continuous robustness proof or an executed task success.

RoboCasa r26 and Molmo r25 are terminal. R26 verified pickup but stopped on rotational slip during transport. The trace
retains gripper contact and records no scene contact: relative translation drift
was about 4.5 mm, but relative rotation reached approximately 0.106 rad versus
the declared 0.1 rad retention bound. This is not a physical drop; the bound was
not relaxed. R25 exhausted its 1800 s trial deadline during static
placement search, so it did not exercise the final-turn controller repair.

Molmo diagnostic r27, seed 0 (`8e9d3189`, job `20260927_120832_8191e6`,
`/tmp/physical-tamp-molmo-20260927-r27`) is running and declares 1800 s overall and 60 s per
base route before execution. The nominal/measured base retains the 22 cm
center-clearance gate; tracking offsets check full robot/payload geometry without
applying that center margin a second time. Placement rejects endpoint and arm-path
failures before spending the carried-route budget. Exact archived Molmo replay
passed the full previous route's 26-offset checks in 16.17 s; route search plus
verification found 18 waypoints in 33.71 s. Thus the prior 10 s route budget could
not even verify this valid route. This is static evidence, not physical success.
The longer budget is not a matched-budget comparison with earlier diagnostics.


The first asset-only 50-case recheck (`b48b8898`, job
`20260927_075453_e06bcd`) retained 50 scene-resolution errors. Installing the assets
was insufficient: FloorPlans 13–22 belong to MolmoSpaces' validation split. The
corrected registry adds only `scene_split: val` to those 50 definitions; the other
150 rows are unchanged. Original results and the failed asset-only attempt remain
archived. The corrected subset must be reported separately from the original
full 200-case run.

The corrected subset finished all 50 cases: **18 successes, 15 executed task
failures, 16 missing-landmark errors, and one invalid fixture**. No scene-resolution
errors remained. Seven executed failures were cleanup placement failures; eight
were navigation cases whose final route remained blocked. Clearing the selected
objects did not guarantee a usable route: two failed navigation cases cleared all
8/8 selected objects. The invalid case (`ithor_nav_goal_19_rby1_n8_counter_3`)
had an open initial route and therefore failed the intended blocked-route fixture
condition. Its retained execution record shows 7/8 relocations and a blocked final
route; it is not a success. All 50 cases use kinematic latch.

The corrected matrix and row-level CSV are in
`/tmp/tamp-fixture-recheck-report-20260927/{full_matrix.png,full_results.csv}`.
The original full run and the corrected subset have different frozen sources and
must not be presented as one 200-case rerun.

## Transport tracking-envelope repair

RoboCasa r23 (`1b3f57bf`) passed measured approach arrival and fresh grasp-path
validation. Its pickup passed the independent scorer. During transport the can
contacted the stove at approximately 65.8 s simulation time and left the gripper
at 66.53 s. The task failed at 383.1 s wall time, with forbidden gripper
self-contacts also recorded after payload loss. The stage event correctly recorded
`payload_not_retained`; the terminal message reduced it to `invalid_measured_pose`.

Exact-state replay reproduces the contact using both the optimized collision pass
and full MuJoCo position pipeline. The last nominal segment passes, but a −0.015
rad yaw deviation, inside the existing acceptance bounds, hits the stove. A new
opt-in carried-route check samples 26 offsets around the existing 20 mm / 0.03 rad
bounds, with no relaxation of contact or IK criteria. It rejects that segment and
the original placement endpoint, whose envelope also intersects the knife block.
The bounded search must select another candidate; this rejection is not a live
success. Navigation failure results now retain the measurement exception reason.
The combined targeted suite passes **248 tests**. The repair is frozen at
`4a821be1` in r24, queued under job `20260927_111453_79d184` with the same 600 s
RoboCasa budget. Molmo r23-long retained its original frozen revision and failed
approach at 921.3 s after bounded final-turn retries, with no pickup.

## Fresh position acquisition for final turns

Molmo r23-long passed its approach translations but stalled on final-turn attempts
at 612.6 s and 757.5 s, with XY errors already inside the declared 20 mm bound.
Command startup reset XY acquisition and required another tiny translation before
turning; the controller and arrival monitor disagreed about useful progress in
that interval. The next repair initializes XY acquisition from fresh measured
feedback when it already meets the declared bound. It does not reuse cached
success or change arrival tolerances. The existing hysteresis still requires
returning to the inner target after drift outside the outer bound. Execution-path
screening uses the same startup rule. A real-controller regression verifies
immediate turning inside the bound and required translation with fresh feedback
outside it, even when cached feedback was inside. The combined suite passes
**250 tests**. Source `cc44da56` is frozen in r25; Molmo job
`20260927_112354_b401b7` uses the same 1800 s budget as r23-long.

## Placement alternatives after measured lift

RoboCasa r24 (`4a821be1`) found an envelope-checked static alternative and again
passed independent pickup. It refused transport at 253.8 s because the measured
carried configuration made the planned placement endpoint intersect the knife
block within the tracking envelope. This is a safe rejection, not task acceptance;
no forbidden actuation or robot contacts were recorded.

The next repair extracts the existing bounded placement loop into a shared helper
and reruns it after measured lift, trying the original placement before the same
ordered alternatives. A replacement must pass transport and all placement/retreat
paths; no live motion begins on search failure. Private planning state and the
original grasp transform are restored, preserving retention checks. Original and
executed placement witnesses remain in the artifact. Exact r24 replay finds a
replacement at approximately (3.242, −1.175, 2.358) in 30.9 s after 143 rejections.
The targeted set now covers **253 passing tests**, including alternative selection,
exhaustion, state restoration, and propagation of placement-replan failures.

## Implemented motion contracts

- Raw obstacles, uncertainty, and footprint are separate. Continuous-pose SAT
  rasterization fixes fractional-cell over-inflation without marking unknown cells
  free. The archived handoff start covers 24 cells with no unknown or occupied cells.
  The 0.22 m planner clearance remains unchanged; physical-map mode remains opt-in.
- Bounded candidate search requires orientation-aware arm IK, collision-checked
  approach and arm segments, payload-aware transport, and support release/retreat.
  Exhaustion reports `no_plan_within_budget`, not geometric impossibility.
- IK rejects nonfinite/out-of-limit targets and weights position/orientation errors
  by their existing tolerances. Coupled telescoping joints preserve their actuator
  relationship. Support candidates use real collision surfaces and bounded interior
  alternatives, avoiding parked fixture geometry and occupied surface centers.
- Execution uses shared wheel, arm, and gripper APIs. Measured arrival triggers
  fresh IK/path validation; arm convergence, base drift, actual lift, and payload
  retention are checked. Missing collision support and failed motion remain failures.
- A live actuation guard prohibits pose writes, teleport, latch/attachment, reset,
  and kinematic base holding during the scored physical interval. Independent
  every-tick scoring checks contacts, lift, retention, release, and stable support.
- Manifests archive source/config/model hashes, initial states, tolerances, seeds,
  budgets, candidates, rejected contacts, controller residuals, and execution mode.
  Physical trials save head/depth/side views, route maps, exact MJBs, and traces.
  Registry accounting retains errors, invalid fixtures, crashes, timeouts, and skips.

## Demonstrated faults and repairs

| Fault | Evidence and repair |
| --- | --- |
| Disabled visualization consumed a CPU core | Stretch treated its null visualizer as enabled and ran an unthrottled loop. Checking enabled state and limiting update rate reduced RoboCasa planning from 548 s to 13.7 s. |
| Arm acknowledgment preceded measured convergence | Client tolerances admitted the previous wrist target. The physical executor now checks measured convergence within the declared tolerance and bounded settling window. |
| Coarse navigation did not establish manipulation reach | Precision policy uses 20 mm XY / 0.03 rad yaw; fresh IK is still required at the measured arrival. |
| Mixed meter/radian IK residuals rejected reachable arrivals | Exact archived RoboCasa replay changed from 160-iteration failure at 26.3 mm to an 11-iteration solution at 5.7 mm / 0.0977 rad, retaining 10 mm / 0.1 rad acceptance. |
| Wall time misclassified slow simulated progress | Progress and low-level motion budgets now account for advancing simulation time. Stale-data checks and overall wall deadlines remain active. |
| Arm commands restored a stale manipulation base pose | RoboCasa r11 missed the can because the base turned back toward the pre-navigation reference. Navigation now explicitly changes mode; arm paths refuse measured base drift. |
| Physics acknowledgements erased new arm targets | RoboCasa r12 acknowledged −0.502 rad while the wrist remained at −0.407 rad. Producer and physics consumer now share an interprocess lock around command read/modify/acknowledgement. |
| A trial timeout became an ordinary approach failure | The whole-trial deadline now escapes controller/task recovery handlers and records a timeout before cleanup. |

RoboCasa r11 failed lift-start collision revalidation after closing an empty gripper.
The scorer recorded opposing finger/pad contacts. Those contacts were not waived;
the stale base frame was repaired. Molmo r11 timed out at 606 s before grasp.
RoboCasa r12 stopped safely in preparation on measured wrist tracking failure.
R13 completed preparation on both fixtures but rejected navigation because a mode
field violated the standalone-command wire contract. Revision `dbe0f4da` moves
the mode transition inside the accepted server command and adds a regression.
All failed trials remain archived under `/tmp/physical-tamp-{scene}-20260926-r*`.

Subsequent physical-only candidate `04cc4960` reserves 5 mm at each telescoping
joint limit during pose IK. The measured 0.4 mm overshoot remains a rejection;
commanded poses must now leave tracking room, and search may choose a closer base.
It also uses position hysteresis for precision navigation: translation targets
10 mm, final turning remains active within 20 mm, and leaving that outer limit
requires returning to the inner target. Yaw control targets 0.015 rad; independent
arrival acceptance remains 20 mm / 0.03 rad. Molmo r14 took 199 s for its second
small waypoint while repeatedly crossing the translation/turn threshold. No
clearance, hard joint limit, contact, or scoring threshold was relaxed.

The r17 physical pair (`014b6ab4`) finished after the full r14 control registry:
`20260926_131239_4220c6` (RoboCasa), `20260926_131240_49836d` (Molmo). R15/r16 jobs
were canceled before execution as the coupled precision/cancellation repairs were
completed. The new interprocess lock exposed cancellation waiting for an
acknowledgement while holding the consumer's lock; r17 releases it before waiting
and still requires consumed stop plus fresh measured rest. Timeouts also retain
independent trace scoring. This interaction explains the unconfirmed stop in
Molmo r14; it is not counted as successful navigation.

A separate 30-row Stretch recheck at `b1c4400c` finished after the physical pair:
job `20260926_131456_47c2bb`, artifacts `/tmp/tamp-stretch-recheck-20260926-r18`.
The explicit `--robot stretch` subset preserves original registry case definitions.
It produced 16 successes and 14 fixture errors, exactly matching the original Stretch outcomes, without overwriting the full r14 ledger. The full registry remains frozen at `dbe0f4da`; later physical-only results
must carry their own source revision and may not replace those control results.

## Validation and remaining gates

The combined targeted suite passed **254 tests for the r27 candidate**, including command
snapshot, deadline, adapter-mode, joint-margin, hysteresis, cancellation, and
explicit-subset regressions. Tests cover collision and unknown
space rejection, coupled/pose IK, alternative search, measured arrival, arm/base
tracking, payload slip/drop, forbidden actuation, and full-registry accounting.
They do not establish live physical acceptance.

| Plan gate | State |
| --- | --- |
| A: frozen fixtures and reproduced failures | Established for the two exact Stretch fixtures; initial diagnostics are not a matched baseline comparison. |
| B: complete GT static witnesses | Demonstrated on both fixtures, with unit-level blocked/alternate/unknown negatives; broader live layout coverage remains outstanding. |
| C: physical pick, carry, release, stable support | Not established. Three matched seeds per fixture and baseline/candidate comparison remain outstanding. |
| D: partial-observation physical execution | Not established; depends on C. GT has not been injected into observed maps. Learned comparisons excluded by the user’s scope. |
| Full TAMP control accounting | Complete: all 200 rows terminal. 54 successes, 19 executed task failures, 126 errors, one invalid fixture. |

All heavy jobs run serially through `emet jobs --cpu-safe --gpu-exclusive`, with
OMP/OpenBLAS/MKL threads set to one. Physical trials have a 600 s wall budget,
except the explicitly declared 1800 s Molmo r23-long/r25/r27 diagnostics; registry rows
have 900 s. A crash/timeout stops registry execution for cleanup
inspection and explicit resume; earlier failures are never overwritten.
Other agents’ checkouts and jobs are untouched. Physical controller support is
currently limited to the supported Stretch telescoping-arm interface; the wider
robot control batteries do not imply physical support for every robot.
