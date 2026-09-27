# Physical motion and full TAMP acceptance — September 26, 2026

**Physical acceptance remains unestablished.** Both matched Stretch fixtures now
produce complete static navigation/grasp/lift/transport/release witnesses, but no
contact-based pickup and placement has passed the independent scorer. Static
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
| Physical RoboCasa can → counter | `dbe0f4da` | Failed before closure: measured telescoping joint exceeded nominal limit | `/tmp/physical-tamp-robocasa-20260926-r14` |
| Physical Molmo tomato → bowl | `dbe0f4da` | Failed approach, unconfirmed stop after 431 s | `/tmp/physical-tamp-molmo-20260926-r14` |

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

Full job: `20260926_125932_8d0984`. Physical jobs: `20260926_125430_da88fd` (RoboCasa),
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

The 127 error/invalid rows comprise 50 missing-scene merge failures (RBY1
scenes 12–21), 45 missing landmarks, 12 missing receptacles, and 20 Mars startup
timeouts. These remain in the denominator. The final matrix and row-level CSV
are in `/tmp/tamp-acceptance-report-20260926/{full_matrix.png,full_results.csv}`.

R17 RoboCasa failed fresh arrival IK at 15.7 mm position / 0.066 rad orientation;
its measured base arrival was 12.0 mm / 0.0149 rad from the planned endpoint.
R17 Molmo timed out at 603 s during approach. Neither trial picked or placed.
Independent scoring found no forbidden actuation or audited robot collision.

Revision `8c3b91b6` replaces interpolated-yaw waypoints with collision-validated
turn/drive/turn routes, keeping translations at most 0.2 m and checking the
travel heading even when it lies outside the endpoint-yaw interval. The focused
navigation/evaluation suite passed 25 tests. Physical r19 jobs
`20260927_074239_7a500c` (RoboCasa) and `20260927_074239_3ed41a` (Molmo) are queued
serially; no new physical outcome is claimed from submission.

The exact missing scenes 12–21 and their object dependencies were installed on
September 27. The first installer process crashed after scene 14; the remaining
assets completed with safe CPU affinity. A separate 50-row recheck, frozen at
`b48b8898`, retains the original case definitions and runs under job
`20260927_074941_39ce43`, root `/tmp/tamp-fixture-recheck-20260927-r20`.
It does not replace the original 50 fixture errors; results are pending.

A subsequent bounded grasp-candidate filter checks IK at 16 base-pose samples
(eight directions at the existing 20 mm arrival radius, each at ±0.03 rad yaw).
It retains the original IK tolerances and 5 mm joint reserves. Exact r17 replay
rejects the selected candidate at a sampled arrival with 13.7 mm position error.
This is sampled candidate screening, not a continuous robustness proof;
execution still requires fresh measured-state IK and collision-checked paths.
The combined targeted suite now passes **236 tests**. Live validation is pending.

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

The combined targeted suite passed **232 tests at `b1c4400c`**, including command
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
OMP/OpenBLAS/MKL threads set to one. Physical trials have a 600 s wall budget;
registry rows have 900 s. A crash/timeout stops registry execution for cleanup
inspection and explicit resume; earlier failures are never overwritten.
Other agents’ checkouts and jobs are untouched. Physical controller support is
currently limited to the supported Stretch telescoping-arm interface; the wider
robot control batteries do not imply physical support for every robot.
