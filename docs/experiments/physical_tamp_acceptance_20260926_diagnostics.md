# Historical diagnostic notes

These notes preserve intermediate submissions and findings. Statements about pending
runs describe the time they were written. See [the current report](physical_tamp_acceptance_20260926_results.md)
for the latest results and acceptance verdict.

# Physical motion and TAMP experiment ledger — September 26, 2026

This report separates contact-based physical acceptance from the existing teleport
and latch controls. The requested scope is GT/MCTS only. No learned-agent comparison,
real robot test, or main-branch push is part of this run.

## Completed evidence

| Check | Outcome | Evidence |
| --- | --- | --- |
| Archived fractional start footprint | 24 cells; zero unknown or occupied cells after continuous-pose rasterization | Original map: `/home/cpaxton/runs/emet/navigation-floor-recovery-20260926/offline_initial_footprint/map.npz`; regression in `test_navigation_sweep.py` |
| Targeted motion, scoring, task, and runner tests | 147 passed at `7b6ad51d` | Includes collision/unknown-space negatives, orientation IK, bounded tracking, payload slip/drop, forbidden actuation, and 200-row registry accounting |
| Molmo physical diagnostic, seed 0, `b43dcc5e` | Timeout at 600-second budget; no complete witness or executed approach | `/tmp/physical-tamp-molmo-20260926-r0/result.json`, `stages.jsonl`, `physical_trace.jsonl`, `motion_overview.png` |
| Protocol infrastructure attempt, `b43dcc5e` | 24/24 failed before simulation: frozen checkout could not locate the installed MolmoSpaces wrapper | `/tmp/tamp-protocol-20260926-r0/ledger.json` and `summary.json` |

The first physical diagnostic is an implementation diagnostic, **not a matched
baseline comparison**. Its rejected approaches exposed arm/furniture collisions
while retaining the startup arm pose. Subsequent changes add a collision-checked
profile navigation posture, bounded joint-limit IK, continued placement-pose search,
correct wrist command ordering, and exact live-model provenance. No collision margin
or unknown-space rule was relaxed. The MolmoSpaces environment binding is repaired
for subsequent oracle trials; the failed infrastructure attempt is retained.

The overview image for the first trial was generated after the run from its archived
model and measured initial state using the revised visualization helper. It is a
geometry diagnostic, not a contemporaneous execution image.

## Frozen experiment revision and submissions

Source: `f4bbc3741847a7307fb2fce26a6c8f9a30443bc6` in
`/tmp/emet-physical-tamp-20260926-r2`. Installed Python environments are bound into
this worktree; the MolmoSpaces package is also placed on its explicit Python path
for the oracle suites. Source changes are not made in a running worktree.

All jobs use `emet jobs --cpu-safe --gpu-exclusive`, one numerical thread per
library, and the shared GPU lock. Other agents' jobs are left untouched. Physical
trials have a 600-second wall budget; registry rows have 900 seconds, with four
such budgets per protocol process. A timed-out registry run stops for cleanup
inspection and retains pending rows for explicit resume.

| Submission | Job ID | Artifact root | Latest recorded state |
| --- | --- | --- | --- |
| Molmo physical, seed 0 | `20260926_095526_87bd04` | `/tmp/physical-tamp-molmo-20260926-r2` | Timeout; no complete witness |
| RoboCasa physical, seed 0 | `20260926_095529_bbe280` | `/tmp/physical-tamp-robocasa-20260926-r2` | No plan within budget; no physical pickup |
| 24-case oracle protocol | `20260926_095623_46f75a` | `/tmp/tamp-protocol-20260926-r2` | Cancelled before execution to prioritize fixture repairs |
| Seven-row small registry | `20260926_095804_73c06c` | `/tmp/tamp-small-20260926-r2` | Cancelled before execution to prioritize fixture repairs |
| Four GT floor rows | `20260926_095807_b4c7a6` | `/tmp/tamp-floor-20260926-r2` | Cancelled before execution to prioritize fixture repairs; find-only row is deferred |
| Full 200-row registry | `20260926_095810_983359` | `/tmp/tamp-full-20260926-r2` | Cancelled before execution to prioritize fixture repairs |
| Three scripted tool controls | `20260926_095907_ffcfa4` | `/tmp/tamp-tools-20260926-r2` | Cancelled before execution to prioritize fixture repairs |

Submission is not completion or acceptance. Read each terminal `result.json` or
`summary.json` together with its ledger. Protocol/registry labels distinguish
`oracle_teleport` and `kinematic_latch`; neither establishes physical grasping.
The physical adapter currently requires the supported telescoping-arm client
interface. Other interfaces must report unsupported capability until implemented.

Three earlier revision-1 submissions were cancelled while still queued to repair
the physical wrist binding and put fixture diagnostics first. They executed no
trial and are not counted as successes or failures: `20260926_094735_98ce0f`,
`20260926_095011_ee8874`, and `20260926_094936_3da2e7`.

## Acceptance still required

The three-seed controlled physical gate, broader layout positives/negatives,
partial-observation execution, and a matched baseline are not yet established.
Partial-map adoption must preserve observed-space rejection and measured arrival
checks. The full suite requires terminal accounting for all 200 rows, including
unsupported robots, invalid fixtures, timeouts, and crashes. Publication claims
and promotion to shared live defaults remain gated on that evidence.


## Revision 3 fixture diagnostic

Revision `f7ea8921` is frozen in `/tmp/emet-physical-tamp-20260926-r3`. It generates
40 approach candidates from five actual arm-extension fractions and eight base
yaws, plus the measured stationary pose. It tries both nominal and transit wrist
orientations, bounds each base-route search to 10 seconds, and avoids dynamics
solves in offline kinematic updates. Measured height/tilt are retained. The arm
planner can try raise/orient/extend group orders when a simultaneous coupled-joint
move collides. Arrival triggers fresh world-pose IK; lost payload or invalid feedback
stops subsequent motion.

- Molmo seed 0: `20260926_101911_c23092`, `/tmp/physical-tamp-molmo-20260926-r3`.
- RoboCasa seed 0: `20260926_101915_21ea6a`, `/tmp/physical-tamp-robocasa-20260926-r3`.

These submissions are diagnostics, not evidence of a passed gate. The revision-2
bulk jobs were cancelled before execution so fixture debugging would not wait
behind the long registry sweep; they need new submissions after this diagnostic.

## Revision 4 and regression submissions

Both revision-3 physical trials timed out without a complete witness (Molmo
653.2 seconds including cleanup; RoboCasa 621.4 seconds). Revision 4 at
`0fab5a0e` adds geometry-based support heights, verifies the rigid collision
kernel against the full position pass, records actual gripper-opening commands,
and holds the measured head pose during arm execution.

- Molmo: `20260926_104222_c774d8`, `/tmp/physical-tamp-molmo-20260926-r4`.
- RoboCasa: `20260926_104225_b9df40`, `/tmp/physical-tamp-robocasa-20260926-r4`.

The Molmo collision-kernel comparison matched 189 contacts across 2178 geoms.
Full position took 0.0052 seconds and collision-only took 0.0063 seconds at the
measured initial state; this does **not** establish a performance improvement.
A bounded archived-route profile is queued as `20260926_105125_c4bed4` with
output in `/tmp/tamp-route-profile-job-20260926`. Revision 5 adds route-level
query counts/timing and prevents process failures from inheriting task success
from a written artifact. Signal crashes stop the serial runner for cleanup.

Frozen regression source: `7b6ad51d`, `/tmp/emet-physical-tamp-20260926-r5`.
All submissions below are pending until their terminal ledgers are inspected.

| Suite | Job ID | Artifact root |
| --- | --- | --- |
| Protocol, 24 tests | `20260926_105141_134946` | `/tmp/tamp-protocol-20260926-r5` |
| Small, 7 rows | `20260926_105235_29c089` | `/tmp/tamp-small-20260926-r5` |
| GT floor, 4 rows | `20260926_105236_f9f497` | `/tmp/tamp-floor-20260926-r5` |
| Full, 200 rows | `20260926_105236_43eed7` | `/tmp/tamp-full-20260926-r5` |
| Scripted tool controls, 3 items | `20260926_105251_7b1af8` | `/tmp/tamp-tools-20260926-r5` |

## Revision 4 terminal outcomes and follow-up

Molmo timed out without a full witness (657.2 seconds including cleanup).
RoboCasa exhausted its search without a full witness (617.9 seconds including
cleanup). Two RoboCasa candidates passed the offline approach/grasp/lift sequence,
but each rejected all 40 transport alternatives. Many endpoint contacts involved
the toaster at the selected counter center. No physical pickup or placement ran.

Revision `0d08387a` searches nine interior release positions per support level,
retaining payload-sized edge clearance and full geometry validation. Its focused
16 tests pass. It is frozen in `/tmp/emet-physical-tamp-20260926-r6`; RoboCasa
job `20260926_110716_7c4a47` was cancelled before execution to add a live
planning profile. Replacement revision `9893261c` is frozen in
`/tmp/emet-physical-tamp-20260926-r7`; job `20260926_111629_106b81` is queued
with output `/tmp/physical-tamp-robocasa-20260926-r7`.

The archived-route profile completed two routes in 1.035 seconds (798 validity
calls). A separate symbolic candidate profile took 0.058 seconds. Neither
reproduces the live route delays; these are diagnostic timings, not a speedup
claim or an execution witness. A moved-pose contact-count probe matched the full
position pass, but this alone is not broad collision-kernel validation.

The revision-5 protocol has completed Nori scene 0 with all four oracle tests
passing. The remaining protocol, registry, floor, and tool results are pending.

Protocol interim result: 9/12 tests passed across Nori scenes 0/1 and Mars scene 0.
Nori scene 1 cleared 8/8 objects and found an open route but did not report final
navigation arrival. Mars scene 0 passed manipulation and failed both navigation
checks, including an explicit no-route rejection. These are preserved failures,
not evidence of fixture invalidity or a matched regression attribution. The
historical 24/24 result predates current navigation guards.

The Mars scene-1 protocol failures were all startup failures: the server logs
stopped at collision-free spawn search before the client's 60-second observation
wait expired. Planning never started. RBY1 scene 0 subsequently passed 4/4;
completed groups total 13/20, with RBY1 scene 1 still in progress.

## Completed regression suites

The revision-5 protocol is complete: **17/24 tests passed**, with all six process
ledgers terminal. Four failures were Mars scene-1 startup timeouts; three were
navigation failures (Nori scene 1 blocked-navigation, Mars scene 0 both navigation
checks). RBY1 passed 8/8. Evidence: `/tmp/tamp-protocol-20260926-r5`.

The small registry is complete: **4/7 tasks succeeded**. The six-object latch
row cleared 5/6 and failed potato attachment verification; its teleport control
cleared 6/6. Two navigation rows failed with `missing_landmark` for their named
sofa/fridge. The eight-object latch/navigation row and final Stretch row passed.
All seven rows remain in `/tmp/tamp-small-20260926-r5/ledger.json`.

The unstarted full job `20260926_105236_43eed7` was cancelled only to reorder our
queue. Its unchanged source, commands, output root, and budgets were resubmitted
as **`20260926_114041_1123fe`**, job log
`/tmp/tamp-full-job-20260926-r5-final/job.log`. It follows the floor, scripted tool,
and profiled physical diagnostic jobs, so the 200-row sweep is the final run.
It contains 110 kinematic-latch and 90 oracle-teleport rows.

Scripted tool controls finished **3/3 PASS** at `7b6ad51d`; see
`/tmp/tamp-tools-20260926-r5/gate_summary.txt`. These remain explicitly teleport
and kinematic controls, not physical grasp acceptance.

The original floor run completed with three infrastructure errors from missing
RoboCasa asset bindings and one `deferred_gt_only` find-only row. The installed
25 GB assets were bound into a separate checkout of the same source,
`/tmp/emet-tamp-floor-20260926-r5-assets`; its asset-completeness check passed.
The rerun is `20260926_114728_7be2d7`, output
`/tmp/tamp-floor-20260926-r5-assets`. The full job `20260926_114041_1123fe`
was cancelled before execution while this prerequisite environment problem is
repaired; the full sweep still needs final submission after these diagnostics.

## Static witnesses and live scheduling repair

Revision 7 did find a complete RoboCasa static witness; the wall limit then
interrupted approach after a measured navigation-posture transition. Its earlier
interim absence of a witness must not be treated as the terminal outcome.

Live profiling exposed a Stretch client scheduling defect: the disabled
`NullVisualizer` was truthy, so Stretch launched a busy Rerun loop. Per-thread CPU
was about 91% for that loop versus 5% for the planning thread. Revision `88f8c48e`
checks the visualizer's explicit enabled flag and caps live visualization at
30 Hz. The expanded targeted suite passes **175 tests**.

At revision 8, **both fixtures found complete static pick/place witnesses**.
RoboCasa planning took 13.7 seconds; its full trial failed at 60.8 seconds.
Molmo's full trial failed at 65.8 seconds. Both stopped on measured arm residuals
during the navigation-posture transition, before base approach or pickup. For
RoboCasa the pitch command was -1.400 rad while the simulator still held the
previous command near -1.300 rad; client success used a looser tolerance.

Revision `195e3952` adds a bounded five-second wait for measured convergence,
retaining the 0.03 joint residual gate and checking payload retention. It also
preserves executor failure messages through approach reporting. Its 45 focused
tests pass. New trials are:

- RoboCasa: `20260926_120811_875d2a`, `/tmp/physical-tamp-robocasa-20260926-r9`.
- Molmo: `20260926_120811_8211ce`, `/tmp/physical-tamp-molmo-20260926-r9`.

The first asset-bound floor retry stopped before creating any episode ledger
because local dependency symlinks appeared untracked. After fixing only local
Git ignore metadata, revision-8 floor job `20260926_120320_037935` completed:
**2/3 executed tasks passed**, with one find-only row explicitly deferred for
GT-only scope. RBY1 latch pickup passed but independent placement failed despite
both controller calls returning success. Sourccey and Stretch oracle controls
passed. Evidence: `/tmp/tamp-floor-20260926-r8/ledger.json` and `summary.json`.
The full 200-row sweep remains unstarted pending the final physical diagnostics.

## Measured arrival follow-up

Revision 9 passed arm settling. RoboCasa physically executed its base approach,
then correctly refused grasping: fresh IK at the measured arrival had 68.3 mm
position error. Molmo stopped on base tracking. Revision `fdbd0402` requests the
existing `precision` navigation policy (20 mm XY, 0.03 rad yaw) and removes a
duplicate terminal waypoint. Its complete targeted suite passed **179 tests**.
RoboCasa then arrived within 13.4 mm / 0.0296 rad, but IK still reported 26.3 mm
error; Molmo again stopped in navigation. Neither performed a physical pickup.

Replaying RoboCasa's exact archived final pose exposed mixed-unit IK weighting.
Revision `25925b37` normalizes position and orientation residuals by their existing
tolerances. The same target now solves in 11 iterations at 5.7 mm / 0.0977 rad,
versus exhaustion at 160 iterations / 26.3 mm before. Acceptance limits remain
10 mm / 0.1 rad. A regression tests this underactuated orientation-slack case.

Revision `d65b72ac` also requires measured time as well as wall time to advance
before declaring a navigation stall; stale-data and command wall deadlines stay
active. Fifty-four focused motion/arrival tests and 31 command runtime/retry/tracker
tests passed. Navigation stage events now retain controller-result details.
Both changes are frozen in `/tmp/emet-physical-tamp-20260926-r11`; fresh physical
trials are pending. The final full registry must use the final candidate revision,
not silently reuse earlier pre-repair control scores.


## September 27 controller and fixture diagnostics

R17 RoboCasa failed fresh arrival IK at 15.7 mm position / 0.066 rad orientation;
its measured base arrival was 12.0 mm / 0.0149 rad from the planned endpoint.
R17 Molmo timed out at 603 s during approach. Neither trial picked or placed.
Independent scoring found no forbidden actuation or audited robot collision.

Revision `8c3b91b6` replaces interpolated-yaw waypoints with collision-validated
turn/drive/turn routes, keeping translations at most 0.2 m and checking the
travel heading even when it lies outside the endpoint-yaw interval. The focused
navigation/evaluation suite passed 25 tests. Physical r19 jobs
`20260927_074239_7a500c` (RoboCasa) and `20260927_074239_3ed41a` (Molmo) ran
serially. RoboCasa r19 finished in 114.5 s but failed fresh arrival IK at
23.7 mm / 0.0588 rad, with no pickup or placement. Molmo r19 timed out in placement search without executing pickup.

The exact missing scenes 12–21 and their object dependencies were installed on
September 27. The first installer process crashed after scene 14; the remaining
assets completed with safe CPU affinity. A separate 50-row recheck, frozen at
`b48b8898`, retains the original case definitions and runs under job
`20260927_075453_e06bcd`, root `/tmp/tamp-fixture-recheck-20260927-r20`.
It finished with 50/50 scene-resolution errors. Installed FloorPlans 13–22
belong to the validation split in MolmoSpaces, so the inherited training split
still could not resolve them. The original 50 fixture errors are retained. The first
submission `20260927_074941_39ce43` was canceled before execution to place physical
validation first; the replacement explicitly waits for both r21 jobs.

A subsequent bounded grasp-candidate filter checks IK at 16 base-pose samples
(eight directions at the existing 20 mm arrival radius, each at ±0.03 rad yaw).
It retains the original IK tolerances and 5 mm joint reserves. Exact r17 replay
rejects the selected candidate at a sampled arrival with 13.7 mm position error.
This is sampled candidate screening, not a continuous robustness proof;
execution still requires fresh measured-state IK and collision-checked paths.
The combined targeted suite now passes **236 tests**. Candidate `0425e4aa` is
frozen in r21, with jobs `20260927_075238_66924f` (RoboCasa) and
`20260927_075238_2b2a3e` (Molmo). Both returned `no_plan_within_budget`
before execution (65.2 s RoboCasa, 100.9 s Molmo).

## Next candidate after controller-aware replay

The archived Molmo transport witness contains 132 pose samples. Required travel
turns collide with the island when those samples are executed by a differential
drive. Controller-aware RRT steering, followed by compression of collinear drives
and in-place turns, found an 18-waypoint route in 1.83 s under the same 10 s / 400
iteration budget. A separate full resweep passed with the original collision and
0.22 m clearance checks. Measured-state steering is also checked before each
command; this replay remains an offline result.

R21’s 20 mm / 0.03 rad arrival envelope rejected every grasp candidate. A closer
RoboCasa candidate passes the same IK checks at 10 mm / 0.015 rad. The next
candidate uses an explicit `manipulation` navigation policy enforcing those
stricter measured bounds; controller targets are 5 mm / 0.0075 rad with XY
hysteresis out to 10 mm. The original precision policy remains available. No arm
reach, joint limit, collision, or physical task-scoring criterion was relaxed.

The registry generator now records `scene_split: val` for the 50 affected rows.
A comparison confirmed this is their only definition change; the other 150 rows
are identical. Exact FloorPlan13–22 files and dependencies are installed and all
ten resolve in preflight. A fresh 50-row recheck is required; failed attempts are
not overwritten or treated as successes.

The combined targeted suite passes **244 tests** for this candidate. Source
`fb817dcc` is frozen in r22. Physical jobs are `20260927_081943_2853a1`
(RoboCasa) and `20260927_081943_d2934c` (Molmo). The corrected 50-row registry
job `20260927_082035_9a7218` explicitly waits for both; output is
`/tmp/tamp-fixture-recheck-20260927-r22`. The physical pair is terminal; the
corrected registry recheck is running.

R22 RoboCasa failed approach after 426.6 s with `navigation stalled`; bounded
replanning could not satisfy the tighter policy during final turns. Molmo r22 found a complete witness at 230.2 s and executed successful
approach segments, then timed out at 602.3 s before pickup. The next candidate retains the controller-aware route search
but returns the physical driver to the original 20 mm / 0.03 rad precision policy.

Instead of requiring one fixed grasp frame at every possible arrival, the driver
now declares three yaw alternatives (0, −0.08, +0.08 rad). Offline replay of the
closer RoboCasa candidate passes all 16 original arrival samples using those
frames. The generic executor only tries caller-provided alternatives and validates
all three grasp/lift paths before any gripper command; every attempt and selected
world-frame target is logged. Individual pose IK tolerances, joint reserves,
collision checks, and physical scoring remain unchanged. The combined targeted
suite passes **246 tests**. The tighter policy remains opt-in and its failed live
result is retained; it is not the current physical driver's default.

The grasp-alternative candidate is frozen at `1b3f57bf` (r23). RoboCasa job
`20260927_083457_d7d269` retains the 600 s budget. Molmo job
`20260927_083844_3b2889`, root `/tmp/physical-tamp-molmo-20260927-r23-long`, uses
an explicitly declared 1800 s wall budget: r22 measured approximately 64–68 s
per 17 cm approach segment, leaving insufficient time for a complete task in
600 s. Per-command timeouts and bounded retries remain unchanged. The original
unstarted Molmo r23 job `20260927_083457_1cc47f` was canceled before execution;
it contributes no trial. This budget change is a diagnostic condition and must
not be described as a matched-budget improvement over earlier trials.
