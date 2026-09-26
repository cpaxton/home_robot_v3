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
job `20260926_110716_7c4a47` is queued with output
`/tmp/physical-tamp-robocasa-20260926-r6`.

The archived-route profile completed two routes in 1.035 seconds (798 validity
calls). A separate symbolic candidate profile took 0.058 seconds. Neither
reproduces the live route delays; these are diagnostic timings, not a speedup
claim or an execution witness. A moved-pose contact-count probe matched the full
position pass, but this alone is not broad collision-kernel validation.

The revision-5 protocol has completed Nori scene 0 with all four oracle tests
passing. The remaining protocol, registry, floor, and tool results are pending.
