# Benchmark failure audit — 2026-10-29

This audit concerns the frozen-main refresh at `cd169673`, under
`~/runs/emet/benchmark-refresh-audited-20261029/`. Preserve that manifest and
ledger: diagnostic reruns and repairs are separate evidence, not replacement
scores. The research branch is `feat/harness-evidence-study`; main is untouched.

## DynaMem EQA: missing navigation sample

All five exceptions (q104, q21, q38, q41, q47) reach the same expression in
`controller/dynamem/eqa.py`: `sample_navigation(...)[-1]`. The sampler can
legitimately return `None` when it cannot find a reachable viewpoint. The caller
indexes that result instead of handling unavailable geometry.

Commit `6ae052fc` preserves an unspecified target heading when sampling fails,
letting the existing navigation path handle failure while retaining the model's
answer. It does not fabricate a heading, loosen clearance, or count a failed
motion as success. A focused reproduction failed before the repair; both the
missing-sample and valid-sample cases now pass. Together with the existing
DynaMem response/client tests, validation is 11 passing tests. The five live
episodes have not yet been rerun on this repair.

## Molmo RBY1: incompatible strict-physical protocol

All three attempted methods fail during their first mapping turn, before a
meaningful object-search comparison. The sweep enables
`EMET_PHYSICAL_EXECUTION=1`. The legacy RBY1 simulator bridge uses a kinematic
stationary-base hold, which the strict audit correctly forbids.

The actual physics-thread traceback is:

```
_sim_loop -> _step_physics_tick -> _step_controlled_physics_once
 -> _hold_stationary_base_if_idle -> _hold_stationary_base_freejoint_if_idle
 -> audit_physical_action
RuntimeError: forbidden_physical_actuation:stationary_base_pose_hold
```

The bridge remains connected after that worker dies. Subsequent navigation
commands therefore wait against a frozen simulation; the scan reports that its
view did not change. This is a simulator/protocol mismatch, not evidence of a
Qwen recognition or memory failure. The strict guard must remain enabled for
rows claiming physical actuation.

Required follow-up: reject unsupported execution modes before starting a live
episode, and propagate physics-worker failure to service health. Either qualify
an actuator-driven RBY1 controller, or declare a separate assisted-simulator row
with its actual base holding/velocity-drive assumptions. Turning off the guard
does not establish physical support. `passive_base_support` also writes pose
state and does not remove this incompatibility.

The temporary server stderr and audit were copied without alteration to
`~/runs/emet/benchmark-failure-audit-20261029/`; `preserved_logs.json` records
source paths and SHA-256 hashes.

## S0: long navigation failures followed by poor camera views

The first lazy-graph router image contains both the red cylinder and blue cube.
A later router image faces empty sky. The first investigate attempt reports
568.3 seconds of navigation and `aborted_waypoint_timeout`; the measured base
pose (XY/yaw) is approximately `[0.668, 0.079, 0.380]`. Its requested candidate
anchor XYZ was `[0.600, -0.300, 1.0]` (the third number is height, not yaw).
Another saved exploration receipt succeeds, so the
available evidence does not support saying physics stopped throughout this run.

Stretch scales motion waits using the reported simulator real-time factor.
Individual waits can therefore consume roughly 200 wall seconds even with a
30-second nominal navigation timeout. That amplifies the failure cost; it does
not explain the original tracking failure on its own.

A private-data endpoint collision check found no penetrating contact at the
home-posture simple and S0-adjacent candidate poses. This does not certify a
swept route or the live navigation posture. A separate no-VLM route diagnostic
completed under `s0-mujoco-nav-cause-20261029` (job `20261029_115949_4f646d`),
recording measured receipts, posture, actuator targets and RGB-D. All three
motions succeed: short forward, `[0.600, -0.300, -2.3]` (episode XY/yaw), and
return to the origin. Base-up stays above `0.99999` throughout the saved poses.
A read-only state sample during its return leg
reports a simulator real-time factor of `0.0259`; a ten-times wall-budget cap
still provides less simulator time than intended at that rate. The original
planner's exact failed waypoint remains unqualified. These direct low-level
goals are a control, not clearance-checked search or OVMM success. The adjacent
pose's saved image places the cylinder near the lower-right edge: arrival
alone does not establish useful camera framing.

There is also a concrete evidence handoff: failed navigation exits
`agentic/investigate.py` before arrival capture, target aiming and grounding.
At the failed arrival, the cylinder lies roughly behind the camera's base
heading. Subsequent single-view captures therefore assess an unrelated view.
This explains the later absence judgments without assuming the model cannot
recognize the initially visible cylinder. The original trace does not retain
all waypoint requests/receipts, so it cannot establish which intermediate goal
failed. Next isolate those commands and simulator-clock budgets, and preserve
the measured post-failure view for the model without claiming arrival.

## Native TAMP: release verifier and final scorer disagree

The live learned control transfers the first object, then stops correctly on
`placement_verify_failed`. The later independent scorer finds that object
supported, settled and inside its goal region. The second object is not moved.

These checks differ: the native release verifier additionally checks distance
from the planned center, expected center height, and a continuous two-second
stable interval. It uses simulator time but also a 30-second wall-clock cap.
The independent final scorer alone cannot establish that all those conditions
were satisfied during verification.

Job `20261029_115747_9057f0` reuses the original fixture and task commands through
the current production CHAT API, with read-only verification tracing. It does
not change the predicate, timeout, return value or exception. Its grasp check
passed after 1.002 simulator seconds / 20.17 wall seconds. The release check
failed after **30.21 wall seconds / only 1.35 simulator seconds**. At exit the
object contacts only the table, lies inside its region, has 5.25 mm XY error
and 0.48 mm height error, and has speed 0.096 mm/s and spin 0.0024 rad/s. It
cannot satisfy a continuous two-second interval before this wall cap, even
when placement geometry is good. This confirms a clock-budget failure.

Commit `898b31b7` reuses the existing bounded simulation-rate multiplier for
physical settling wall budgets. The simulator-time interval and all placement
predicates stay unchanged; wall waits still expire for a frozen clock. A real
native-verifier regression reproduces the rejection at 0.05x before the repair
and passes afterward. Existing physical-motion tests and explicit frozen-clock
bounds also pass. The live two-object task has not been rerun on the repair.

The diagnostic's additional post-task scorer wait hit the same kind of fixed
wall cap and the job exited nonzero after writing both verifier traces. Do not
count it as a successful task or invent a final summary. The diagnostic now
refreshes and labels its final score as an instantaneous snapshot, with
continuous stability reported separately in `verification.jsonl`. Artifacts:
`~/runs/emet/native-verification-cause-20261029/`.

## Focused repair restart

The authorized restart uses a new output root,
`~/runs/emet/benchmark-repair-check-20261029/`, and isolated source `5a41747d`
in `.worktrees/benchmark-repairs-20261029`: original `cd169673` plus just the
two repairs above. No unrelated CHAT or capability-suite changes are included.
`assignment.json` pins the original protocols, fixture and scoring inputs.

The live two-object TAMP control starts first, using the original CLI, remote
Qwen endpoint, fixture, goals and deliberate invalid-plan boundary. Its acceptance
still requires both objects correctly placed and stable, valid tool contracts,
proper failure stopping, and zero forbidden actuation. A successful first release
alone is not successful completion of the two-object task.

Then rerun exactly q104/q21/q38/q41/q47 with DynaMem, seed 0 and the original
model, exploration and answer budgets. The existing matrix driver now accepts
repeatable `--task-id` arguments, rejects unknown IDs, and freezes the selected
population in a new manifest. Completion without the exception and answer
correctness are reported separately. Keep the original failed rows; do not
combine the five selected retries into an unbiased overall accuracy estimate.

All heavy jobs remain serial on CPU 12 with single-thread libraries and the
managed experiment lock. The EQA coordinator waits for the TAMP supervisor and
yields to other GPU/RL work between cases. Retain the 20 GiB disk reserve; if it
is reached, remaining assignments stay pending. Molmo remains paused for its
execution-mode mismatch; S0 search needs its exact planner-command trace before
claiming a repaired integrated search result. This is not a full sweep restart.

### Restart results checked 2026-10-29

Live control `20261029_155400_ad3c3c` **passes** on the isolated repaired source:
Qwen calls scene discovery, task planning, and task execution; both transfers
complete with `goal_verified=true`. Independent final scoring finds both whole
object footprints inside their respective regions, supported and settled. All
533 audited commands are free of forbidden actuation. The deliberate invalid
plan is rejected, and the agent stops before the separately authorized task
turn. The CLI exits zero. This qualifies this one privileged stationary task,
not learned grounding, exploration, or general manipulation reliability.

The first EQA retry window completes q104/q21/q38 without exceptions; q38 is
correct and q104/q21 are wrong. Its coordinator then stops admission at the
20 GiB reserve, retaining q41/q47 as pending. This is a disk-capacity stop,
not another navigation exception or a completed five-question result.

An offline `uv cache prune` removes unreachable package-cache objects, with no
benchmark evidence, dataset, or model-cache deletion. Available disk rises to
about 28 GB. Resume job `20261029_175505_ce82c1` uses the same manifest/source
and skips the three completed entries; only q41/q47 remain to execute. The
launch and cleanup record is `eqa_resume_launch.json` in the repair output root.
Do not pool these failure-selected retries into an overall accuracy claim.
