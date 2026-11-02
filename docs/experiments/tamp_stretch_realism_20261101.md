# Live Stretch agent placement pilot — November 1

The production `run agent` CLI completes **2/4 tasks** in this fixed development
pilot. Both familiar controls pass; shifting both initial object X positions by
+15 mm causes both tasks to stop during the second object's lift. The failure
is `payload_not_retained`, not destination search or malformed agent calls.
The independent scorer verifies **5/8 final object goals**. The tools complete
**7/10 planned transfers**; the last transfer of the failed swap is unattempted.

| Case | Initial X shift | Complete task | Completed / planned transfers | Final goals | Audited commands |
| --- | --- | --- | --- | --- | --- |
| Two-goal control | 0 mm | pass | 2/2 | 2/2 | 477 |
| Buffered-swap control | 0 mm | pass | 3/3 | 2/2 | 750 |
| Two-goal shift | +15 mm | fail: cube lift | 1/2 | 1/2 | 450 |
| Buffered-swap shift | +15 mm | fail: cube lift | 1/3 | 0/2 | 476 |

All four traces have valid JSON tool contracts. Both failed executions produce
partial outcomes, preserve the completed first transfer and stop further
dispatch. Failure-stop checks remain unassessed in the successful controls.
The audit records **2,153 commands and zero forbidden commands**. There are no
infrastructure failures in the four agent trials. CLI exit zero after a correct
failure report does not count as task success.

## Protocol and provenance

Source `d4a05983`; branch `feat/tamp-sim-realism`, stacked on #226. The
[case definitions](../../configs/eval/tamp_stretch_realism_pilot.json) reconstruct
the population recorded in the raw batch manifest before launch. Initial object
positions, instructions, final regions and seeds are explicit JSON. Each shifted
case retains its control's goals, tolerances and budgets. Scene-file hashes and
task definitions were checked against the frozen inputs after execution.

These are stationary simulated Stretch trials with native MuJoCo contact, live
asynchronous ZMQ execution, privileged geometry/state and no cameras. Caliban
serves `Qwen/Qwen3-VL-8B-Instruct`; no local VLM or GPU is used. Hardware Stretch
trials are excluded. The agent chooses discovery, `plan_task` and
`execute_task_plan`; task search chooses operation order and the swap buffer.
The evaluator does not provide an action sequence. Each episode uses four
inference calls, including its final report, for **16 calls** in total.
Recorded inference time is about 38–45 seconds per task. The tool-round limit is
12 and the CLI timeout is 900 seconds. Remote checkpoint/sampling are not
independently pinned.

Both controls preserve the successful outcomes of the
[October 22 integration cases](tamp_main_integration_20261022.md). One repeat per
configuration does not establish a reliability estimate or rule out regressions.
The two shifts are development perturbations, not held-out or reference-certified
benchmark instances. No global collision-integrity, observed grounding,
navigation, exploration or hardware result is claimed. The live model signature
is checked against the packaged model, but these cases lack a complete initial
state replay certificate. Earlier failed cases remain unchanged.

The [compact result manifest](tamp_stretch_realism_20261101.json) retains all four
outcomes, measured final bounds, planning costs, actual CLI integration checks,
job IDs and raw-summary/protocol/audit/CHAT hashes. Raw records remain under
`.tamp-runs/stretch-realism-20261101/`; dense traces and simulator models are not
copied into the repository.

## Failure localization

Both shifted agents correctly obtain a complete task skeleton, place the red
cylinder and refine the next cube motion. Both cube executions fail at **lift**
before the placement segment. The native executor checks the measured
object-to-gripper transform against its grasp reference: translation may change
by at most 2 cm and rotation by at most 0.1 rad. The original agent log records
the failure phase/code, but does not retain the rejecting residual; it cannot
establish whether translation, rotation or both crossed the bound.

The shifted cube's execution refinement uses **52 IK calls**, versus **14** in
each control. Whole-provider planning consumes 143/436 IK calls in the controls
and 227/385 in the two shifted cases, within the unchanged 2,048-call allowance.
Rejected grasp/IK candidates are planning telemetry, not executed collisions.
This pilot exposes sensitivity to initial placement, not a need for an extra
buffer or more task-search depth.

Separate post-pilot scripted public-tool replays in **in-process stepped
physics** complete both two-transfer layouts. They omit the actual CLI's
startup/posture and asynchronous control timing, so they do not explain away the
live failures or replace either failed agent outcome. A first scripted live
task-tool replay stops at `search_exhausted` before any transfer. Its initial
base position differs from the captured agent state by about 3 mm, and its
simulation time is 3.51 s versus 9.35 s in the agent snapshot. That comparison
does not isolate timing as the cause. These diagnostic attempts remain in the
JSON manifest separately from the four-case agent denominator.

A separately labelled live replay adds six simulated seconds of startup
settling, then uses the same task tools without LLM calls. It completes the first
transfer and reproduces the cube-lift rejection. The measured relative
translation is **1.315 mm**, within the 20 mm bound; relative rotation is
**0.100849 rad (5.78 degrees)**, just beyond the unchanged 0.1 rad limit.
At that snapshot the cube is 21.33 mm above its initial center height. A private
MuJoCo contact reconstruction finds contact with both fingertips and no table
contact. Contact poses and passive finger angles here are explicitly simulator
debug truth, not hardware measurements.

This diagnostic establishes a rejection of the **rigid payload pose** during
contact, rather than evidence that the cube dropped. It does not establish
sustained retention or prove identical residuals in the original agent trials.
The existing collision path assumes a fixed object-to-gripper transform, so
merely loosening the rotation limit would invalidate its payload-volume claim.
Investigate contact/grasp stability and measured payload pose bounds together.
Keep original agent outcomes unchanged. The complete live diagnostic traces,
including the earlier planning rejection, are preserved separately.

## Validation and next gate

The focused CHAT failure-boundary, independent goal-scoring and live-snapshot
suite passes **21 tests**, with zero failures/errors/skips. All four task configs
pass the native schema validator. Input hash checks confirm the intended initial
placements and unchanged goals/instructions within each matched family.

Next compare selected grasp geometry, actuator tracking and object/contact
evolution through closure and the first loaded lift. A fix must either stabilize
the grasp or revalidate motion against measured/bounded payload rotation; an
unaccounted rigid-transform change must still stop execution. Keep follow-up controls separate from these
original failures. Qualify the exact-geometry native path across placements
before promoting the same cases to assisted RGBD and observed inputs. Robot
self-occlusion, historical free-space evidence and phase-specific contact/source
qualification remain independent observed-execution gates. There is no TAMP
training loop or hardware contact-sensor claim here.

The paper appendix includes both this failed-placement pilot and the separate
Panda controller-reference results. Its build succeeds at 54 pages, with no
undefined references or citations. No production motion behavior, contact
parameters or retention thresholds changed in this experiment update.
