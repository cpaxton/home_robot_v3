# Mars agent/controller acceptance and loaded lift diagnosis

The agent can issue a delivery through the public `scene_tasks`,
`plan_pick_place`, and `execute_pick_place_plan` tools. A real remote agent
completes one simulator delivery with the reference backend, including release,
independent physical acceptance, and a further two-second quiet check. The learned
backend is wired into the same dispatch, but **does not yet complete the task**.
This is a prototype integration, not a qualified learned manipulation controller.
The final-source paired comparison completes **3/3 reference deliveries and 0/3
learned deliveries** from identical initial physics. The first fixture uses the
real agent for both backends; the other two use scripted public-tool selection.

The agent is the existing `Qwen/Qwen3-VL-8B-Instruct` service at
`http://192.168.1.55:8000/v1`, with an explicit dummy API credential. No local GPU
or hardware is used. Prompts use simulator ground truth; physics pauses while
the model selects tools. Server weight revision is unverified and sampling uses
server defaults. This does not establish observed-agent or real-time performance.

## Acceptance case and completed training

`configs/benchmarks/mars_agent_controller_v1.json` adds an explicitly separate
single-delivery development program on three existing frozen fixtures. Selection
uses `left:20g:0`; that case is development data. The original multi-transfer
programs, placement regions, hand goals and verification predicates are intact.
No reset, teleport, attachment, fault clearing, or reference fallback is allowed
between discovery, planning, grasp, carry, placement, and release.

The matched v12 runs both **completed 16,384 PPO steps**, followed by their common
evaluation. Adding fixed-orientation translation resets did not improve task
execution: source, control and treatment each succeed on 1/8 translation cases
and 0/3 original public programs. The teacher reaches hand accuracy on 8/8
translation cases but retains final settling on 5/8. The control's nearby
position error is approximately 0.18–0.19 mm; its public pregrasp error remains
4.37 mm / 0.024 rad. Accurate local probes alone are insufficient evidence.
Complete training, evaluation and failures are archived in
`paper/data/whole_body_control/pose_translation_{training,validation}_v12.*`.

## Fixed-checkpoint diagnostic

The v12 control is the source for an explicit, offline checkpoint transformation.
It scales only learned arm responses to current/preview hand-pose errors.
Base and gripper heads, measured-velocity feedback, native output limits and
zero-error equilibrium are unchanged. This is **zero additional PPO steps**;
it is not learned improvement or a teacher residual. Every factor has its own
manifest, weight checksum, source provenance, fresh optimizer and admission
evidence. The source checkpoint is unchanged. Mars and Stretch layout tests
cover the transform; physical integration here is Mars only.

| Variant, same single-delivery development case | Outcome before contact-aware lift-off |
| --- | --- |
| Reference + real agent | Complete delivery and quiet verification |
| Source + real agent / 1x diagnostic | Pregrasp timeout: 4.38 mm / 0.0240 rad |
| 1.5x | Pregrasp timeout: 3.17 mm / 0.0140 rad |
| 2x | Pregrasp completes; wheel unloading during grasp |
| 3x | Pregrasp and grasp complete; lift-start collision misreported as IK failure |
| 4x | Pregrasp and grasp complete; lift-start collision |

These are sequential development diagnostics, not independent validation or a
factor selected for deployment. No factor passes the task. The unloading stop
is retained at the original 0.1 mm limit, even though the 2x excess is small.

## Executor fixes and contact semantics

A valid Cartesian endpoint need not reproduce the planner's particular joint
solution. If measured joints differ from the witness, execution can reconnect
to its endpoint through the ordinary collision planner **only if private FK at
the measured base still meets the original world-pose tolerance**. Otherwise it
replans the unchanged world pose. The first naive bridge was rejected at a
7.6 mm endpoint mismatch and is preserved as a failed experiment. Planning IK
still serves geometric admission; the actor consumes pose history/preview and
emits bounded velocities, without a runtime IK action loop.

Lift planning had five successful IK seeds followed by one failed seed. Every
solved seed was rejected because its *starting configuration* collided; returning
only the last error concealed that fact. Per-seed residuals and path rejections
are now recorded, and a later IK failure cannot overwrite a solved-pose path
failure. Regression tests preserve measured state and check both replan routes.

The collision was the **payload against its source table**, not a fingertip:
measured compression was about 1.09 mm, just beyond the generic 1 mm penetration
threshold. The opt-in Cartesian lift stage now has an explicit shallow support
departure contract. It admits only existing held-payload/static-support geometry
pairs exceeding the ordinary tolerance but with initial compression at most
**2 mm**, an upward-facing support normal,
and the original support height. Sampled compression cannot exceed the measured
initial depth, apart from 0.1 micrometre numerical slack. Permission clears even
on failure. New contacts, deeper compression, side impacts, moving supports,
finger/robot–table collisions and later placement retain the ordinary checker.

This is a deliberate contact-model change for separating an already supported
object, **not a statement that the old collision gate passed**. Generic penetration
tolerance, force monitoring, retention, static support, wheel unloading, base
drift, tracking tube and final pose requirements remain unchanged. Historical
reports do not qualify this new behavior. The independent runtime monitors still
run every cycle; sampled contact checking is not a continuous dynamics proof.

With departure enabled, 3x begins lifting but exceeds the 0.1 rad path tube
(0.10039 rad, position 8.07 mm). The 4x variant begins lifting but trips base drift.
Both stop; neither completes a transfer. Loaded pose tracking and base stability,
rather than public-tool wiring or an inability to solve that lift pose, are the
next blocker. Raising gains further is not supported by these results.

The final-source comparison reproduces that lift-orientation stop on both 20 g
fixtures (543 actor calls each); the 50 g fixture stops on base drift (537 calls).
All three measured stops are confirmed. The real agent respects the partial
result and stops instead of retrying a held-object failure. Reference execution
passes all three, so these particular deliveries have a demonstrated physical
baseline. Their repetition and small development population do not qualify
general task success. Full evidence is `agent_controller_v1/final_comparison.*`.

## Debugging and next gate

Opt-in observation traces now contain both the physics state **before** the
actor input and the state after its control step. Recording stays off by default.
These are diagnostic snapshots, not complete resumable MuJoCo saves.
All screens, the rejected naive bridge, corrected error reports, loaded failures,
model/source fingerprints and portable checkpoint evidence are retained under
`paper/data/whole_body_control/agent_controller_v1/`; weights remain local.
Measured plots show the
[reference delivery](../../paper/data/whole_body_control/agent_controller_v1/reference_agent_delivery.png)
and [learned lift stop](../../paper/data/whole_body_control/agent_controller_v1/learned_agent_lift.png).
Endpoint requirement lines apply at endpoints, not throughout a moving path;
the base plots show stage-goal errors rather than a full 6D body-drift check.

Next training should sample loaded grasp-to-lift transitions and terminal holds
from admitted public-tool paths as a bounded subset of the existing tracking
distribution. Keep ordinary base/hand/combined goals, retargets and disturbances;
do not turn the entire policy into this one delivery. Evaluate every candidate
first against this integrated agent case, then all three unchanged multi-transfer
cases and the paired general tests. Report first failing stage, pose/velocity
error, compression, force, jerk, retention and unloading. A failed reference
trajectory is not an infeasibility label. No Herman rollout or automatic
checkpoint promotion is authorized by these simulator results.

The final code passes **784 fresh source-bound regression tests**, with verified
XML/receipt `agent-controller-v2-regressions` under `paper/data/whole_body_control/`.
Relevant Ruff checks, core runtime parity and the paper build pass. All six
portable checkpoint-evidence wrappers verify against current model admission.
Offline lock validation passes with `uv lock --check --offline
--no-build-isolation`, using existing build tools/cache and the exact pinned SAM2
submodule initialized locally; no dependency sync or environment install occurs.
An empty-cache default offline check cannot resolve SAM2's build dependencies.

The subsequent [supported-lift experiment](mars_grasp_lift_20261027.md) implements
that bounded subtype and completes matched training, independent teacher replay
and all public programs. Coverage alone does not improve success. An isolated
objective correction improves component tracking but still fails strict settling;
this earlier report and its source-bound results remain historical evidence.
