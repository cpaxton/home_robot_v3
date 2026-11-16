# Multi-step TAMP task plans

`plan_task` and `execute_task_plan` are CHAT tools using the existing
[schema-version-1 JSON envelope](tamp.md). They require a qualified provider;
registration alone does not enable multi-step execution on a connected robot.
The production loop omits these tools when no session is bound, preserving
existing agent conditions. Direct registry calls without a provider return
`task_provider_unavailable`, with no physical command.

`plan_task(instruction, strategy="beam")` accepts the user's complete final goal
and returns an opaque `plan_ref`, number of steps, strategy and
`execution_started: false`. Currently implemented strategies are `greedy`,
`beam` and `mcts` (the existing UCT implementation). Other methods explicitly
return `strategy_unavailable`; they are not aliases. Planning calls a shared
operator/feasibility provider and must not mutate live simulator state.

`execute_task_plan(plan_ref)` consumes the handle once, including on rejection.
It rejects changed model, robot profile, boot, reset, revision or initial
predicates. After each native skill it reads measured predicates, checks the
operator's effects and protected goals, and revalidates before the next skill.
A command returning success is insufficient. Any failed/unknown result stops the
plan and the production CHAT dispatch batch. There is no automatic retry.

Successful execution reports `completed_steps`, `execution_started` and
`goal_verified`. Failures retain completed-step counts and return `partial` if
execution may have begun. Public errors never expose provider exception strings
or private body names. Inspect the scene before deciding how to recover.

Python integration: construct `TaskSession(provider, budget=BudgetSpec(...))`
and pass it as `tamp_task_session` to `run_agent_with_robot`. The provider owns
fresh grounded `TaskState`, instruction-to-`GoalSpec` compilation, qualified
operators, private-state feasibility and measured native execution. It must
bind the connected robot and correct input mode; this seam does not waive any
native/observed qualification requirement. Session locking serializes its own
calls, not unrelated external controllers; provider execution must atomically
validate its planning snapshot before commanding hardware.

Benchmark support and remaining qualification gates:
[MuJoCo implementation plan](../plans/2026-10-22_mujoco_tamp_benchmarks.md).

## Privileged native Stretch integration

`run agent --native-task-config FILE` opts into stationary region-transfer
planning. It requires online LLM CHAT, `--state-only-sim`, physical manipulation
mode and privileged inputs. [Example configuration](../../configs/eval/native_stretch_transfer_debug.json)
declares public roles, private body bindings, one support and disjoint XY regions.
Its diagnostic `reset_id` is not an upstream reset certificate. The configuration
must match the connected scene; it is not portable table calibration.

Discovery reports public object and region names, measured locations, free
regions and supported instruction forms. Body bindings stay private. The bounded
compiler accepts `swap OBJECT and OBJECT` or semicolon-separated clauses such as
`move OBJECT to REGION`. Unknown roles, conflicting goals, overlapping regions,
uncertain occupancy and unsettled/held objects are refused. It does not claim
unrestricted natural-language grounding or observed-input support.

Plans report `planning_scope: task_skeleton`, `motion_refinement: per_step`.
Search checks occupancy/footprint fit and probes native arm/payload planning for
candidate first moves from the measured state. Infeasible first moves are pruned
before selecting ordering. Hypothetical later states remain symbolic: a returned
skeleton is not a full-episode collision-free trajectory certificate. Every
transfer is refined again from fresh measured geometry before native actuation.
Motion probes use private data, discard unused motor-plan handles, share a
bounded planning-IK allowance and record semantic rejection counts. Execution-time
replanning and outer LLM calls are not yet included in that allowance; no complete
matched-budget baseline claim follows from it.

`TaskExecutionResult(success, execution_started, failure_code)` distinguishes a
known precommand failure from possible physical progress. A failed command or a
failed later step preserves partial progress and stops execution. Legacy boolean
receipts cannot establish no actuation. Measured effects, support, containment,
stillness and final goals remain mandatory regardless of command receipts.


Failures also identify `planned_steps`, `failed_step_1based` (or null before a
step/at final verification), and `failed_phase`. For example, one completed step
followed by step-2 preflight rejection is partial progress, not a failed first
transfer. Debug mode can include private simulator state in JSON logs for offline
replay; public tool responses and public motion evidence never include it.
See [measured integration cases and captured-state controls](../experiments/tamp_native_task_provider_20261022.md).

## Provider failures and budgets

Discovery and planning use the same public failure translation as execution.
A changed scene during planning returns `stale_task_plan` with `recovery: replan`
and stores no plan. Typed unavailable observations return `stale_observation`
for known stale measurements, otherwise `observation_unavailable`. These errors
remain failures and do not authorize an automatic execution retry.

Execution preserves allowlisted provider reasons, including
`task_object_not_settled_on_support`, `ambiguous_region_occupancy`,
`native_execution_unavailable` and `object_not_lifted`, through both exceptions
and typed receipts. Unknown provider errors return `task_provider_failed` without
exception text. Partial-progress counts and one-shot consumption still apply.

`TaskBudget` is currently tested infrastructure without a production caller.
`BudgetSpec` bounds task search, and native planning probes separately consume
the provider's IK allowance. Neither constitutes a shared whole-agent ledger;
CHAT calls and execution replanning must be connected before matched-budget
whole-agent comparisons are claimed.

## Preserved objects and execution preflight

Explicit move goals protect the current `at` predicates of every unmentioned
object. For example, `move alpha to right` cannot silently relocate beta from
right into a buffer; that request is infeasible under the preserve-beta
constraint. Explicitly include beta's desired destination, or request a swap,
when moving it is authorized. Swap goals likewise protect other objects.

The region provider converts errors from its read-only snapshot, feasibility
and operator-selection preflight into typed no-actuation receipts. The transfer
callback is outside that exception boundary: unknown exceptions there still
mean execution may have started. Earlier measured progress is preserved in either
case. This distinction does not infer no actuation from arbitrary provider errors.

`--native-task-config` is incompatible with `--eqa-eval`; it cannot silently
configure a different orchestrator. Both remote OpenAI text and vision clients
keep robot telemetry live during inference. Local inference retains its existing
stream-pause guard.

Native execution refinement also has a planning-only exception boundary: timeout,
malformed-result and invalid-handle errors before `adapter.execute` return
no-actuation receipts. Typed failed receipts with `execution_started: false`
report `failed_phase: preflight`, even when an earlier step made physical
progress. Exceptions from the actual adapter executor remain possible physical
execution and are never reclassified by the planning-only catch.

## Native motion policy

`agent.native_tamp_policy` selects the shared native adapter implementation for
both primitive tools and a bound multi-step native task provider. Default
`legacy` preserves the existing motion policy. The opt-in bounded release profile
is Stretch-specific. Selecting it on the Mars native adapter returns
`native_policy_unsupported` before commands and invalidates the cached adapter;
switching back cannot resurrect its old plan handles. Mars keeps its own grasp,
approach and transport hooks under the legacy native profile.

Explicit
`--set agent.native_tamp_policy=bounded_release` enables the finer grasp yaw set,
a checked loaded-payload handoff, and bounded measured-state release search.
Use it with `agent.manip_mode=physical` and `agent.tamp_inputs=privileged` in
simulation. It does not enable observed execution or imply hardware sensors.

The handoff checks the original retention bound through an additional stable
bilateral-contact interval before binding the measured transform once. Release
checks arm/payload motion, nominal and lower opening sweeps, support/drop limits,
and measured supported stillness before retreat. Numerical collision/retention
limits and independent final-goal checks are unchanged. The 40 mm maximum
raise/drop policy is explicit and remains opt-in while broader qualification is
in progress. A failed or partial result still stops dispatch; do not replay it.

Native discovery reports `native_motion_policy`. A primitive adapter cache is
invalidated when the configured policy changes, and multi-step scene bindings
include the policy identity. Task JSON remains the same versioned goal/scene
contract; motion-policy options are not new LLM tools or object-specific features.
See [qualification protocol](../plans/2026-11-13_native_tamp_qualification.md).

The November 13 follow-up keeps existing grasp candidates first and adds a
collision-checked vertical pregrasp only after they fail, sharing the original
IK cap. The native policy permits up to 12 simulated seconds for measured
supported settling before retreat, matching the existing native verifier budget;
this changes the wait budget, not the speed, collision or retention criteria.
Invalid policy/mode/input combinations are rejected before starting the client.

The bounded policy reserves 0.5 mm inside each telescoping arm segment's
joint limits when solving IK targets. The coupled extension therefore reserves
2 mm overall. This leaves room for the small measured limit overrun seen after
opening; it is not a certified tracking-error bound. Model limits, measured-start
validation, collision checks and acceptance thresholds remain unchanged. An
already out-of-range measured start still rejects; no live state is clamped.
This reserve applies to IK targets, not every path waypoint: the path planner
uses the original model limits, so a valid measured start within the reserve
band and the initial path segment can remain outside the target reserve.

Failure receipts can include a finite-vocabulary `data.verification` witness:

```json
{"substage":"post_lift_hold","reason":"payload_not_retained"}
```

Native substages distinguish `lift_motion`, `post_lift_hold`,
`loaded_payload_handoff`, `loaded_motion` and `post_release_support`. The primitive tool and
multi-step receipt preserve the same witness without private object IDs or
provider exception text. Unknown/malformed witness fields are discarded at the
native-provider boundary. Missing evidence remains absent; it is not inferred
from the generic failure code or reused from an earlier step/handle.

Original payload retention is checked at every sampled post-lift hold update,
as well as during motion and the loaded handoff. A sampled violation rejects
immediately and cannot be renewed by a later good sample. This strengthens
coverage of the existing bound; it does not relax its 20 mm / 0.1 rad limits or
establish detection between samples.

The bounded development policy tests a contact-relative **position target**.
After the normal full-close command settles, it targets the midpoint between
the measured motor position and the original close preset, then requires a
second settling interval within the same total closing budget. This retains
half the position error without using pad pose or force feedback. Lift,
retention and bilateral hold must still pass; motor settling alone is not
pickup success. The ordinary legacy command and observed/hardware defaults
are unchanged. No contact physics or sensor availability is changed.

The prior fixed loose-close pilot passed the swap but failed the familiar
control; it is not a general solution. Its results remain in
[the pilot report](../experiments/tamp_native_closure_pilot_20261114.md).
The [live contact-relative pilot](../experiments/tamp_native_contact_relative_pilot_20261114.md)
passes the two-goal controls but still fails swap retention. The diagnostic
`--gripper-close-preset default|loose` override disables the contact-relative
adjustment and archives the effective policy.

For an object described by one collision box, the bounded development policy
also ranks the existing grasp candidates by jaw alignment with the box's
measured world-space face normals. This lets a higher aligned grasp precede a
lower edge pinch. Compound and other shapes keep their prior order. No candidate
is removed or added, and the shared IK cap stays unchanged. This is a geometric
preference, not a contact/force certificate; execution still has to pass the
same physical checks.
The [geometric-order pilot](../experiments/tamp_native_face_alignment_pilot_20261114.md)
also retains the failed swap. At that revision, an executed base approach was
unqualified; the private base-offset IK probe is not task success. The separate
approach follow-up below records subsequent live development results.

### Experimental checked base approach

`agent.native_tamp_policy=bounded_approach` extends the opt-in bounded policy
with unloaded pre-grasp base positioning in the native simulated Stretch path.
The default remains `legacy`; base motion and loaded candidate refresh require
this explicit opt-in. A shared placement-frame correction described below also
applies to the other profiles. The same option is accepted by the native
evaluators and actual `run agent` configuration through shared policy validation.

The planner tries the current base and short forward/reverse alternatives at
40/80 mm along the measured drive axis, preferring the best box-face alignment
across those alternatives before stationary fallback grasps. All alternatives
share 256 IK calls and a 120-second planning deadline. The complete grasp,
placement and retreat must plan before an approach handle is stored.

Base sweeps include the whole robot at its measured arm posture and forbid
finger/object contact during transit. Samples are spaced at most 5 mm and
0.025 rad, including a sampled ±10 mm XY / ±0.05 rad tracking envelope. This
is a static sampled geometry check, not a continuous or dynamics certificate.
A colliding held-arm posture rejects; no unchecked tuck is inserted.

Execution rechecks current geometry, uses the ordinary world-frame base
controller, and requires measured arrival within 10 mm / 0.015 rad, 0.3 seconds
of quiet pose before confirmed navigation cancellation, then another 0.3 seconds
of quiet pose. Measured arm drift, corridor departure, stale state, changed
obstacle geometry or uncertain stops prevent grasp dispatch. Scene joint motion
triggers a new remaining sweep. Grasp planning restarts from actual arrival.
No loaded base transport or live root-pose writes are permitted by this path.

Results publish `base_motion=checked_unloaded_approach` and
`collision_scope=sampled_base_sweep_arm_and_payload`. Pre-command rejection has
`execution_started=false`; possible dispatch remains partial and consumes the
handle. Physical task success still requires the existing retention, release
and final-goal gates. The [November 16 development replay](../experiments/tamp_native_base_approach_20261116.md)
passes eight native tasks and four actual-agent tasks. These tuned layouts are
not held-out reliability, observed navigation/exploration or hardware support.

After verified pickup, this profile refreshes held-object bounds and free support
centers inside the original destination region. Initial and loaded placement use
one shared path-planning function and the same local support patches. The original
release proposal is retained when its measured whole footprint still fits the
original support/region, and its complete updated paths must certify it again.
Loaded refinement can try up to 16 centers
using the initial primitive's unused IK/time budget; its extra calls also debit
the task provider's episode budget. Controller execution time is excluded from
planning time. The original grasp reference is preserved and no command is sent
during this refinement. Exhaustion remains a partial held-object failure with
`verification.substage=loaded_placement_planning`. The execution receipt reports
the resulting world-space placement goal.

Payload-center offsets are expressed in the measured EE frame and transformed
through the desired placement rotation, rather than reusing a world-frame
offset when the target EE orientation differs. This coordinate-frame correction
is shared by all native profiles. All placement/release/effect
checks still apply. See the [initial base-approach pilot](../experiments/tamp_native_base_approach_initial_20261116.md)
for the retained third-transfer failure motivating this refinement.

The command client retains one exact boot/session/sequence stop witness after
a navigation cancellation is confirmed. Later arm commands may evict the server's
terminal receipt without losing that witness at client shutdown. A different
goal, server restart or unconfirmed stop cannot use it to release ownership.
