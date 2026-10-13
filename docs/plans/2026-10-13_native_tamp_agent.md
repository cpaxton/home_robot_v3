# Native TAMP to exploration acceptance

The finish line is an agent discovering a task, planning it, executing with normal
robot controllers, and reporting measured results through the existing versioned
JSON tools. Native contact and observed perception are separate acceptance gates.
The earlier 6/6 scripted Stretch contact result does not establish either public
tool execution or exploration. Keep its denominator separate.

## Gate 1: stationary native public-tool execution

Controlled positives now pass at `d1d451c5`: 4/4 live ZMQ transfers and 6/6
separate CPU transfers, with both shapes represented. The original cluttered
scene refuses before actuation. See the [execution report](../experiments/tamp_native_agent_20261013.md)
for the distinct denominators and remaining scope; this does not complete Gates
2 or 3.

Implement `scene_tasks -> plan_pick_place -> execute_pick_place_plan` for the
packaged Stretch table scene with explicit privileged simulator inputs. Reuse the
physical executor, collision checker, and ordinary arm/gripper controllers.
Planning must not actuate. Execution must bind to the physics model and server
identity, reject stale captures or changed scenes, and consume handles once.
No latch, object teleport, direct joint setting, or native-articulation fallback.

Acceptance requires fresh live trials with sustained bilateral hold, transfer,
release, and settled support; no forbidden actuation in the audit. Repeat with
both object shapes, an obstructed target, stale state, and failed grasp. Preserve
failed trials and separate infrastructure, admission, planning, controller,
retention, and placement failures. A single passing smoke is integration evidence,
not a robustness rate. Record source SHA and exact command for every trial.

## Gate 2: agent selection and composability

Run the same public tools from `run agent` with an LLM selecting the task and
handling structured failure responses. Validate two consecutive pick/place tasks,
rediscovery after scene changes, and conservative handling of partial execution.
No automatic retry of a grasp after a partial placement. Add native navigation
only with independently checked base paths and measured arrival. Preserve the
stationary regression cases as controls.

## Gate 3: full exploration and observed inputs

Start without privileged task metadata. Explore, update the map, discover an
object and support from observations, then plan and execute using timestamped
object estimates and observed obstacle/payload geometry. Reject missing or stale
observations; do not silently fill gaps with simulator truth. Use GT only for an
independent evaluator and explicitly labeled debugging traces.

Measure discovery recall, valid task admission, planning success, physical task
completion, collision/retention failures, and wall time separately. Fix an
observed test set before tuning; report paired regressions against frozen seeds
and a held-out set. Add Sourccey only after the Stretch route passes and its own
native actuator/contact checks pass. Hardware readiness additionally requires
calibrated sensing, transforms, controller limits, and hardware validation.

## Merge boundaries

1. Existing contact-baseline evidence (#207) stays separate from implementation.
2. Executor settling and failure attribution (#208): independent measured-motion
   fixes; optional endpoint settling defaults off for existing callers.
3. Native public-tool integration: experimental privileged Stretch adapter and
   its live evidence, with observed mode still rejecting unavailable providers.
4. Observed exploration integration: perception providers and discovery/geometry
   wiring, with independent GT scoring and no claim of hardware acceptance.

Each implementation PR needs passing targeted regressions and concrete live
results. Do not promote a default preset or merge on unit tests alone.

## Integration follow-up: CHAT batches and composition

The next branch, `feat/tamp-native-composition`, fixes JSON outcome accounting
and prevents a failed TAMP call from executing the remainder of a precomputed
CHAT batch. At `9b44dd7f`, four scripted CHAT-dispatched CPU episodes each complete
two consecutive transfers without resetting physics. This is a repeated-object
integration control, not LLM or multi-object task acceptance.

Two-object fixtures still lack a validated reference sequence. On the
depth-separated fixture, trying the cylinder first also fails pregrasp planning;
do not blame only the earlier box placement or score this as an agent failure.
The next planner investigation needs explicit source/target regions and retained
candidate rejection evidence, then a verified full sequence before LLM scoring.
See [the composition report](../experiments/tamp_native_composition_20261013.md).
