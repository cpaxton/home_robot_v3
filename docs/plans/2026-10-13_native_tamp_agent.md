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

## October 13 review: finish the agent-facing milestone

Current stack: #207 records an independent scripted contact baseline; merge #208
(executor) before #209 (native public tools). Both implementation PRs are open and
non-draft as of this review. Documentation of the next gates must not be mistaken
for implemented or accepted behavior.

[TiPToP](https://tiptop-robot.github.io/implementation) supports the architectural
choice of grounded objects/goals, explicit geometric planning, and separate
execution. Its task search considers rearrangement sequences. Our local
pick/place tool interface is a useful execution boundary, but does not itself
provide that search. Keep task search replaceable: test our existing search before
choosing MCTS, another symbolic planner, or cuTAMP. Algorithm choice should follow
measured failures. TiPToP's fixed-view perception and open-loop trajectory
execution also leave a relevant research direction for dynamic memory and
measured recovery; we have not yet demonstrated that full integration.

### Atomic next implementation: native task composition

1. Freeze a minimal suite with two consecutive transfers, move-away-and-return,
   and blocker relocation followed by target placement. Include a feasible
   temporary support and verify the whole reference sequence with native dynamics
   before scoring an agent. Keep impossible/no-free-support cases as labeled
   refusal tests, not successful manipulation trials.
2. Run the same fixtures with a scripted public-tool sequence first. Re-read state
   and reacquire task handles after every move; verify the first placed object is
   still supported after subsequent moves. Never chain cached scene assumptions.
3. Expose/test prerequisites, expected effects, supported execution/input modes,
   state freshness, and recovery actions in tool schemas/descriptions. Extend the
   existing versioned JSON envelope where needed; do not introduce another prose
   protocol or object-name-specific branch. An unavailable capability must be
   distinguishable from an infeasible plan or uncertain executed command.
4. Make the task-search layer consider alternate grasps/approaches/placements and
   bounded blocker relocation. Test that a locally feasible grasp without a
   subsequent feasible placement does not prematurely terminate all search.
   Return budget exhaustion distinctly from a proven infeasible task.
5. Run `run agent` selection on the same fixtures. Test partial execution, stale
   handles, timeout with unknown outcome, and interruption between actions. Require
   inspection/rediscovery before replanning uncertain outcomes; no blind replay.

Acceptance: every frozen positive reference sequence must complete with measured
final predicates and a clean actuation audit; every injected fault must return
an honest outcome without continuing from an unverified state. Score LLM trials
separately, including failures, and record model/version/prompt, seeds, tool trace,
source SHA, geometry provenance, budgets, elapsed time, and final predicates.
This milestone can land before exploration; it must not enable observed mode by
silently substituting privileged state.

### Hill-climbing protocol

Keep the current box/cylinder cases as regression controls. Before tuning the
next suite, check in a JSON manifest of scene seeds, initial object transforms,
goal predicates, reference validation, input mode, and execution mode. Reserve a
held-out split; use matched fixtures and budgets for every comparison. Report:

- Full-goal completion over all admitted attempts, plus task progress per step.
- Construction/reference failures separately, fixed before agent scoring.
- Discovery/admission, geometric planning, tracking, retention, released support,
  and infrastructure failures separately; preserve partial/unknown outcomes.
- Collision/audit violations, recovery count, planning time and total wall time.

Compare scripted execution, task search, and LLM orchestration without pooling
results. Pair privileged and observed runs only after the observed provider is
wired. Report both whole-suite and per-task results so adding easier cases cannot
hide regressions. Archive videos with task/active-skill overlays and link them to
JSON traces; renderings alone are not acceptance evidence.

The next separate PR wires timestamped observed objects/supports and voxel
payload/obstacle geometry into the same contracts. Its acceptance starts from
exploration without privileged task handles. Hardware validation must replace
simulator contact/pose checks with supported, calibrated sensor evidence;
privileged contacts remain available only for explicit debugging and evaluation.
