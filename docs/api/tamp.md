# Agent-facing TAMP simulator API

Start with the [TAMP overview](../tamp.md) for the workflow and architecture, or
the [agent tool index](../agent_tools.md) for the complete tool inventory.
Main through #212 includes structured tools, collision-aware placement,
simulator-assisted articulation, observed/privileged input contracts, and the
stationary Stretch native-debug adapter for the packaged table scene. Observed
geometry providers remain unwired. See the [placement API](placement.md),
[articulation API](tamp_articulation.md), and
[latest live agent report](../experiments/tamp_native_live_agent_20261014.md) for
tested revisions and remaining gates; assisted and native results are separate.

All four CHAT tools (`scene_tasks`, `plan_pick_place`, `execute_pick_place_plan`,
`pick_place`) return one JSON object encoded as tool-result text. This replaces
prose-only outputs. Parse the JSON once; never infer success from `message`.
Tool names and input arguments are unchanged. The legacy Python bridge tuple
helper remains available; tool callers use the structured bridge result.

## Common response, schema version 1

```json
{"schema_version":1,"tool":"execute_pick_place_plan","status":"partial","code":"tracking_timeout","message":"tracking timeout","data":{"plan_ref":"plan:example-namespace:1","mode":"kinematic","assistance":["object_latch","object_placement"],"collision_scope":"base_endpoint_only","planned_ops":["approach","grasp","place"],"completed_ops":["approach"],"failed_stage":"grasp","measurements":[]},"recovery":"inspect"}
```

The same seven fields appear on successes and failures, including controller
fallbacks, invalid arguments and exceptions. `status` is `ok`, `error`, or
`partial`. `code` is a machine reason; `message` is explanatory text and is not
an API discriminator. `data` is an object; unavailable measurements are `null`,
and JSON never contains NaN or Infinity. Internal simulator body IDs, file paths,
and exception details are not part of the public contract. `assistance` describes
the selected manipulation mode; `base_motion: null` means this response does not
independently verify the base execution mode. Managed experiment manifests record
whether base teleportation was enabled.

`recovery` is one of:

| Value | Agent action |
| --- | --- |
| `none` | Continue with the requested workflow. |
| `rediscover` | Call `scene_tasks`, then select an unambiguous semantic task. |
| `replan` | Request a new plan; the previous handle cannot be reused. |
| `inspect` | Inspect the scene/state before deciding whether another action is safe. |

Partial execution always requires inspection. Completed operations are retained,
but their list does not prove that a failed operation had no physical effect.
Never automatically replay a grasp or placement after an uncertain outcome.

## Tools

| Tool | Inputs | Successful `data` |
| --- | --- | --- |
| `scene_tasks` | Optional `object_filter`, `robot` | Semantic tasks with `task_ref`, object/receptacle names, category inventories, optional reachability summary and its source/status. |
| `plan_pick_place` | `task_ref`, or `object_name` and `receptacle_name` | `plan_ref`, mode, assistance, collision scope, operation list. Planning executes no motion. |
| `execute_pick_place_plan` | `plan_ref` | Completed operations and available measurements. Consumes the handle once, even on refusal/failure. |
| `pick_place` | `object_name`, `receptacle_name` | Same execution data for the combined plan/execute path. Configured-controller fallback reports its own mode and uses null when detailed evidence is unavailable. |

Example sequence:

```json
{"name":"scene_tasks","arguments":{"object_filter":"bowl"}}
{"name":"plan_pick_place","arguments":{"task_ref":"task:example-namespace:1"}}
{"name":"execute_pick_place_plan","arguments":{"plan_ref":"plan:example-namespace:1"}}
```

Use handles returned by preceding responses, not assumed counter values.
Task handles have a random registry namespace and remain stable within a session,
including filtered discovery. Scene/server changes create a new namespace. Plan
handles also have a random context namespace; neither handle type is reusable
across contexts. These opaque strings do not expose server boot IDs.
The scripted smoke runner supports `$task_ref` (first task supporting the script’s requested articulation states in the preceding
successful discovery) and `$plan_ref` (preceding successful plan). These are
runner substitutions, not arguments understood by the public tools. Failed
planning stops dependent execution; scoring uses the resolved task reference.
`scene_tasks` requires matching scene metadata; missing metadata returns
`metadata_unavailable`. Reachability priors are advisory and identify whether
live placements or a zero-pose proxy supplied their inputs.

## Preconditions and execution guarantees

Before grounding, the bridge captures an immutable snapshot of command-protocol
server boot, scene, capabilities and placements. Semantic resolution, grasp
resolution and approach grounding share those copied placements. The original
snapshot is revalidated after planning, before storage and before execution;
storage never replaces it with newer poses. Unbound builds cannot be stored. Quaternion signs are equivalent. Translation
changes over 0.01 m or rotation over 5 degrees require replanning. Restarted
servers invalidate stored plans.

Guarded agent planning and stored execution require an advertised
`sim_base_pose_query: true` endpoint and a callable client `check_base_poses`.
The in-process Robosuite server and the measured-snapshot Stretch MuJoCo server
provide it (Stretch support merged in #205). An unsupported provider returns `base_pose_validation_unavailable`
before grounding or motion, with `recovery: "rediscover"`: connect to a provider
with endpoint validation, rediscover its tasks, and request a new plan. Repeating
the same request on an unsupported provider cannot fix it. Capability loss also
consumes existing plan handles. There is no pose-only fallback. Advertised queries
that time out or return malformed evidence still return
`approach_validation_failed`.

Robosuite agent plans use the online 16-pose approach ring, including rotated
base yaw. This changes the prior fixed +Y approach baseline; historical scores
must not be treated as validation of this approach policy.

 The executor queries base endpoint clearance
again immediately before motion; failed, malformed or unsupported queries stop
execution. Arm execution uses measured joints and rejects observations whose
client receive age exceeds two seconds. Missing measurements do not establish
arrival. Existing 35 mm EE arrival and 80 mm lifted-object distance bounds remain.

Common errors include `ambiguous_object`, `ambiguous_receptacle`, `unknown_plan`,
`scene_changed_replan`, `approach_changed_replan`, `approach_validation_failed`,
`joint_bounds`, `collision`, `ik_failed`, `planning_failed`, `tracking_timeout`,
`stale_observation`, and `verification_failed`. A tracking timeout does not imply
contact: contact causation requires separate measured evidence. Public results
expose safe measurement summaries. Each operation starts with cleared verification
fields and a unique internal operation ID; results copy only evidence belonging
to that operation. Early failure produces no new measurements, while completed
operations retain their evidence. Executors without this evidence interface
report no measurements. Detailed joint/contact diagnostics remain in
experiment logs. Release or retraction failure must not report full success.

## Scope and validation

These interfaces support GT-backed simulator planning. Kinematic manipulation
uses object latch/placement assistance; the managed gates also enable base
teleportation. The public `collision_scope` currently reports base endpoint checks. Placement
now performs additional sampled arm/payload checks against declared geometry
through the [placement Python API](placement.md), but those details are not yet
propagated per stage in public results. Neither interface certifies swept base
transport or complete physical robot geometry. Configured
hardware controllers retain their existing behavior and do not gain physical
certification from this API change.

Run the offline contract suite in `src/test/controller/task/tamp/`, agent tool
sequence tests, arm-planning/measurement tests, and scripted-tool runner tests.
Run live smoke through `scripts/run_tamp_agent_tools_gate.sh` under `emet jobs`.
Readiness additionally requires three reference-admitted cleanup fixtures
(scene00/02/12 cleanup0), three fresh process repeats each, with all nine tasks
passing. Preserve admission failures and failed repeats; do not substitute easier
fixtures or pool assisted historical scores with new results.

## Native execution integration (experimental debug adapter)

`agent.manip_mode=physical` selects actuator/contact execution. It never falls
back to object teleportation or attachment. Inputs default to
`agent.tamp_inputs=observed`. Observed discovery uses the enabled open-vocabulary
RGBD processor; without a provider it returns `observed_scene_unavailable`.
Observed planning and execution remain unavailable.

Explicit `privileged` inputs enable the packaged **Stretch default table scene**
adapter. The existing four tools discover stationary pick/place tasks, validate
arm and payload paths on private MuJoCo state, then execute through normal client
arm/gripper commands. Planning has no actuation. Execution consumes its plan
handle, rechecks measured scene state, and replans from measured joints. It verifies
one second of bilateral hold and two seconds of settled support after release.
These checks use **simulator ground truth**, not inferred tactile sensors. Public
results label `geometry_source: simulator_ground_truth`, `base_motion: stationary`,
and `assistance: []` (no runtime latch/teleport; privileged perception remains explicit).

A model signature and server boot identity bind the adapter to the snapshot.
Physics capture and client receive timestamps must both be fresh within two
seconds. Missing snapshots, unsupported models, and stale state reject execution.
Native `set_receptacle_state` returns `native_articulation_unavailable`; it never
uses assisted articulation. Other robot/scene adapters and navigation are outside
this initial native integration. Existing assisted defaults are unchanged.

The scripted public-tool validation accepts `--manip-mode physical
--tamp-inputs privileged --physical-audit PATH`; the audit rejects simulator
teleport, attachment, and direct joint-setting commands. Passing this validation
would establish deterministic public-tool execution, **not LLM task selection,
exploration, observed perception, or hardware readiness**.

The shared physical executor now requires an observed object-pose provider and
observed collision provider by default. Object poses carry an observation ID,
source and receive time; missing, stale or privileged evidence cannot verify an
observed grasp. The existing physical benchmark explicitly selects privileged
inputs. Native dynamics and realistic sensor inputs are separate requirements.

The Stretch and generic clients expose `get_joint_state_evidence()`, an immutable
position vector paired atomically with its local receive timestamp. Native
adapters can reject unknown/stale feedback without racing two separate reads.
This is receive freshness, not a sensor capture timestamp or evidence of contact;
no missing effort, object pose, or localization values are synthesized.

## Current placement and recovery limitations

Placement has no separate public CHAT tool: it runs within execution of the
existing pick/place tools. Placement failures preserve codes including `no_collision_free_placement`,
`placement_geometry_unavailable`, `placement_invalidated`, and
`release_execution_error`. Per-place measurements include sanitized
`placement_search` counts, geometry source, and its sampled collision scope.
The top-level scope remains conservative for the entire task. `completed_ops` alone cannot tell an
agent whether release occurred during a failed `place`. Do not infer that a new
grasp is safe from this response. There is not yet a dedicated recovery tool in
this four-tool API.

The latest full readiness gate (`20261009_081750_e342cc`, source `1335d5d5`)
failed CHAT placement and admitted 0/3 fixtures; repeats did not run. See the
[merge and metric plan](../plans/2026-10-09_tamp_merge_and_metrics.md) for acceptance
criteria and the distinction between historical assisted scores and current
end-to-end readiness.

The clutter benchmark runner now stops after the first execution or relocation
verification failure. It reports `recovery_required: true` and
`unattempted_bodies` in its private benchmark JSON, skips navigation, and retains
all requested objects in the denominator. These body IDs are not public tool
output. Planning failures before execution can still proceed to another object.

Per-place `release` evidence records `detach_command` as `not_attempted`,
`unknown` (command attempted but completion unavailable), or `returned` (the helper returned without error; terminal completion is not
independently checked), plus
`placement_verified` as a boolean or null. `held_state` remains `unknown`: a
returned detach helper and XY placement check do not provide a current
attachment observation. These fields help distinguish partial effects, but do
not authorize blind replay. Evidence resets at each operation and does not leak
arbitrary planner messages, internal body identities or previous search results.

## Simulator-assisted articulation

[`set_receptacle_state`](tamp_articulation.md) opens/closes a supported receptacle
using a current task handle and returns the same JSON envelope. Discovery now
includes an `access` field. Closed supported fixtures must be opened before
planning; unsupported articulated goals fail before pickup. This separate action
is explicitly assisted, invalidates prior plans, and does not certify a door sweep.

## Placement state refresh

Placement waits up to two wall seconds for the state receiver to recover before
planning or executing a segment. The maximum accepted receive age remains two
seconds; command targets never replace measured joints. Incomplete or nonfinite
joint arrays cannot partially overwrite the offline planning model. Logs include
`Measured state refresh` JSON with `code`, `wait_s`, and `state_age_s`.

Public placement failures retain `placement_stale_observation`,
`placement_missing_joint_state`, `placement_nonfinite_joint_state`, or
`placement_path_invalidated`, including nested preplace/place failures. They
require inspection of the held object and current scene before recovery; a failed
refresh never causes release. This distinguishes a state-stream failure from
changed obstacle geometry. A bounded wait is a recovery mechanism, not evidence
that the original live failure was caused by receiver starvation.

Native Stretch grasp calibration uses the opposing finger-pad contact edges from
the bound robot model, rather than treating the named EE frame as the pinch point.
This requires no additional contact sensor. Native debug execution uses measured
simulation time for settling, bounded by a 30-second wall watchdog per wait;
ordinary hardware executors keep wall-clock timing. Simulator snapshots and final
contact checks remain explicitly privileged debug evidence.

The native candidate search tries a 15-degree pitched robot frame before the
level and measured frames, for all objects. Every accepted candidate still needs
arm/payload collision validation and measured hold/release verification. The
current controlled acceptance is 4/4 live ZMQ and 6/6 separate CPU transfers;
this is not validation of clutter rearrangement, native base navigation, or
observed-input exploration.

If a native command acknowledgement times out after execution starts, the result
is `partial` with `code: command_outcome_unknown`, the last attempted operation,
and any completed operations. The handle remains consumed. Inspect measured
state before replanning: a missing acknowledgement does not mean the controller
failed to receive the command, and the tool does not automatically replay it.

### CHAT composition and failure handling

CHAT decodes the versioned TAMP envelope for outcome accounting. `status: partial`
or `error` cannot become success because the accompanying message sounds positive.
Malformed TAMP responses also fail outcome accounting. EQA and legacy non-TAMP
results retain their existing interpretation.

A non-successful TAMP call stops the remainder of that model-generated tool batch.
A JSON `dispatch` result with `code: batch_stopped` lists `unattempted_tools`.
The agent session stays alive and receives the failure for inspection or replanning
on its next turn. This is a batch boundary, not an autonomous recovery policy or
a persistent motion interlock: fresh state and appropriate recovery must still
precede a new plan. Successful execution should be followed by reading its result
and rediscovering before planning the next manipulation.

For native composition diagnostics, `scripts/eval_native_table_contact.py` accepts
`--steps N` (default 1) for repeated transfers without resetting physics, or
`--sequence object1 object2` for an explicit fixture sequence. `--separation`
and `--separation-axis x|y` alter initial fixture positions only. Every step goes
through the CHAT dispatcher, reacquires a task, and plans from measured state;
a failed call stops the sequence. After a successful step, the evaluator checks
released support for all objects already placed, using privileged simulator
state. These are scripted CPU integration checks, not LLM, live ZMQ, exploration,
or hardware acceptance. Launch them through managed CPU jobs.

### Native placement regions and bounded grasp search

`plan_pick_place` optionally accepts `placement_region: [[xmin, ymin], [xmax, ymax]]`
in world-frame meters. Coordinates must come from grounded geometry or an explicit
fixture goal. The native privileged Stretch adapter clips the selected support to
this region and searches for placements containing the full object footprint.
Measured release verification checks the whole footprint against the requested
region with a 5 mm tolerance, in addition to sustained support, position, and
velocity checks. The native planner retains its 2 cm placement clearance margins;
very narrow regions can therefore be rejected even if an idealized object fits.
Other adapters return `placement_region_unsupported` rather than ignoring the goal.
Malformed/nonfinite/empty bounds return `invalid_placement_region`; a region not
overlapping the support returns `placement_region_unavailable`.

Native grasp search tries the existing 15-degree approach, bounded 45/60-degree
approaches with object-height-derived pinch offsets, then level/measured frames.
Every candidate uses the same collision, tracking, and retention checks. A grasp
with no downstream placement now resumes grasp search. A shared limit of 256
pose-IK calls bounds grasp and placement exploration. `no_plan_within_budget`
means this search did not find a complete plan, not proof of infeasibility.
`planning_rejections` preserves sanitized per-stage failure counts instead of
attributing the whole search to the last failed wrist candidate. Public plan
responses include `planning_ik_calls`, `max_ik_calls`, the requested region, and
the proposed `placement_goal` in the world frame. Raw colliding body names stay
in privileged diagnostic events, outside the public response.

The frozen two-object region file is
`configs/eval/native_table_two_object_regions.json` (cylinder goal, then box goal).

Native planning accepts either a task handle or semantic names. If names accompany
a handle, they must match that task's full published names (ignoring case and
outer whitespace); a contradiction returns `task_selector_mismatch` with
`recovery="rediscover"` before planning or motion. The error identifies the task
selected by the handle. Successful plan results and consumed-plan execution
results, including failures, carry `object_name` and `receptacle_name`. These
semantic labels are saved in the plan and do not expose private simulator IDs.
An operation's `ok` verifies that selected operation, not every goal in a
multi-step task.

Use the contact harness with `--sequence object2 object1 --separation .15
--separation-axis y --placement-regions PATH`. The separate
`scripts/eval_native_table_agent.py` evaluates LLM selection with the CHAT
prompt/parser/dispatcher and independent region/support scoring; it is not the
interactive `run agent` CLI. By default it uses in-process physics; `--live-sim`
selects live ZMQ execution as described below. Its `--device` defaults to `cpu`;
`--device cuda` requires a managed GPU reservation. Both harnesses require managed jobs.

For an existing remote OpenAI-compatible server, supply both
`--base-url http://HOST:8000/v1 --remote-model MODEL_ID`. This uses the ordinary
OpenAI-compatible agent client and loads no local model weights. Remote requests
have a 120-second timeout and no automatic retries; the managed runner should
also bound total episode time. Remote decoding uses the server defaults, and
`--seed` only controls the local process. Archive the served model revision and
sampling configuration alongside results. Physics still pauses during inference
in this in-process harness; remote hosting does not establish live execution.

To evaluate the same regional agent goals against a live server, add
`--live-sim PATH_TO_SIM_YAML` to `scripts/eval_native_table_agent.py` and launch
through a managed job. Reserve the GPU when rendering GPU camera images. Use
a matching packaged Stretch scene with fixture
initial transforms set before startup. The harness enables the physical actuation
audit, reads fresh measured snapshots, and records simulator time on both sides
of every inference request. Passing requires physics progress during inference,
a clean nonempty actuation audit, and the same independent stable final-goal
checks. Remote inference avoids local model weights; GPU camera rendering still
needs a GPU reservation. The scope remains privileged and stationary; this
is not exploration or observed-input acceptance.

For this privileged text-only evaluator, `no_cameras: true` in the simulator YAML
avoids rendering and local GPU use entirely. The harness opts into
`StretchZmqClient(state_only=True)`: missing image streams are permitted only
when state explicitly advertises `is_simulation=true` and integer
`capabilities.num_cameras=0`. Default clients and hardware still require their
normal startup streams. No image observations are fabricated. This mode does
not validate visual exploration or perception.

The [live agent experiment](../experiments/tamp_native_live_agent_20261014.md)
records two successful two-goal episodes with physics advancing during every
model call, plus the IK binding mitigation and its remaining runtime uncertainty.

The subsequent [held-out and actual CHAT gate](../experiments/tamp_native_heldout_chat_20261014.md)
records placement failures and bounded session recovery. See
[run-agent TAMP sessions](../AGENT_RUN.md#bounded-multi-step-tamp-sessions) for the
round budget and explicit camera-free diagnostic mode.

## Observed discovery integration hook

Python embedders of `run_agent_with_robot` can pass `observed_tamp_provider`, an
`ObservedTaskProvider` from
`emet.controller.task.tamp.observed_discovery`. The tool context uses the same key.
The provider declares `input_mode = "observed"` and implements `observe_tasks()`
to return one coherent tuple of `ObservedTaskCandidate` values. Each candidate
contains semantic object/receptacle names and object/support evidence. Tracked
poses can use `ObjectPoseEvidence` (`rgbd_tracking` or `stereo_tracking`); segmented
depth positions use `ObservedPositionEvidence` (`rgbd_pointcloud` or
`estimated_depth_pointcloud`). Position evidence carries no inferred rotation.
Labels alone are insufficient to establish a support.

Both estimates preserve their original observation receive times and must be at
most two seconds old. Discovery does not renew
those times. Unknown/occluded entities must be omitted; empty observations return
an empty task list. Stale, future-dated, or privileged evidence rejects the entire
snapshot, including evidence excluded by the caller's category filter. Receive
freshness alone is not a claim of synchronized sensor capture or calibrated poses.

`scene_tasks` returns the existing versioned JSON envelope, with
`data.inputs = "observed"`, `discovery_only = true`, `planning_available = false`,
and `execution_available = false`. Task entries contain opaque per-discovery
handles, semantic names, world positions, sources and observation ages. These are
proposals, not executable plan handles or certified affordances. The CHAT loop
summarizes this result and stops the turn instead of continuing into manipulation.
Native planning/execution gates remain unchanged and still reject observed mode.

With `--memory-backend open_vocab`, `run agent` automatically uses the
`RgbdTaskProvider` attached to its existing scene-graph processor. Explicit Python
injection takes precedence. No new detector or tracking model is introduced. The
provider uses only the current segmented RGBD frame; accumulated scene-graph
bounds, cached semantic localizations and simulator metadata are not inputs.

The provider proposes small segmented objects (robust extent 1–30 cm) and broad
horizontal upper depth patches (at least 15 cm in each horizontal direction and
0.03 square metres, with 30 depth points in a 2 cm height band). These geometric
heuristics produce task candidates, not support/load-bearing or graspability
certificates. Each object is paired with other observed support candidates in
that frame. Estimated depth is labeled separately from sensor RGBD. Unknown,
empty or untimestamped frames clear the proposals; a processing failure also
clears the previous frame before processing starts.

Stretch stamps its RGBD receive time locally when a full observation arrives.
The timestamp travels with that observation through depth resolution, mapping
and scene-graph segmentation; reading or reprocessing the same frame does not
renew it. Reused inferred depth is not stamped with a newer image's receive time
and cannot produce fresh proposals. Slow processing can make the result stale,
which is rejected. This is
local receive freshness, not a hardware capture-time or clock-synchronization
claim. Other robot clients must supply the same local observation provenance
before they can produce fresh proposals.

The camera-free diagnostic and plain `dynamem`/GraphEQA presets do not activate
this processor and still need an explicit provider for discovery. Observed
obstacle/payload geometry and execution verification remain prerequisites for
motion. The first real camera/detector diagnostic rejected stale evidence after 16.5 s
of CPU processing and exposed incorrect support labels. Live discovery acceptance
is still pending; see the [diagnostic report](../experiments/tamp_observed_discovery_20261014.md).
Controlled segmentation tests cover real depth projection and the processor/tool
paths, but do not establish detector accuracy.
