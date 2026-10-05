# Agent-facing TAMP simulator API

All four CHAT tools (`scene_tasks`, `plan_pick_place`, `execute_pick_place_plan`,
`pick_place`) return one JSON object encoded as tool-result text. This replaces
prose-only outputs. Parse the JSON once; never infer success from `message`.
Tool names and input arguments are unchanged. The legacy Python bridge tuple
helper remains available; tool callers use the structured bridge result.

## Common response, schema version 1

```json
{"schema_version":1,"tool":"execute_pick_place_plan","status":"partial","code":"tracking_timeout","message":"tracking timeout","data":{"plan_ref":"plan:1","mode":"kinematic","assistance":["object_latch","object_placement"],"collision_scope":"base_endpoint_only","planned_ops":["approach","grasp","place"],"completed_ops":["approach"],"failed_stage":"grasp","measurements":[]},"recovery":"inspect"}
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
{"name":"plan_pick_place","arguments":{"task_ref":"task:1"}}
{"name":"execute_pick_place_plan","arguments":{"plan_ref":"plan:1"}}
```

Use handles returned by preceding responses, not assumed counter values.
`scene_tasks` requires matching scene metadata; missing metadata returns
`metadata_unavailable`. Reachability priors are advisory and identify whether
live placements or a zero-pose proxy supplied their inputs.

## Preconditions and execution guarantees

Plans bind to the command-protocol server boot, scene, selected capability, and
finite object/receptacle poses. Quaternion signs are equivalent. Translation
changes over 0.01 m or rotation over 5 degrees require replanning. Restarted
servers invalidate stored plans. The executor queries base endpoint clearance
again immediately before motion; failed, malformed or unsupported queries stop
execution. Arm execution uses measured joints and rejects observations whose
client receive age exceeds two seconds. Missing measurements do not establish
arrival. Existing 35 mm EE arrival and 80 mm lifted-object distance bounds remain.

Common errors include `ambiguous_object`, `ambiguous_receptacle`, `unknown_plan`,
`scene_changed_replan`, `approach_changed_replan`, `approach_validation_failed`,
`joint_bounds`, `collision`, `ik_failed`, `planning_failed`, `tracking_timeout`,
`stale_observation`, and `verification_failed`. A tracking timeout does not imply
contact: contact causation requires separate measured evidence. Public results
expose safe measurement summaries; detailed joint/contact diagnostics remain in
experiment logs. Release or retraction failure must not report full success.

## Scope and validation

These interfaces support GT-backed simulator planning. Kinematic manipulation
uses object latch/placement assistance; the managed gates also enable base
teleportation. Endpoint collision checks do not certify swept arm or payload
clearance, and RBY1 arm collision geometry remains incomplete. Configured
hardware controllers retain their existing behavior and do not gain physical
certification from this API change.

Run the offline contract suite in `src/test/controller/task/tamp/`, agent tool
sequence tests, arm-planning/measurement tests, and scripted-tool runner tests.
Run live smoke through `scripts/run_tamp_agent_tools_gate.sh` under `emet jobs`.
Readiness additionally requires three reference-admitted cleanup fixtures
(scene00/02/12 cleanup0), three fresh process repeats each, with all nine tasks
passing. Preserve admission failures and failed repeats; do not substitute easier
fixtures or pool assisted historical scores with new results.
