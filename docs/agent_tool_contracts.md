# Agent tools and contracts

Source ownership and execution diagrams: [Agent architecture](agent_architecture.md).

Use this page to see what an agent can request, what can change in the world,
and what a successful result establishes. The [tool overview](agent_tools.md)
links to deeper setup/API guides. The executable schemas live in
[skill specs](../src/emet/agent/skills/specs.py); the actual session's advertised
schema is authoritative for optional arguments and capabilities.

CHAT has 34 registry entries, typically 32 advertised when the general task-plan
provider is absent. EQA has ten registry names including the compatibility alias
and two mutually exclusive terminal tools. Model-final EQA normally advertises
eight base tools, with up to three shared floor/base tools when supported.
These are two orchestration modes of shared components, not promises that every
robot supports every action. No new tool was added for task gaze or movement-view
retention.

## Complete inventory

Inputs below are the main arguments, not a substitute for JSON schemas.
Read-only queries may write diagnostic files. Photo delivery and speech depend
on the active communication integrations.

<!-- tool-inventory:start -->

### CHAT: observation and memory

| Tool | Main inputs | Effect | Contract |
| --- | --- | --- | --- |
| `describe_scene` | optional pan/tilt | Caption, or explicit head motion/capture | No args: caption and attach the same owned head snapshot without motion; freshness is independently unverified. Angles: measured stationary gaze, no extra caption call; arguments appear only with a qualified head contract. |
| `ground_object` | complete object query | RGB-D perception and object admission | Uses the decision's owned snapshot. Returns world geometry or explicit failure, never a retrieval-anchor fallback. |
| `query_memory` | question | Read memory/model inference | Answer about recorded evidence; not a fresh visibility or localization receipt. |
| `query_scene_graph` | question | Read memory/model inference | GraphEQA when configured, otherwise open-vocabulary graph; overlaps query_memory but retains different fallback paths. |
| `list_scene_relations` | none | Read graph | Recorded relations and identities; no new observation or physical feasibility proof. |
| `send_image` | none | Head photo/delivery | Current available head image; no promised camera motion or independent acquisition freshness. |
| `take_picture` | none | Local capture command and head photo | Runs the configured local capture command and queues an available head frame for the reply. Missing frame is an explicit failure; local persistence and remote delivery are not inferred from executor liveness. |
| `send_object_image` | object label | Stored crop/delivery | Historical object crop, explicitly different from a live head image. |
| `aim_arm_at` | object label | Arm/kinematic camera aim | Requires controller support; inspect failure. This legacy aim skill does not authorize pickup/contact. |
| `take_ee_picture` | none | Wrist photo | Requires successful prior arm aim in the session. Missing/dark stream cannot be substituted with an invented image. |

### CHAT: movement and mapping

| Tool | Main inputs | Effect | Contract |
| --- | --- | --- | --- |
| `scan_environment` | none | In-place base/head survey and map updates | May rotate; scan completion is not target finding, safe translation or changed-view proof by itself. |
| `explore` | none | Navigate/map | Coverage action. Read plan, measured movement, rejection and cancellation feedback. |
| `find_objects` | text query | Search, navigate and inspect | Failed localization/approach remains failure even if the executor stays alive. Search candidates are hypotheses. |
| `observe_floor` | blocker flag, optional pan/tilt | Bounded downward sensing and mapping | Requires measured head/fresh calibrated depth. Rechecks a fixed rejected footprint; restores task gaze. Sensing does not authorize a route. |
| `rotate_base` | degrees | Validated relative yaw | Positive means left/CCW. Versioned receipt distinguishes request, execution and measured rotation. |
| `move_forward` | meters | Validated relative translation | Bounded explicit distance; physical mode checks the swept footprint. Requested travel is not measured travel. |
| `face_toward` | object label | Turn toward remembered XY | Resolves coordinates to the planning world and uses validated rotation. A remembered location is not current visual confirmation. |
| `go_home` | none | Navigate | Requires a recorded starting pose/map. Legacy action dispatch, not a native TAMP certificate. |
| `navigation_diagnostics` | none | Read map/plan | Map counts, planning-world clearance and last outcome; center clearance and full swept-footprint checks are distinct. |
| `send_map_snapshot` | none | Render/deliver stored map | Shows observed/occupied map and optional plan, not GT traversability or permission to execute that plan. |

### CHAT: tasks and manipulation

| Tool | Main inputs | Effect | Contract |
| --- | --- | --- | --- |
| `scene_tasks` | optional category/robot | Discover tasks | Returns opaque handles plus input/execution modes and assistance. Proposals do not certify grasps or routes; discovery can be unavailable. |
| `plan_pick_place` | task_ref or names; optional region | Plan only | A plan handle is not execution. Check support, input mode and feasibility. Region constraints are capability-specific. |
| `execute_pick_place_plan` | plan_ref | Revalidate/actuate | One-shot plan consumption; stale/partial/unknown execution requires inspection and replanning, not replay. |
| `pick_place` | object/receptacle names | Plan/actuate through configured path | Inspect versioned outcome, assistance and partial execution. Find success is not pick/place success. |
| `set_receptacle_state` | task_ref, state | Assisted simulator articulation | Joint-state setting, not a physical opening skill or door-sweep collision guarantee. Physical mode refuses assistance. |
| `plan_task` | instruction, strategy | Multi-step plan | Advertised only with a bound qualified provider. Planning/input modes remain explicit; a semantic plan is not an execution receipt. |
| `execute_task_plan` | plan_ref | Revalidate/execute skill sequence | Provider-gated, consumed once; checks measured effects and protected goals. Partial results stop for inspection. |
| `hand_over` | none | Legacy person-search/navigation/arm action | Requires implementation on the connected robot; not a qualified native TAMP recovery primitive. |

### CHAT: communication and session

| Tool | Main inputs | Effect | Contract |
| --- | --- | --- | --- |
| `say` | text | Speech | Requires TTS; speech is not evidence of physical action. |
| `wave` | none | Gesture/arm motion | Legacy robot-supported gesture; no manipulation success guarantee. |
| `nod_head` | none | Head gesture | Legacy gesture, distinct from measured task inspection. |
| `shake_head` | none | Head gesture | Legacy gesture, distinct from measured task inspection. |
| `avert_gaze` | none | Look-away gesture | Legacy gesture, not a qualified capture or localization action. |
| `quit` | none | End session/request stop | Conversation termination through the executor. |

### EQA_EPISODE: evidence, action and termination

| Tool | Main inputs | Effect | Contract |
| --- | --- | --- | --- |
| `read_history` | message_id, optional pagination | Read archived dialogue | Returns exact historical text and image IDs; does not reacquire a view. |
| `recall_image` | image_id | Reattach retained pixels | Historical evidence only; not continued visibility, new localization or motion. |
| `inspect_graph` | none | Read detections/views | Distinguishes proposal handles from actual camera-view identities. A viewpoint is not an object position. |
| `explore_frontier` | optional toward hint | Navigate/capture/assess | Coverage bias is weak semantic context, not an object coordinate or guaranteed arrival. |
| `investigate` | obs_id, optional approach_index | Approach/capture/assess | Use listed handles. Preserve failed-motion receipt even if a subsequent current-pose assessment succeeds. |
| `navigate_to_obs` | obs_id, optional approach_index | Compatibility alias | Same investigate path; not advertised by modern model-final routing. |
| `look_around` | optional qualified pan/tilt | Scan or measured head capture | No args retain configured scan policy. Explicit gaze uses adapter limits and post-arrival capture, not a scan. |
| `verify_siglip` | phrase, optional view obs_id | Retrieval/visual assessment/admission | Similarity, VLM presence, answerability and admitted geometry are separate fields and separate claims. |
| `submit_answer` | optional semantic answer | Answer-mode termination | Model-final preserves an explicit model answer; omission requests a final model read. Semantic correctness does not satisfy OVMM localization. Legacy policies can have different termination gates. |
| `finish` | optional summary | Explore-mode termination | Mutually exclusive with submit_answer; completion follows the selected episode policy. |

<!-- tool-inventory:end -->

EQA can additionally bind the same `observe_floor`, `rotate_base` and
`move_forward` contracts when the adapter/controller supports them. Runtime
limits and the actual advertised pack remain the source of truth.

## Result handling

CHAT retains typed outcomes through dispatch, TAMP continuation and historical
summaries. Prompt/log text is a rendering, not an execution receipt to parse
again. Session continuation and task success are independent. `send_image`,
`take_picture`, `scan_environment` and `explore` return explicit outcomes;
missing frames, interrupted scans and rejected exploration remain failures.
Legacy prose receives `unverified_result`, never success inferred from wording.
Unknown legacy results are recorded as unknown in the opt-in attempt ledger.
Versioned TAMP and relative-motion results retain their stricter envelope checks.
Their nested receipt data is copied at admission, so later provider updates cannot
rewrite retained progress. A typed wrapper whose success flag disagrees with its
embedded receipt is rejected; partial execution cannot become a successful
continuation. Rotate-only refusals retain their versioned reason and recovery
instruction through dispatch.

## Evidence boundaries

| Evidence | Establishes | Does not establish |
| --- | --- | --- |
| Retrieval score/search candidate | A possible place/object to inspect | Presence, an object XYZ or a reachable route |
| Model visual assessment | A fallible judgment on the supplied views | Calibrated confidence, exhaustive room coverage or physical geometry |
| Historical RGB/view | What was captured at its recorded pose/time/scope | Current presence or fresh manipulation permission |
| Admitted masked depth | Object-specific world points from the identified RGB-D view | Continued visibility, collision-free approach or feasible manipulation |
| Motion receipt | Whether execution started and measured progress/status | That requested travel happened in full or the next route is safe |
| Floor observation | New depth/map evidence and a fixed-footprint recheck | Free space inferred from absent depth; clearance through an occupied cell |
| Valid task plan | Feasibility under the stated provider/model | Successful physical execution or ground-truth-free perception |

Keep the original question/options and the model's submitted answer intact.
Search phrases and auxiliary assessments are hints. A visible count is not
automatically a total count; supply wider evidence and coverage limitations for
the coordinator to judge. No minimum-view rule or automatic answer replacement
is implied by these contracts.

## Head and frame contract

Explicit head angles require declared finite limits, pose feedback, stationary
base feedback and a post-arrival frame mechanism. Capability binding removes
unsupported optional arguments rather than making the model discover them by
spending an action. It does not select behavior using robot names.

- Streaming bridges use advancing observation reception, retaining available
  acquisition timestamps. Reception alone is labeled accordingly.
- Habitat declares a non-inverted virtual-camera orientation range and renders
  synchronously after arrival; its native acquisition clock must advance. These
  are virtual sensor limits, not simulated mechanical actuators.
- Out-of-range requests reject without silent clipping at the inspection boundary.
  Failed movement or stale capture cannot produce successful inspection.
- `describe_scene`/`look_around` remain available without head arguments. Floor
  inspection keeps its separate downward/depth requirements.
- CHAT `describe_scene` with a non-null pan or tilt invalidates TAMP evidence and
  handles before/after the possible head motion, including failure. Zero-valued
  angles count as requests; caption-only and explicit null angles remain read-only.

The implementation is [head observation](../src/emet/controller/dynamem/head_observation.py),
[head capabilities](../src/emet/robots/head_capability.py),
[Habitat adapter](../packages/emet_habitat/emet_habitat/robot_client.py) and
[session bindings](../src/emet/agent/skills/eqa.py).

## Movement-view retention

[NavigationViewBuffer](../src/emet/memory/navigation_views.py) samples already
acquired RGB/optical-pose/timing from native Habitat steps and decoded ZMQ
observations. Defaults: 32 frames, 640-pixel maximum side, at least 0.5 m
translation or 20 degrees of viewing-direction change from the last retained
view. These are sampling settings for evidence, not task-completion thresholds.
Missing acquisition metadata is counted and skipped, not replaced with invented
sensor time. A map reset replaces the buffer. Frames are historical evidence;
no depth, masks, semantic IDs, GT object states, model calls or map nodes are
added by retention.

Qualified movement frames enter the existing bounded
[view selector](../src/emet/memory/view_selection.py), using pose diversity when
no explicit ranking policy was configured. Current assessment and qualified
pre-turn views remain pinned; the four-image assessment ceiling still applies.
Images retain roles, view IDs, scope and age. Selection/sampling diagnostics are
traced. A view must be retained before any ranking strategy can choose it.
When query-image dumps are enabled, `vlm_packet_pngs` lists the assessment
packet images in prompt order (an unavailable dump path is null). Historical
packet dumps do not overwrite the canonical current-view PNG.

Configure `mapping.eqa.navigation_views` in the unified YAML (`eqa.navigation_views`
in resolved controller parameters):

```yaml
mapping:
  eqa:
    navigation_views:
      enabled: true  # false is the retention ablation
      max_frames: 32
      min_translation_m: 0.5
      min_rotation_deg: 20.0
      max_side: 640
    # Existing view_selection controls ranking and selected-image budget.
```

## Outcomes and recovery

Structured CHAT/EQA outcomes use `ok`, `status`, a note/reason and separate
observation/visual/geometry/motion payloads. TAMP and relative base-motion tools
use versioned envelopes: inspect `schema_version`, `status`, `code`, `data` and
`recovery`, not prose. [ToolOutcome](../src/emet/agent/tool_outcome.py) renders
actionable details into the next model context. Legacy action-only tools above
still dispatch through their executor and do not supply this full measured
contract; do not interpret their acknowledgement as task success.

| Status/reason | Meaning for the coordinator |
| --- | --- |
| `invalid_head_pan` / `invalid_head_tilt` | Request exceeds active bounds; no successful head view. Read the returned limits. |
| `head_limits_unavailable` | Explicit gaze contract is unavailable; use advertised alternatives. |
| `head_pose_unconfirmed` / `stale_observation` | Arrival/capture was not established; do not claim a new inspected view. |
| `base_moved_during_head_observation` | Stationary-inspection assumption failed. |
| `observation_scope_mismatch` | Map changed since the decision snapshot; acquire a new decision view. |
| `not_grounded` | Read visual/model/depth reasons; no admitted target XYZ. |
| `motion_rejected` + unknown footprint | Execution did not begin; bounded sensing may provide missing evidence, followed by replanning. |
| `motion_rejected` + occupied footprint | Execution did not begin; sensing is not a reason to erase occupancy. Consider another route/view. |
| TAMP partial/unknown execution | Inspect current state and obtain a new plan; never replay a consumed handle. |

Physical guards, admission and measured execution remain deterministic. Choosing
another view, route, tool or final answer belongs to the coordinator, subject to
the declared action budgets. No harness retry script is implied.

Episode dialogue archives older state text by reference when needed. Current
instructions, assistant actions and tool receipts remain exact. If those
required fields alone exceed the configured text budget, inference fails
explicitly and the original dialogue is preserved; they are not silently
truncated to force another model call.

Experimental request recording uses private artifact directories and redacts
common JSON credential fields on disk while forwarding original network bytes.
Task text and images remain sensitive; free-text secrets are not automatically
detected. Use ignored `logs/` or `runs/`, or storage outside the checkout, and
check recording status and original/recorded hashes before treating an artifact
as an exact request.

## Updating a contract

Change the canonical spec, mode binding, handler and model-facing receipt
together. Keep compatibility aliases explicit. The
[inventory coverage check](../src/test/agent/test_tool_contract_catalog.py)
ensures every registry name appears here. Tests for behavior are in
[task gaze](../src/test/agent/test_task_gaze.py),
[CHAT/HTTP history](../src/test/agent/test_chat_camera_history.py),
[movement evidence](../src/test/memory/test_navigation_views.py),
[assessment selection](../src/test/memory/test_assessment_view_history.py), and the
[reusable observation cases](testing/agent_observation_cases.md). Passing unit
contracts is separate from live model/task acceptance.
