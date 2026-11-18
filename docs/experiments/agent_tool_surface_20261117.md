# Agent tool surface and live grounding follow-up

The current registry has **34 CHAT tools**. The live operator session advertises
**32**, because `run_agent_with_robot` suppresses `plan_task` and
`execute_task_plan` without a qualified task provider. The previous comparable
session advertised 31. `ground_object` adds one operation; this trial does not
add another tool or change tool selection policy.

Source: `abc68d70029b594e1cb61088fe0d2f2874bcf22e`, branch
`fix/closer-view-feedback`. The exact production request is the authority for
advertisement, rather than counting registry declarations alone.

## Inventory

| Role | Registered | Advertised in this session | Names |
| --- | --- | --- | --- |
| Perception/memory | 10 | 10 | `ground_object`, `describe_scene`, `query_memory`, `query_scene_graph`, `list_scene_relations`, `send_image`, `take_picture`, `send_object_image`, `take_ee_picture`, `aim_arm_at` |
| Navigation/mapping | 10 | 10 | `explore`, `find_objects`, `navigation_diagnostics`, `send_map_snapshot`, `observe_floor`, `scan_environment`, `rotate_base`, `face_toward`, `move_forward`, `go_home` |
| Task/manipulation | 8 | 6 | `plan_task`, `execute_task_plan` (provider-gated here), `pick_place`, `plan_pick_place`, `execute_pick_place_plan`, `set_receptacle_state`, `scene_tasks`, `hand_over` |
| Communication/session | 6 | 6 | `say`, `wave`, `nod_head`, `shake_head`, `avert_gaze`, `quit` |
| Total | 34 | 32 | No duplicate registry names. |

Model-final EQA advertises eight base tools: `read_history`, `recall_image`,
`inspect_graph`, `explore_frontier`, `investigate`, `look_around`,
`verify_siglip`, and `submit_answer` (or `finish` in explore mode). Floor and
relative-motion capabilities can add `observe_floor`, `rotate_base`, and
`move_forward`, for eleven. These counts were checked using declared-capability
fixtures; they are not a new live Habitat or hardware inventory. The legacy
navigation alias remains handled but is not advertised in model-final mode.

Default JSON serialization of all 34 schemas is 18,905 characters. The actual
32-tool CHAT system prompt is 22,534 characters including instructions/examples,
versus 21,594 in the prior session, a 940-character increase. These are character
counts, not model token counts or total multimodal context measurements. All
twelve live requests advertise exactly the same 32 names.

## Consolidation priorities

The list is broad enough to warrant consolidation before further expansion.
The active trial retains its existing surface so tool pruning is not another
experimental variable.

1. **Memory queries:** `query_memory` and `query_scene_graph` call the same
   GraphEQA backend in this session. Make one advertised entry preserve the
   graph, voxel and open-vocabulary fallback behavior. Keep historical trace
   compatibility. Search coordinates must retain hypothesis status.
2. **Head-camera delivery:** `send_image` and `take_picture` differ mainly in
   delivery/storage. Qualify one capture interface with those outputs. Retain
   the distinction between a current head frame, a stored object crop, and a
   wrist capture requiring successful arm aim.
3. **Communication gestures:** assess grouping gestures into one operation or
   a session capability pack. Use existing registry/session machinery, without
   task-specific intent classification. Robot capabilities remain explicit.
4. **Planning layers:** the general task-plan pair is already provider-gated.
   Consolidate additional pick/place wrappers only after their execution and
   recovery semantics are proven equivalent; preserve OVMM/TAMP coverage.

Keep `ground_object` distinct from semantic memory questions and route execution:
it supplies admitted object geometry without moving. Use richer receipts and
existing capability interfaces for the remaining view problem rather than
proliferating separate recovery skills. There is no arbitrary hard tool-count
limit or new routing framework in this change.

## Bounded live operator retry

Job `20261117_004238_757a18`; artifacts:
`/home/cpaxton/runs/emet/external-coordinator-trial-20261117-r1/`.
The launch description says “34-tool pack”; that is the registry count. Actual
wire requests confirm 32 advertised tools, with the existing provider gate above.
The source remains unchanged throughout execution. The informed operator uses
the production CHAT loop, actual schemas/pixels/receipts and separate Caliban
Qwen perception; it has no simulator object poses, GT masks or private logs.
This is usability qualification, not a blind model comparison.

The same Molmo iTHOR scene and `rby1`/Galaxea proxy run under physical guards,
with teleport disabled. Verified query-instance admission is explicitly enabled,
matching the opt-in find profile. This and the code changes since the previous
trial prevent attributing differences solely to one tool. Startup image
simulation time advances 1.348→1.684 s; base-up dot world-Z is 0.999995.

| Observation | Result |
| --- | --- |
| Coordinator budget | 12 responses, including final summary; 11 tool calls. |
| Tool usage | Five base-tool requests, three floor inspections, three grounding calls; no manipulation. |
| Recovery | First floor observation adds 77 observed cells; targeted recovery takes a checked footprint from six unknown cells to zero. |
| Motion | Measured turns complete; requested 0.2 m observation retreat produces 0.130138 m measured progress. |
| Grounding | All three calls return `not_grounded` / `geometry.status=not_admitted`; no accepted XYZ. |
| Model replies | Parsed semantic abstentions with explicit reasons; depth support is null because admission stops before that stage. |
| Task | Microwave visually identified in a clearer view, but **no admitted object position**. |
| Process/audit | Normal public session end, managed job done; 27 accepted actuation-audit rows, zero forbidden entries. |

Base-tool requests are macro calls, not a count of physics commands. Process
completion and absence of forbidden actuation do not prove task success or
optimal route clearance. No broad sweep, real robot or manipulation is tested.

## The remaining gap is the view handoff

The subagent uses `ground_object` on its first visible-target opportunity and
does not use the redundant scene/memory query paths. It reports the single-query
schema as clear and discoverable. This informed GPT operator result does not
establish that a smaller Qwen coordinator is unaffected by the tool-list size.

Parent visual inspection confirms a concrete mismatch for the second grounding
attempt:

- [Coordinator's clearer microwave image, request 6](/home/cpaxton/runs/emet/external-coordinator-trial-20261117-r1/public/request_0006/message_011_image_01.png).
- [Actual grounding input, source voxel frame 14](/home/cpaxton/runs/emet/external-coordinator-trial-20261117-r1/private/perception/grounding/grounding-2ee03213b72549a6aff1a18420586cc7.png).

The first contains the microwave; the second crops most of it off the upper
edge. The controller's `ground_current_view` captures again through `update`,
so the grounding input is not the exact RGB-D snapshot behind the coordinator's
decision. A voxel source index is not automatically an identity for the camera
attachment. The cache proves which pixels Qwen received; its semantic abstention
does not establish target absence in the scene or a missing-depth failure.
The cause of the intervening optical change was not fully diagnosed here.

Next preserve and identify the exact decision RGB-D snapshot, calibration and
scope within the existing grounding capability. Test an unclipped decision view
followed by a clipped newer capture, and retain timing/physical reacquisition
boundaries. Stable task gaze and general pitch control may also be needed; if
so, consolidate existing inspection/gaze interfaces instead of adding separate
look-up/look-down recovery tools. Retaining the better frame may still expose
legitimate depth-range or route limits, so it is not a guaranteed task success.
Then repeat this bounded case before small paired EQA/OVMM acceptance gates.

No runtime code or new tool is added by this inventory/report checkpoint.
Source/wire inventory checks passed; a new accuracy claim or merge acceptance
is not established. Raw trial output, actor notes, inventory and a separate
manual evaluation are preserved under the run directory.
