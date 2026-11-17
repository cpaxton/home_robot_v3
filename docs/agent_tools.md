# Agent tools overview

This index lists the tools in the shared skill registry. **CHAT** is the
interactive terminal/Discord agent (`run agent`); **EQA_EPISODE** is the separate
embodied question-answering/exploration loop. Their tool packs are distinct.
Enabling a caption model in CHAT does not switch to the EQA episode loop.
See [running the agent](AGENT_RUN.md) and the [TAMP overview](tamp.md).

A registered tool is not a guarantee that the connected robot can execute it.
Handlers depend on the configured executor, robot, cameras, memory, and simulator
capabilities, and may return unavailable/refused results. Offline CHAT has no
robot execution. `EMET_BASE_ROTATE_ONLY` retains names but disables translation
and several manipulation tools. Inspect actual outcomes before continuing.

The source of truth for names, argument JSON schemas, and prompt descriptions is
[skill specs](../src/emet/agent/skills/specs.py). Implementations are
[CHAT handlers](../src/emet/agent/tools.py) and
[EQA bindings](../src/emet/agent/skills/eqa.py). This index describes capabilities;
use those schemas for exact arguments rather than copying prose into calls.

The current CHAT registry has 34 tools; a normal session without a qualified
task provider advertises 32 because the general task-plan pair is gated. The
model-final EQA pack has 8–11 depending on capabilities. See the
[exact inventory and consolidation review](experiments/agent_tool_surface_20261117.md)
for the current names, measured prompt sizes and live qualification limits.

## CHAT: perception and memory

| Tool | Purpose and requirements | More |
| --- | --- | --- |
| `describe_scene` | Caption the live head-camera view; needs camera/caption backend. Does not move closer | [Agent guide](AGENT_RUN.md#skill-checklist-interactive) |
| `ground_object` | Request observed 3D geometry for one object in the latest RGB-D view, without navigating; returns geometry or an explicit abstention | [Grounding provenance](#current-view-object-grounding) |
| `send_image` | Capture and attach the head-camera image; Discord delivery when connected | [Discord](discord_bot.md) |
| `take_picture` | Capture a head-camera image locally | [CHAT handlers](../src/emet/agent/tools.py) |
| `query_memory` | Ask about remembered objects/locations; uses graph EQA when enabled, otherwise voxel memory | [Graph memory](graph_memory.md) |
| `query_scene_graph` | Ask GraphEQA where/what questions, or inspect the open-vocabulary graph fallback | [Graph EQA](graph_eqa.md) |
| `list_scene_relations` | List remembered objects and spatial relations | [Dynagraph](dynagraph.md) |
| `send_object_image` | Send a stored object crop from the open-vocabulary graph; not a live image | [CHAT handlers](../src/emet/agent/tools.py) |
| `aim_arm_at` | Localize a named target and aim the arm/wrist camera where supported | [CHAT handlers](../src/emet/agent/tools.py) |
| `take_ee_picture` | Capture the wrist camera after a successful `aim_arm_at`; needs an available wrist stream | [CHAT handlers](../src/emet/agent/tools.py) |

## Current-view object grounding

Call `ground_object(query="silver microwave above the stove")`
when the object is visible and the coordinator needs its geometry. The query
must contain the complete identifying object phrase; do not pass an EQA question
or answer options. The VLM grounder intentionally treats EQA's separate question
context as non-binding, so constraints belong in this explicit object query.
The tool captures a view through the controller and invokes the existing
`ground_query_view` boundary, also used for confirmed EQA views. It does not
require a retrieval candidate, navigate, or manipulate anything.

With `query_memory.grounding_backend: vlm`, the configured VLM selects a region
under the query's visual constraints. Configured RGB-D/SAM2 providers supply
surface masks; `yoloe_sam2` can use query-conditioned YOLOE masks as proposals.
The VLM supplies semantic selection, while valid masked depth pixels, camera
calibration and camera world pose supply the geometry. XYZ is the median of
the admitted object's world points, not coordinates invented by the VLM or a
depth median across the whole frame. Sensor or estimated-depth provenance follows
the configured depth provider. Simulator object poses and ground-truth masks
are not inputs. The separate `yoloe` grounding backend remains configurable;
using the tool does not silently select a different perception model.

Results distinguish `grounded`, `not_grounded`, `observation_unavailable`, and
unsupported configurations. `geometry` contains admitted world XYZ, graph
observation ID, geometry source and point count only on success; `observation`
identifies the captured voxel frame and available acquisition timing. Image
timing does not become a freshness guarantee merely because a frame was added.
The result belongs to its recorded view, which can differ from the next live
camera attachment. Failure clears the controller's current grounding reference;
unverified search points never become a fallback success.

Successful grounding stores the normal query target, so `find_objects` with
the same query can plan an approach from that reference. Route clearance and
manipulation reacquisition remain independent checks. No answer is overridden.

Use a controller supporting current-view grounding (currently LazyGraph) and
enable object admission explicitly. `configs/emet/shared_find_acceptance.yaml`
opts into `graph_object_fusion.enabled` and `use_instance_nodes`; generic defaults
remain unchanged. In LazyGraph, ordinary updates still record viewpoints rather
than stream object instances. Configure `query_memory.grounding_cache_dir` or
`EMET_EQA_EPISODE_DIR` to retain the exact RGB, depth, masks, selection prompts,
raw responses, admission settings and calibration for manual inspection.

## CHAT: navigation and mapping

| Tool | Purpose and requirements | More |
| --- | --- | --- |
| `explore` | Navigate to grow map coverage; requires navigation/mapping executor | [Agent guide](AGENT_RUN.md#skill-checklist-interactive) |
| `scan_environment` | In-place survey to update map and memory; can rotate the robot | [Agent guide](AGENT_RUN.md#lifelong-reload) |
| `find_objects` | Find and navigate toward a named object/location; inspect rejection or cancellation before retrying | [Navigation diagnostics](AGENT_RUN.md#skill-checklist-interactive) |
| `go_home` | Navigate to the starting position using the accumulated map | [CHAT handlers](../src/emet/agent/tools.py) |
| `rotate_base` | Relative base yaw in degrees; positive is left/counterclockwise | [CHAT handlers](../src/emet/agent/tools.py) |
| `face_toward` | Turn toward a remembered object's XY position; does not drive closer | [CHAT handlers](../src/emet/agent/tools.py) |
| `move_forward` | Short mapped forward motion, shortened/refused if blocked or unexplored; requires explicit distance | [CHAT handlers](../src/emet/agent/tools.py) |
| `navigation_diagnostics` | Inspect map coverage, base clearance, and last navigation outcome | [Navigation diagnostics](AGENT_RUN.md#skill-checklist-interactive) |
| `send_map_snapshot` | Render a top-down map and optional last path; Discord/Rerun output when configured | [Rerun](rerun.md) |

## CHAT: manipulation and TAMP

| Tool | Purpose and requirements | More |
| --- | --- | --- |
| `scene_tasks` | Discover supported task candidates and opaque handles; does not certify feasibility | [TAMP API](api/tamp.md#tools) |
| `plan_pick_place` | Plan without acting; accepts a task handle or names, and supported placement-region constraints | [TAMP API](api/tamp.md#tools) |
| `execute_pick_place_plan` | Revalidate and execute a one-shot plan; inspect measured outcome before continuing | [TAMP API](api/tamp.md#preconditions-and-execution-guarantees) |
| `pick_place` | Combined pick/place through the configured manipulation path | [TAMP overview](tamp.md) |
| `set_receptacle_state` | Simulator-assisted open/closed state setting; native physical mode refuses assistance | [Articulation API](api/tamp_articulation.md) |
| `plan_task` | Plan a multi-step goal with a bound provider; native Stretch debug opt-in is available; currently greedy, beam or MCTS | [Task plans](api/tamp_task_plans.md) |
| `execute_task_plan` | Consume a multi-step plan once; verify measured effects and protected goals after each skill | [Task plans](api/tamp_task_plans.md) |
| `hand_over` | Request the configured executor's person-search/navigation/handover behavior | [CHAT handlers](../src/emet/agent/tools.py) |

TAMP tools return versioned JSON. Read `status`, `code`, `data`, and `recovery`;
message wording is not a success signal. Partial/unknown outcomes stop a batch
and require inspection. `hand_over` is a separate executor skill, not a verified
native TAMP recovery primitive. See [current limits and evidence](tamp.md#what-has-been-demonstrated).

## CHAT: communication and session

| Tool | Purpose and requirements | More |
| --- | --- | --- |
| `say` | Speak through the executor's text-to-speech backend | [Agent guide](AGENT_RUN.md) |
| `wave` | Robot-supported waving gesture | [CHAT handlers](../src/emet/agent/tools.py) |
| `nod_head` | Robot-supported nod | [CHAT handlers](../src/emet/agent/tools.py) |
| `shake_head` | Robot-supported head shake | [CHAT handlers](../src/emet/agent/tools.py) |
| `avert_gaze` | Robot-supported look-away gesture | [CHAT handlers](../src/emet/agent/tools.py) |
| `quit` | End the conversation and request robot stop through the executor | [CHAT handlers](../src/emet/agent/tools.py) |

## EQA_EPISODE tools

These are bound to the EQA executor, not exposed as the default CHAT pack.
See [Graph EQA](graph_eqa.md), [Habitat EQA](habitat_eqa.md), and
[mode selection](AGENT_RUN.md#skill-library-vs-orchestrator-modes).

| Tool | Purpose and requirements |
| --- | --- |
| `inspect_graph` | Read detections and stored views without motion; detection handles are proposals, not camera images |
| `explore_frontier` | Grow coverage; optional weak `toward` bias does not replace investigating a detection |
| `investigate` | Navigate around a listed detection/place, capture a fresh view, and assess it |
| `navigate_to_obs` | Compatibility alias for `investigate` |
| `look_around` | Scan in place to refresh map/graph |
| `verify_siglip` | Visual matching plus VLM assessment of the current/stored view; matching alone does not establish answerability |
| `submit_answer` | Answer-mode terminal action, gated by VLM answerability or exhausted round budget |
| `finish` | Explore-mode terminal action, gated by exhausted frontiers or exploration budget |

`submit_answer` and `finish` belong to different submodes. The conceptual pairs
`scan_environment`/`look_around` and `explore`/`explore_frontier` share intent but
have different orchestration contracts. Preserve their names in traces.

## Extending the index

Add or change tool metadata in [skill specs](../src/emet/agent/skills/specs.py),
wire the relevant handler, and update this page with its requirements and docs.
The [skill-pack tests](../src/test/agent/test_skill_packs.py) check mode membership;
[agent prompt/tool tests](../src/test/agent/test_agent_prompt_and_tools.py) cover
CHAT wiring. CLI commands, internal planner functions, and tools on unmerged
branches are not additional tools available to the current LLM pack.
