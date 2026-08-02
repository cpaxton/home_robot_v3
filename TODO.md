# Living TODOs (small / near-term)

Short checklist for agent/hardware polish that is not worth a full plan doc yet.
Strike through or move to a PR when done.

## Prompt / information flow (THIS BRANCH)

- [ ] **One fact, one line — de-duplicate EQA memory blocks**: SCENE_GRAPH node lines, CONFIRMED_MEMORY (`PRESENT`/`nearest:`), and IMAGE_DESCRIPTIONS currently restate the same object 2–3× in different formats. Merge into one serialization: node line tagged with room + confirmed status + image ref (`Node 3 (kitchen): sink at (…) [Image 2] present`), and drop/trim the redundant blocks.
- [ ] **Room context never reaches the answer VLM**: router stamps rooms on clusters (room_clusters.py), but `query_answer` (graph_memory.py:3943) builds no room line — the answer model re-derives rooms from bare XYZ. Plumb `format_rooms_line` / `current_room` and per-node `room=` tags into `to_string` so "kitchen vs bathroom" evidence is explicit.
- [ ] **EQA answer format → JSON**: `parse_answer` (graph_memory.py:3273) regex-scrapes lowercased `Reasoning:/Answer:/Confidence:/Action:` fields — brittle to caption-runaway, case loss, mid-stream truncation. Convert to the same JSON contract as chat/router (`{"answer", "confidence", "action", …}`); keep `assistant_prefill` for decode discipline.
- [ ] **Unified token budget for the EQA prompt**: caps are independent today (48 nodes / 4 images / 4 history / CONFIRMED_MEMORY unbounded). Add one budget (~2.5k text tokens) with a fixed truncation order (HISTORY → CONFIRMED_MEMORY → SCENE_GRAPH edges → labels) so prompt size is predictable regardless of graph size.
- [ ] **Router prompt hygiene**: canonical/LLM format blocks in agentic_tools.py duplicate rules (investigate-vs-explore, in_target_area). Single source of truth + a unit test asserting `build_graph_eqa_system_prompt` byte-stability (prefix-KV cache depends on it).
- [ ] **HISTORY loop risk**: replays prior model outputs back (only Caption-strip mitigates). Keep only outcomes — `Nav_result`, salvage, letter — as one-line summaries instead of raw replayed outputs.
- [ ] **CHAT `_FORMAT_BLOCK` is ~90 lines of routing edge cases** (prompt.py): consider tiered prompt (short default; detailed hints appended only for 4B-class routers) and measure system-prompt chars/tokens with and without hints.
- [ ] **describe_scene grounding**: currently caption + optional graph labels appended ad hoc (`describe_head_camera_scene_text`, controller_dynamem.py:1049). Define one consistent grounding format shared with `query_scene_graph` so the chat VLM sees the same memory vocabulary as EQA.

## Embodied agent / Herman

- [x] **One open-vocab scene graph (not two builders)**: CHAT uses mutually exclusive `agent.memory_backend` (`dynagraph` | `graph_eqa` | `open_vocab` | `dynamem`). Discord presets → Dynagraph memory plug-in only; GraphEQA baseline left frozen for paper. Lifelong save/load writes the active plug-in only.
- [ ] **Arm IK “closer look”**: point the wrist / EE camera at a named object (or image region), then capture.
  - Stub tool today: `aim_arm_at` in chat pack (returns not-implemented + suggests `describe_scene`).
  - Needs: Mars/Stretch IK + wrist frame + safety (collision, joint limits); then `take_ee_picture` only *after* aim.
  - Until then: “closer look / inspect X” → **change viewpoint** (rotate / small drive) then `describe_scene` / `send_image` (head), never raw `take_ee_picture`.
- [ ] **Chat verify ≈ EQA look**: prefer `face_toward` + `describe_scene` (not blind +45°). Full EQA-style `navigate_to_obs` + `verify_siglip` in CHAT still optional later.
- [x] **`emet run` + `--connection`**: wrapper no longer injects default `--robot-ip 127.0.0.1`, so `--connection herman` resolves the profile host.
- [ ] **Interruptible explore**: Discord messages queue until `explore` finishes; drain `unified_input_queue` mid-nav.
- [ ] **VRAM for Discord house chat**: default Mars preset loads tool LLM + EQA 8B VL + SigLIP (+ DA3). Easy OOM on 24 GiB when map update / DA3 overlaps generate. Mitigations: `--onboard-da3`, share one VL (`--llm qwen3-vl-eqa --share-memory-vllm`), or lazy-unload captioner between turns. (Dual OV+GE builders removed; still watch two VLMs.)

## Mapping / safety

- [x] Map-clip `move_forward` (including 0.1 m); refuse when map empty/blank.
- [x] Empty cloud guard in `list_objects_in_an_image` (navigate crash).
- [x] Clearance-aware A* + abort on waypoint timeout (prefer open space; do not raise `dilate_obstacle_size` as the primary fix).

## Memory stack (voxel + graph) — review backlog

- [ ] **Drift correction / relocalization** (long real-world runs): voxel map is robot-start-relative with a fixed `grid_origin`; only `--refine-start` applies an SE(2) fudge at checkpoint load. Plan ICP / pose-graph correction against the voxel map or re-anchor graph nodes during operation.
- [ ] **Pin depth units per source + runtime assert**: `sensor_graph_builder.py:80` assumes meters (`scaling=1.0`) while `core/interfaces.py:196` defaults to `1e-3` (mm). Verify per source (Stretch d405, DA3, Habitat) and assert once at attach.
- [ ] **Robot self-filter lost in DynaMem `add()`**: `voxel_dynamem.py:1109` copies base `voxel.py:403` minus the URDF mesh self-filter — the robot may see itself on real hardware. Reintroduce or document why it is off.
- [ ] **Split `query_answer` (462 lines) + de-dup renumber/rebuild blocks**: prompt assembly → `build_eqa_prompt(…)`, execution → `run_eqa_prompt(commands)`, gates → `finalize_eqa_answer(parsed)`. The renumber+rebuild block in `maintain`/`_drop_nodes_near`/`absorb_object_node` is copy-pasted 5×.
- [ ] **Rooms as first-class nodes**: cluster stamps are recomputed from nodes+edges every refresh; persist room id / name / bounds on nodes and in `graph.json` exports (in-flight commits already move this way).
- [ ] **Bound voxel memory growth**: `observations` never pruned, `semantic_memory` keeps every subsampled point, pickles dump everything. Trim frames + downsample old semantic points on a schedule.
- [ ] **Prefix-KV timeout leaves a live CUDA worker** (`qwen3_vl_client.py:549-569` raises while the generate thread keeps running): next generate can race. Serialize on a lock or hard-kill the worker.
- [ ] **Silent exception hygiene**: ~23 `except Exception` in graph_memory.py, bare `except:` at voxel_dynamem.py:1749 — at least `_logger.debug` with node/obs context.
- [ ] **Dead code sweep**: `Qwen3VLClient = Qwen35VLClient` alias (qwen_client.py:738); commented heuristics voxel_map_dynamem.py:178-251; `get_change_events` / `alternate_nav_target_for_failed_action` (graph_memory.py) unused.

## Docs / ops

- [ ] Document Herman Discord happy path with `EMET_BASE_ROTATE_ONLY` + `EMET_ALLOW_SDPA_ATTN` / flash-attn in hardware checklist (partially done).

## Manipulation / MolmoSpaces + rby1 (PR #83 follow-ups)

Offline units + scripted table smokes exist; these are the remaining **real / integration** and product gaps.

### Config / agent path
- [x] **Wire `agent.manip_mode` / `manip_collision` / `manip_planner` into `DynamemTaskExecutor`**: `emet run agent` merges the finalized chat `agent:` section (YAML / `--set agent.manip_*`) into `parameters["agent"]` before executor init; `EMET_MANIP_*` env vars still win. (Manip-only top-level `agent:` blocks are now recognized as chat-agent sections by the loader.)
- [ ] **Stretch MuJoCo pick/place default**: with visual-servo off, any sim advertising `sim_set_body_pose` takes **GT teleport** instead of AnyGrasp / agent grasp. Confirm paper/OVMM Stretch numbers still intend `-V` when comparing to old path; document the default loudly in AGENT_RUN / molmospaces.
- [ ] **Stretch / AnyGrasp `_pickup` / `_place` always return True**: `DynamemTaskExecutor` only sets `_last_exec_ok=False` on sim teleport/kinematic failures. The Stretch visual-servo / `agent.manipulate` / `agent.place` path still returns success unconditionally, so agent `pick_place` tool summaries and scripted `_last_exec_ok` checks never see Stretch grasp/place failures. Propagate real success from the Stretch manip stack (or at least catch known failure signals).

### Real tests (not yet green on every machine)
- [ ] **MolmoSpaces ithor + rby1** kinematic and teleport smokes when `.venv-molmospaces` + assets are warm (`scripted_sim_pick_place` / `scripted_molmo_grasp_mp` / `scripted_tamp_pick_place` with `--sim configs/sim/molmospaces_ithor_train_0.yaml`).
- [ ] **Molmo kinematic approach frame**: table kinematic is green; Molmo iTHOR bowl→microwave still `pregrasp_ik_failed` (grasp_err≈0.60) — approach XYT / nav-world vs MuJoCo base disagree after base teleport (`docs/plans/2026-07-23_molmospaces_rby1_manip_review.md`).
- [ ] **OVMM full** episode `molmo_ithor_rby1_s2_bowl_pp` with `manip_mode=sim` (find + teleport pick/place).
- [ ] **Robocasa / Stretch** pick-place smoke with and without `--visual-servo` to lock the teleport-vs-servo behavior.
- [ ] **CI / overnight**: mark or gate the above under `RUN_MOLMOSPACES_TESTS` / sim markers so agents use `emet test --no-sim` for the offline pack only.

### Motion / grasp quality
- [ ] **IK orientation**: MuJoCo IK is **position-only**; Molmo grasp quats set approach standoff but are not enforced at the EE. Need orientation (or constrained) IK before claiming grasp fidelity.
- [ ] **Kinematic gripper cmd failures**: `_set_gripper` open/close now logs a warning on ZMQ/actuator errors but still continues (attach/detach is the real hold). Decide whether gripper failure should fail the grasp/place result before attach, or verify finger joints after cmd.
- [ ] **GT body / receptacle disambiguation**: teleport + kinematic still take first category substring match (`receps[0]` / object query). Prefer nearest-to-nav-`point` (or explicit `object_gt_body`) when multiple bowls/microwaves exist.
- [ ] **Voxel / AABB arm collision in live agent**: kinematic path supports `EMET_MANIP_COLLISION=voxel|aabb`; default is `none`. Dogfood on a cluttered Molmo scene and turn voxel on by default only if latency is OK.
- [ ] **Instance-memory `plan_to_frontier`**: still one-sample → one plan (often RRT). Dynamem explore already uses multi-goal A*; port the same top-K → `goals=` pattern if SVM/instance agents stay in use.

### Hardware / product
- [ ] **Real-robot manip** (Stretch / Mars): no GT `sim_set_body_pose` — keep visual-servo / AnyGrasp; add a no-LLM tool-sequence test that asserts we **do not** enter teleport when `is_simulation` is false.
- [ ] **`aim_arm_at` / EE look** (see Embodied agent above): still stub; needed for “closer look” after pick failures.
- [ ] **Paper figures**: keep regenerating `manip_figures` / chase-cam MP4s from scripted TAMP on the scene used in the paper; check in paths under `~/runs/emet/` only (not repo blobs).
