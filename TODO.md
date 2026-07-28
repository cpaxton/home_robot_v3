# Living TODOs (small / near-term)

Short checklist for agent/hardware polish that is not worth a full plan doc yet.
Strike through or move to a PR when done.

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

- [x] Map-clip `move_forward` (including 0.1 m); refuse when map empty/blank.
- [x] Empty cloud guard in `list_objects_in_an_image` (navigate crash).
- [x] Clearance-aware A* + abort on waypoint timeout (prefer open space; do not raise `dilate_obstacle_size` as the primary fix).

## Docs / ops

- [ ] Document Herman Discord happy path with `EMET_BASE_ROTATE_ONLY` + `EMET_ALLOW_SDPA_ATTN` / flash-attn in hardware checklist (partially done).

## Manipulation / MolmoSpaces + rby1 (PR #83 follow-ups)

Offline units + scripted table smokes exist; these are the remaining **real / integration** and product gaps.

### Config / agent path
- [ ] **Wire `agent.manip_mode` / `manip_collision` / `manip_planner` into `DynamemTaskExecutor`**: YAML / `--set agent.manip_mode=kinematic` today only live on `AgentSectionConfig`; executor reads `parameters.get("agent")` (mapping explore block). Until fixed, operators must use **`EMET_MANIP_MODE`** (and friends). Docs currently overclaim YAML.
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
